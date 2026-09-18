"""Phase 5 - the route map: a tilted, zoomed-in view that follows the trip.

Rather than one fixed basemap of the whole route, each clip gets a view centred
on where it was taken, zoomed far enough in that town and mountain names are
legible. When the trip moves to a new place the view flies there over a second
or two, becoming more opaque while it travels and settling back afterwards, so
a location change reads as movement instead of a jump.

The map is drawn flat, then put through a perspective warp so it sits like a
tilted ground plane. The route and the travelled track are drawn *before* the
warp, so they lie on that plane; the position marker is drawn *after* it, as an
upright pin.

Tiles come from OpenTopoMap by default (hillshading and peak names suit a trip
in the mountains) and are cached on disk, so re-runs and neighbouring views
cost no new requests.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from models import Timeline

log = logging.getLogger(__name__)

MIN_GPS_POINTS = 3
TILE_SIZE = 256
USER_AGENT = "trip-video-generator/1.0 (personal trip recap; contact: local user)"

ATTRIBUTION = {
    "topo": "© OpenTopoMap (CC-BY-SA) · © OpenStreetMap contributors",
    "osm": "© OpenStreetMap contributors",
}

TILE_SOURCES = {
    # hillshaded contours, names peaks and their heights - best for mountains
    "topo": ["https://a.tile.opentopomap.org/{z}/{x}/{y}.png",
             "https://b.tile.opentopomap.org/{z}/{x}/{y}.png",
             "https://c.tile.opentopomap.org/{z}/{x}/{y}.png"],
    # plain OpenStreetMap: denser street and place labels, no relief
    "osm": ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
}


@dataclass
class MapConfig:
    size: int = 600                 # rendered square, before the tilt
    zoom: int = 12                  # ~ town + valley scale; labels stay readable
    tiles: str = "topo"
    out_dir: Path = Path("./tmp/map_frames")
    cache_dir: Path | None = None   # defaults to <out_dir>/../tile_cache

    # the fly-to animation
    animate: bool = True
    move_seconds: float = 1.6
    settle_seconds: float = 0.7
    fps: float = 12.0
    move_zoom_out: float = 1.15     # pull back a little mid-flight

    # opacity: low while parked, high while travelling
    base_opacity: float = 0.60   # readable at rest ...
    move_opacity: float = 0.95   # ... and near-solid while travelling

    # 3d look
    tilt: float = 0.30              # 0 = flat, ~0.5 = steeply raked
    attribution: bool = True        # OSM-derived tiles need crediting on screen
    route_color: tuple[int, int, int, int] = (70, 70, 78, 190)
    done_color: tuple[int, int, int, int] = (235, 92, 48, 255)
    line_width: int = 4
    marker_radius: int = 11
    request_timeout: float = 12.0


@dataclass
class MapTrack:
    """The map overlay for one clip: one still, or a short animation."""

    frames: list[Path] = field(default_factory=list)
    fps: float = 12.0
    animated: bool = False

    @property
    def first(self) -> Path | None:
        return self.frames[0] if self.frames else None

    @property
    def last(self) -> Path | None:
        return self.frames[-1] if self.frames else None


# ------------------------------------------------------ reverse geocoding

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
PLACE_KEYS = ("city", "town", "village", "hamlet", "municipality", "suburb",
              "county", "state_district", "state", "country")


def _place_label(address: dict) -> str | None:
    """A short, human label: the settlement if there is one, else the region."""
    for key in PLACE_KEYS:
        value = address.get(key)
        if value:
            return str(value).strip().split(",")[0].strip()
    return None


def _reverse(client, lat: float, lon: float, zoom: int) -> str | None:
    """One Nominatim reverse lookup. `zoom` picks the granularity: 12 is
    neighbourhood-ish, 10 lands on the town or municipality people name."""
    response = client.get(NOMINATIM_URL, params={
        "lat": lat, "lon": lon, "format": "jsonv2", "zoom": zoom, "addressdetails": 1})
    response.raise_for_status()
    return _place_label((response.json() or {}).get("address") or {})


class _PlaceCache:
    """Disk-backed {query key -> label}, so re-runs make no requests at all."""

    def __init__(self, path: Path | None):
        import json

        self.path = Path(path) if path else None
        self.data: dict[str, str] = {}
        self.dirty = False
        if self.path and self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self.data = {}

    def get(self, key: str) -> str | None:
        return self.data.get(key)

    def put(self, key: str, value: str | None) -> None:
        if value:
            self.data[key] = value
            self.dirty = True

    def save(self) -> None:
        import json

        if not (self.path and self.dirty):
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            log.warning("could not write place cache: %s", exc)


def resolve_places(timeline: Timeline, cache_path: Path | None = None,
                   precision: int = 2, pause: float = 1.1) -> dict[int, str]:
    """Fill `entry.place` and return one canonical name per GPS cluster.

    Two granularities: each clip gets its local name (useful context for the
    narrative model), while each Phase 2 cluster is looked up once at its
    centroid at a coarser zoom - that is what the on-screen header shows, so a
    single stop reads as "Interlaken" rather than cycling through Wilderswil,
    Unterseen and Matten bei Interlaken.

    Rate limited to Nominatim's one-request-per-second policy. Any failure just
    leaves names unset, and the header is skipped.
    """
    located = [e for e in timeline.selected if e.has_gps()]
    if not located:
        return {}

    cache = _PlaceCache(cache_path)
    groups: dict[int, list] = {}
    for entry in located:
        groups.setdefault(entry.group if entry.group is not None else -1, []).append(entry)

    jobs: list[tuple[str, float, float, int]] = []
    for entry in located:
        key = f"{round(entry.gps.lat, precision)},{round(entry.gps.lon, precision)}@12"
        if cache.get(key) is None and key not in [j[0] for j in jobs]:
            jobs.append((key, entry.gps.lat, entry.gps.lon, 12))
    centroids: dict[int, str] = {}
    for group, members in groups.items():
        lat = sum(e.gps.lat for e in members) / len(members)
        lon = sum(e.gps.lon for e in members) / len(members)
        key = f"g{group}:{round(lat, 3)},{round(lon, 3)}@10"
        centroids[group] = key
        if cache.get(key) is None:
            jobs.append((key, lat, lon, 10))

    if jobs:
        log.info("looking up %d place name(s) via Nominatim", len(jobs))
        try:
            import httpx

            with httpx.Client(headers={"User-Agent": USER_AGENT, "Accept-Language": "en"},
                              timeout=20.0, follow_redirects=True) as client:
                for i, (key, lat, lon, zoom) in enumerate(jobs):
                    try:
                        if i:
                            time.sleep(pause)
                        cache.put(key, _reverse(client, lat, lon, zoom))
                    except Exception as exc:  # noqa: BLE001 - offline, 429, junk
                        log.warning("place lookup failed (%s) - continuing without it", exc)
                        break
        except ImportError:
            log.warning("httpx unavailable - skipping place names")
    cache.save()

    resolved = 0
    for entry in located:
        label = cache.get(f"{round(entry.gps.lat, precision)},{round(entry.gps.lon, precision)}@12")
        if label:
            entry.place = label
            resolved += 1

    headers: dict[int, str] = {}
    for group, key in centroids.items():
        label = cache.get(key)
        if not label:  # fall back to whatever its clips agreed on
            names = [e.place for e in groups[group] if e.place]
            label = max(set(names), key=names.count) if names else None
        if label:
            headers[group] = label
    log.info("place names: %d/%d clips, %d location header(s): %s",
             resolved, len(located), len(headers), ", ".join(headers.values()) or "-")
    return headers


# ----------------------------------------------------------- projection

def lon_to_x(lon: float, zoom: int) -> float:
    return ((lon + 180.0) / 360.0) * (2 ** zoom) * TILE_SIZE


def lat_to_y(lat: float, zoom: int) -> float:
    lat = max(min(lat, 85.05112878), -85.05112878)
    rad = math.radians(lat)
    return (1 - math.log(math.tan(rad) + 1 / math.cos(rad)) / math.pi) / 2 * (2 ** zoom) * TILE_SIZE


def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(h))


# --------------------------------------------------------------- tiles

class TileFetcher:
    """Fetches and caches map tiles; a failed tile becomes blank, never fatal."""

    def __init__(self, cfg: MapConfig):
        self.cfg = cfg
        self.templates = TILE_SOURCES.get(cfg.tiles, TILE_SOURCES["topo"])
        self.cache_dir = Path(cfg.cache_dir or Path(cfg.out_dir).parent / "tile_cache") / cfg.tiles
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory: dict[tuple[int, int, int], Image.Image] = {}
        self.blank = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (238, 236, 231))
        self.client = None
        self.failures = 0
        self._last_request = 0.0

    def _http(self):
        if self.client is None:
            import httpx

            self.client = httpx.Client(headers={"User-Agent": USER_AGENT},
                                       timeout=self.cfg.request_timeout, follow_redirects=True)
        return self.client

    def tile(self, zoom: int, x: int, y: int) -> Image.Image:
        span = 2 ** zoom
        if y < 0 or y >= span:
            return self.blank
        x %= span
        key = (zoom, x, y)
        if key in self.memory:
            return self.memory[key]

        path = self.cache_dir / str(zoom) / str(x) / f"{y}.png"
        if path.exists():
            try:
                with Image.open(path) as img:
                    tile = img.convert("RGB")
                self.memory[key] = tile
                return tile
            except Exception:  # noqa: BLE001 - corrupt cache entry, refetch
                path.unlink(missing_ok=True)

        url = self.templates[(x + y) % len(self.templates)].format(z=zoom, x=x, y=y)
        try:
            gap = time.monotonic() - self._last_request
            if gap < 0.08:  # be a polite client on community tile servers
                time.sleep(0.08 - gap)
            response = self._http().get(url)
            self._last_request = time.monotonic()
            response.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
            with Image.open(path) as img:
                tile = img.convert("RGB")
        except Exception as exc:  # noqa: BLE001 - offline, 404, rate limited
            self.failures += 1
            if self.failures <= 2:
                log.warning("tile %s/%s/%s unavailable (%s)", zoom, x, y, exc)
            tile = self.blank
        self.memory[key] = tile
        return tile

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None


def render_viewport(fetcher: TileFetcher, center: tuple[float, float], zoom: int,
                    size: int) -> tuple[Image.Image, tuple[float, float]]:
    """Stitch the tiles covering a square view. Returns the image and the world
    pixel coordinate of its top-left corner, so geography can be plotted on it."""
    cx, cy = lon_to_x(center[1], zoom), lat_to_y(center[0], zoom)
    left, top = cx - size / 2, cy - size / 2
    x0, y0 = math.floor(left / TILE_SIZE), math.floor(top / TILE_SIZE)
    x1, y1 = math.floor((left + size) / TILE_SIZE), math.floor((top + size) / TILE_SIZE)

    canvas = Image.new("RGB", ((x1 - x0 + 1) * TILE_SIZE, (y1 - y0 + 1) * TILE_SIZE))
    for tx in range(x0, x1 + 1):
        for ty in range(y0, y1 + 1):
            canvas.paste(fetcher.tile(zoom, tx, ty), ((tx - x0) * TILE_SIZE, (ty - y0) * TILE_SIZE))

    offset_x, offset_y = left - x0 * TILE_SIZE, top - y0 * TILE_SIZE
    view = canvas.crop((int(offset_x), int(offset_y), int(offset_x) + size, int(offset_y) + size))
    return view.convert("RGBA"), (left, top)


# ------------------------------------------------------------ 3d tilt

def _homography(src: list[tuple[float, float]], dst: list[tuple[float, float]]) -> np.ndarray:
    """3x3 matrix mapping the four `src` corners onto `dst`."""
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    a = np.asarray(rows, dtype=float)
    b = np.asarray([c for point in dst for c in point], dtype=float)
    solution = np.linalg.lstsq(a, b, rcond=None)[0]
    return np.append(solution, 1.0).reshape(3, 3)


def _project(matrix: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    x, y, w = matrix @ np.asarray([point[0], point[1], 1.0])
    return (x / w, y / w) if w else (x, y)


def tilt_quad(size: int, tilt: float) -> list[tuple[float, float]]:
    """Where the flat map's corners land once raked back into a ground plane."""
    inset = size * tilt * 0.5          # top edge pulled in -> perspective
    squash = size * tilt * 0.22        # and pushed down -> foreshortening
    return [(inset, squash), (size - inset, squash), (size, size), (0.0, size)]


def apply_tilt(flat: Image.Image, cfg: MapConfig) -> tuple[Image.Image, np.ndarray]:
    """Warp the flat map into the tilted plane; also return the forward matrix."""
    size = flat.size[0]
    square = [(0.0, 0.0), (float(size), 0.0), (float(size), float(size)), (0.0, float(size))]
    quad = tilt_quad(size, cfg.tilt)
    if cfg.tilt <= 0:
        return flat, np.eye(3)

    forward = _homography(square, quad)
    inverse = _homography(quad, square)   # PIL samples output -> input
    coeffs = (inverse / inverse[2, 2]).flatten()[:8]
    warped = flat.transform((size, size), Image.PERSPECTIVE, coeffs,
                            Image.BICUBIC, fillcolor=(0, 0, 0, 0))

    # distance haze: fade the far edge so the plane recedes
    haze = Image.new("L", (size, size), 255)
    haze_draw = ImageDraw.Draw(haze)
    horizon = int(size * cfg.tilt * 0.5)
    for y in range(max(horizon, 1)):
        haze_draw.line([(0, y), (size, y)], fill=int(255 * (y / max(horizon, 1)) ** 0.7))
    alpha = warped.getchannel("A").point(lambda v: v)
    warped.putalpha(Image.fromarray(np.minimum(np.array(alpha), np.array(haze)).astype("uint8")))
    return warped, forward


# --------------------------------------------------------------- route

def route_points(timeline: Timeline, ordered_paths: list[str]) -> list[tuple[float, float] | None]:
    points: list[tuple[float, float] | None] = []
    for path in ordered_paths:
        entry = timeline.by_path(path)
        points.append((entry.gps.lat, entry.gps.lon) if entry is not None and entry.has_gps() else None)
    return points


def interpolate(points: list[tuple[float, float] | None]) -> list[tuple[float, float]]:
    """Fill gaps linearly between the nearest known fixes; extend at the ends."""
    known = [i for i, p in enumerate(points) if p is not None]
    if not known:
        return []
    filled: list[tuple[float, float]] = []
    for i, point in enumerate(points):
        if point is not None:
            filled.append(point)
            continue
        before = max((k for k in known if k < i), default=None)
        after = min((k for k in known if k > i), default=None)
        if before is None:
            filled.append(points[after])            # type: ignore[arg-type]
        elif after is None:
            filled.append(points[before])           # type: ignore[arg-type]
        else:
            lat0, lon0 = points[before]             # type: ignore[misc]
            lat1, lon1 = points[after]              # type: ignore[misc]
            f = (i - before) / (after - before)
            filled.append((lat0 + (lat1 - lat0) * f, lon0 + (lon1 - lon0) * f))
    return filled


def _draw_route(view: Image.Image, origin: tuple[float, float], zoom: int,
                coords: list[tuple[float, float]], upto: int, cfg: MapConfig) -> None:
    """Draw the whole route in grey and the travelled part in colour, in place."""
    draw = ImageDraw.Draw(view)
    pixels = [(lon_to_x(lon, zoom) - origin[0], lat_to_y(lat, zoom) - origin[1]) for lat, lon in coords]
    if len(pixels) > 1:
        draw.line(pixels, fill=cfg.route_color, width=cfg.line_width, joint="curve")
    if upto > 0:
        draw.line(pixels[: upto + 1], fill=cfg.done_color, width=cfg.line_width + 2, joint="curve")


def _draw_pin(view: Image.Image, at: tuple[float, float], cfg: MapConfig) -> None:
    """An upright pin at the warped position - it stands on the tilted plane.

    Drawn after the warp on purpose: a pin flattened into the ground plane
    would read as a smear, while an upright one sells the 3d.
    """
    draw = ImageDraw.Draw(view)
    x, y = at
    r = cfg.marker_radius
    height = r * 2.8
    # shadow puddle on the ground, then the stalk, then the head
    draw.ellipse([x - r * 1.1, y - r * 0.4, x + r * 1.1, y + r * 0.4], fill=(0, 0, 0, 110))
    draw.line([(x + 1, y + 1), (x + 1, y - height + 1)], fill=(0, 0, 0, 120), width=4)
    draw.line([(x, y), (x, y - height)], fill=(255, 255, 255, 240), width=3)
    draw.ellipse([x - r - 3, y - height - r - 3, x + r + 3, y - height + r + 3], fill=(255, 255, 255, 245))
    draw.ellipse([x - r, y - height - r, x + r, y - height + r], fill=cfg.done_color)
    draw.ellipse([x - r * 0.38, y - height - r * 0.38, x + r * 0.38, y - height + r * 0.38],
                 fill=(255, 255, 255, 230))


def _frame(fetcher: TileFetcher, center: tuple[float, float], zoom: int,
           coords: list[tuple[float, float]], upto: int, opacity: float,
           cfg: MapConfig) -> Image.Image:
    """One finished map frame: tiles, route, tilt, pin, opacity, rounded corners."""
    flat, origin = render_viewport(fetcher, center, zoom, cfg.size)
    _draw_route(flat, origin, zoom, coords, upto, cfg)

    tilted, forward = apply_tilt(flat, cfg)
    here = (lon_to_x(coords[upto][1], zoom) - origin[0], lat_to_y(coords[upto][0], zoom) - origin[1])
    _draw_pin(tilted, _project(forward, here), cfg)

    if cfg.attribution:
        _draw_attribution(tilted, cfg)
    if opacity < 1.0:
        alpha = tilted.getchannel("A").point(lambda v: int(v * opacity))
        tilted.putalpha(alpha)
    return tilted


def _draw_attribution(view: Image.Image, cfg: MapConfig) -> None:
    """Credit the tile source, as the OSM and OpenTopoMap licences require."""
    from PIL import ImageFont

    text = ATTRIBUTION.get(cfg.tiles, ATTRIBUTION["osm"])
    size = max(9, int(cfg.size * 0.019))
    font = None
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        if Path(path).exists():
            try:
                font = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
    if font is None:
        try:
            font = ImageFont.load_default(size=size)
        except TypeError:
            font = ImageFont.load_default()
    draw = ImageDraw.Draw(view)
    x, y = int(cfg.size * 0.02), cfg.size - size - int(cfg.size * 0.022)
    draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0, 150))
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 225))


def _ease(t: float) -> float:
    """Smooth start and stop (cosine ease-in-out)."""
    return 0.5 - 0.5 * math.cos(math.pi * max(0.0, min(1.0, t)))


# -------------------------------------------------------------- frames

def render_frames(timeline: Timeline, ordered_paths: list[str],
                  cfg: MapConfig | None = None) -> list[MapTrack]:
    """One MapTrack per clip. An empty list means "no map for this trip"."""
    cfg = cfg or MapConfig()
    raw = route_points(timeline, ordered_paths)
    fixes = sum(1 for p in raw if p is not None)
    if fixes < MIN_GPS_POINTS:
        log.warning("only %d clip(s) have GPS - skipping the map", fixes)
        return []

    coords = interpolate(raw)
    if len({(round(a, 4), round(b, 4)) for a, b in coords}) < 2:
        log.warning("all clips share one location - skipping the map")
        return []

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fetcher = TileFetcher(cfg)
    # a move worth animating: far enough that the view actually travels
    move_threshold_km = (cfg.size / 2) * (156543.03 * math.cos(math.radians(coords[0][0]))
                                          / (2 ** cfg.zoom)) / 1000.0 * 0.6

    tracks: list[MapTrack] = []
    try:
        for i, center in enumerate(coords):
            previous = coords[i - 1] if i > 0 else None
            distance = _haversine_km(previous, center) if previous else 0.0
            moving = cfg.animate and previous is not None and distance > move_threshold_km

            if not moving:
                image = _frame(fetcher, center, cfg.zoom, coords, i, cfg.base_opacity, cfg)
                path = out_dir / f"map_{i:04d}.png"
                image.save(path)
                tracks.append(MapTrack([path], cfg.fps, False))
                continue

            clip_dir = out_dir / f"map_{i:04d}"
            clip_dir.mkdir(exist_ok=True)
            frames: list[Path] = []
            n_move = max(2, int(cfg.move_seconds * cfg.fps))
            n_settle = max(1, int(cfg.settle_seconds * cfg.fps))

            for step in range(n_move):
                f = _ease(step / (n_move - 1))
                here = (previous[0] + (center[0] - previous[0]) * f,
                        previous[1] + (center[1] - previous[1]) * f)
                # pull back over the middle of the flight, then come back in
                zoom = cfg.zoom - cfg.move_zoom_out * math.sin(math.pi * f)
                opacity = cfg.base_opacity + (cfg.move_opacity - cfg.base_opacity) * min(1.0, f * 4)
                image = _frame(fetcher, here, int(round(zoom)), coords,
                               i if f > 0.5 else max(0, i - 1), opacity, cfg)
                path = clip_dir / f"f{step:03d}.png"
                image.save(path)
                frames.append(path)

            for step in range(n_settle):   # ease the opacity back down on arrival
                f = _ease((step + 1) / n_settle)
                opacity = cfg.move_opacity + (cfg.base_opacity - cfg.move_opacity) * f
                image = _frame(fetcher, center, cfg.zoom, coords, i, opacity, cfg)
                path = clip_dir / f"s{step:03d}.png"
                image.save(path)
                frames.append(path)

            tracks.append(MapTrack(frames, cfg.fps, True))
    finally:
        fetcher.close()

    animated = sum(1 for t in tracks if t.animated)
    log.info("rendered %d map views (%d animated moves, %d frames total) at zoom %d from %s tiles",
             len(tracks), animated, sum(len(t.frames) for t in tracks), cfg.zoom, cfg.tiles)
    if fetcher.failures:
        log.warning("%d tile request(s) failed - parts of the map may be blank", fetcher.failures)
    return tracks
