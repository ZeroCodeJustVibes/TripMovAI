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

# Terrarium-encoded elevation tiles (AWS Open Data, no key needed):
# height in metres = (R * 256 + G + B / 256) - 32768
DEM_TEMPLATE = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
DEM_MAX_ZOOM = 14

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
    move_grow: float = 2.2       # and briefly this many times its parked size

    # 3d look: the terrain is shaded and raised from real elevation data, while
    # the map itself stays flat and north-up (tilt defaults to off)
    relief: float = 1.0             # 0 disables hillshading and the height bulge
    relief_lift: float = 0.030      # peak displacement, as a fraction of the view
    sun_azimuth: float = 315.0      # north-west light, the cartographic default
    sun_altitude: float = 42.0
    tilt: float = 0.0               # >0 rakes the whole plane back into perspective
    attribution: bool = True        # OSM-derived tiles need crediting on screen
    route_color: tuple[int, int, int, int] = (70, 70, 78, 190)
    done_color: tuple[int, int, int, int] = (235, 92, 48, 255)
    line_width: int = 4
    marker_radius: int = 11
    request_timeout: float = 12.0


@dataclass
class MapTrack:
    """The map overlay for one clip: one still, or a short animation.

    `scales` is parallel to `frames`: 1.0 is the parked size, larger values are
    the brief swell while the view is travelling to a new place.
    """

    frames: list[Path] = field(default_factory=list)
    fps: float = 12.0
    animated: bool = False
    scales: list[float] = field(default_factory=list)

    def scale_at(self, index: int) -> float:
        if not self.scales:
            return 1.0
        return self.scales[min(index, len(self.scales) - 1)]

    @property
    def max_scale(self) -> float:
        return max(self.scales) if self.scales else 1.0

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

    detail: dict[int, str] = {}
    for entry in located:
        label = cache.get(f"{round(entry.gps.lat, precision)},{round(entry.gps.lon, precision)}@12")
        if label:
            entry.extra["place_detail"] = label
            detail[id(entry)] = label

    headers: dict[int, str] = {}
    for group, key in centroids.items():
        label = cache.get(key)
        if not label:  # fall back to whatever its clips agreed on
            names = [detail.get(id(e)) for e in groups[group]]
            names = [n for n in names if n]
            label = max(set(names), key=names.count) if names else None
        if label:
            headers[group] = label

    # `place` is the canonical name for the whole cluster, so neighbouring
    # clips do not flicker between "Matten bei Interlaken" and "Interlaken".
    # The fine-grained name stays in extra for anyone who wants it.
    resolved = 0
    for entry in located:
        entry.place = headers.get(entry.group) or detail.get(id(entry))
        if entry.place:
            resolved += 1
    log.info("place names: %d/%d clips across %d location(s): %s",
             resolved, len(located), len(headers),
             ", ".join(dict.fromkeys(headers.values())) or "-")
    return headers


def resolve_elevations(timeline: Timeline, cfg: MapConfig) -> int:
    """Fill `entry.altitude` from the elevation tiles where EXIF had none.

    Phones record GPS altitude inconsistently, and the terrain tiles are
    already cached for the map, so the ground height is essentially free.
    """
    missing = [e for e in timeline.selected if e.has_gps() and e.altitude is None]
    if not missing or cfg.relief <= 0:
        return 0
    fetcher = TileFetcher(cfg, [DEM_TEMPLATE], "terrarium")
    filled = 0
    try:
        for entry in missing:
            try:
                metres = elevation_at(fetcher, entry.gps.lat, entry.gps.lon)
            except Exception as exc:  # noqa: BLE001 - offline, bad tile
                log.warning("elevation lookup failed for %s: %s", entry.name, exc)
                break
            if metres is not None:
                entry.altitude = metres
                filled += 1
    finally:
        fetcher.close()
    log.info("filled elevation for %d clip(s) from terrain tiles", filled)
    return filled


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

    def __init__(self, cfg: MapConfig, templates: list[str] | None = None, name: str | None = None):
        self.cfg = cfg
        self.templates = templates or TILE_SOURCES.get(cfg.tiles, TILE_SOURCES["topo"])
        self.name = name or cfg.tiles
        self.cache_dir = Path(cfg.cache_dir or Path(cfg.out_dir).parent / "tile_cache") / self.name
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
                log.warning("%s tile %s/%s/%s unavailable (%s)", self.name, zoom, x, y, exc)
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


# --------------------------------------------------------- 3d terrain

def _decode_terrarium(tile: Image.Image) -> np.ndarray:
    a = np.asarray(tile.convert("RGB"), dtype=np.float32)
    return (a[..., 0] * 256.0 + a[..., 1] + a[..., 2] / 256.0) - 32768.0


def render_dem(fetcher: TileFetcher, center: tuple[float, float], zoom: int,
               size: int) -> np.ndarray | None:
    """Elevation in metres for every pixel of the viewport, or None if the
    elevation tiles are unreachable."""
    zoom = min(zoom, DEM_MAX_ZOOM)
    cx, cy = lon_to_x(center[1], zoom), lat_to_y(center[0], zoom)
    left, top = cx - size / 2, cy - size / 2
    x0, y0 = math.floor(left / TILE_SIZE), math.floor(top / TILE_SIZE)
    x1, y1 = math.floor((left + size) / TILE_SIZE), math.floor((top + size) / TILE_SIZE)

    rows = []
    for ty in range(y0, y1 + 1):
        row = [_decode_terrarium(fetcher.tile(zoom, tx, ty)) for tx in range(x0, x1 + 1)]
        rows.append(np.hstack(row))
    if not rows:
        return None
    grid = np.vstack(rows)
    ox, oy = int(left - x0 * TILE_SIZE), int(top - y0 * TILE_SIZE)
    patch = grid[oy:oy + size, ox:ox + size]
    if patch.shape != (size, size):
        return None
    # a blank (failed) tile decodes to a constant -32768: treat as "no data"
    return patch if patch.max() > -1000 else None


def hillshade(dem: np.ndarray, cell_size_m: float, cfg: MapConfig) -> np.ndarray:
    """Standard cartographic hillshade, 0 (shadow) to 1 (lit)."""
    dy, dx = np.gradient(dem, max(cell_size_m, 1e-3))
    z = max(cfg.relief, 0.0) * 1.6
    slope = np.arctan(z * np.hypot(dx, dy))
    # ESRI aspect convention for a north-up raster. Getting this backwards
    # lights the terrain from the south-east, and every peak reads as a crater.
    aspect = np.arctan2(dy, -dx)
    azimuth = np.radians(360.0 - cfg.sun_azimuth + 90.0)
    altitude = np.radians(cfg.sun_altitude)
    shaded = (np.sin(altitude) * np.cos(slope)
              + np.cos(altitude) * np.sin(slope) * np.cos(azimuth - aspect))
    return np.clip(shaded, 0.0, 1.0)


def apply_relief(view: Image.Image, dem: np.ndarray | None, zoom: int,
                 center: tuple[float, float], cfg: MapConfig) -> Image.Image:
    """Shade the map with the terrain, and push high ground upward.

    Two effects combined: a hillshade multiply, which is what makes ridges and
    couloirs read as shape, and a small vertical displacement so peaks stand
    proud of the valleys. The displacement is drawn with a painter's order -
    rows near the viewer are written last - so a summit correctly occludes the
    ground behind it instead of bleeding through.
    """
    if dem is None or cfg.relief <= 0:
        return view

    rgba = np.asarray(view.convert("RGBA"), dtype=np.float32)
    cell = 156543.03392 * math.cos(math.radians(center[0])) / (2 ** min(zoom, DEM_MAX_ZOOM))
    shade = hillshade(dem, cell, cfg)

    # Multiply with a floor, so shaded slopes stay readable rather than going
    # black. The topo tiles already carry their own hillshade, so a full-range
    # multiply here would double the contrast and crush the labels.
    light = (0.72 + 0.50 * shade)[..., None]
    rgba[..., :3] = np.clip(rgba[..., :3] * light, 0, 255)

    lift = cfg.relief_lift * cfg.relief * view.size[1]
    if lift >= 1.0:
        low, high = float(np.percentile(dem, 2)), float(np.percentile(dem, 98))
        if high - low > 30.0:                      # only where there is real relief
            norm = np.clip((dem - low) / (high - low), 0.0, 1.0)
            shift = (norm * lift).astype(np.int32)
            height, width = shift.shape
            ys, xs = np.indices((height, width))
            target = ys - shift
            keep = target >= 0
            lifted = rgba.copy()                   # holes fall back to the flat map
            lifted[target[keep], xs[keep]] = rgba[ys[keep], xs[keep]]
            rgba = lifted
    return Image.fromarray(rgba.astype(np.uint8), "RGBA")


def elevation_at(fetcher: TileFetcher, lat: float, lon: float, zoom: int = DEM_MAX_ZOOM) -> float | None:
    """Ground elevation in metres at one coordinate, from the elevation tiles."""
    zoom = min(zoom, DEM_MAX_ZOOM)
    x, y = lon_to_x(lon, zoom), lat_to_y(lat, zoom)
    tile = fetcher.tile(zoom, int(x // TILE_SIZE), int(y // TILE_SIZE))
    dem = _decode_terrarium(tile)
    value = float(dem[int(y % TILE_SIZE), int(x % TILE_SIZE)])
    return round(value, 1) if -500.0 < value < 9000.0 else None


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
           cfg: MapConfig, dem_fetcher: TileFetcher | None = None) -> Image.Image:
    """One finished map frame: tiles, terrain relief, route, pin, opacity."""
    flat, origin = render_viewport(fetcher, center, zoom, cfg.size)
    if dem_fetcher is not None and cfg.relief > 0:
        flat = apply_relief(flat, render_dem(dem_fetcher, center, zoom, cfg.size),
                            zoom, center, cfg)
    _draw_route(flat, origin, zoom, coords, upto, cfg)

    tilted, forward = apply_tilt(flat, cfg)
    here = (lon_to_x(coords[upto][1], zoom) - origin[0], lat_to_y(coords[upto][0], zoom) - origin[1])
    _draw_pin(tilted, _project(forward, here), cfg)

    if cfg.tilt <= 0:
        _round_corners(tilted, cfg)
    if cfg.attribution:
        _draw_attribution(tilted, cfg)
    if opacity < 1.0:
        alpha = tilted.getchannel("A").point(lambda v: int(v * opacity))
        tilted.putalpha(alpha)
    return tilted


def _round_corners(view: Image.Image, cfg: MapConfig, radius_fraction: float = 0.035) -> None:
    """Soften the card's corners, in place - an untilted map is a hard
    rectangle sitting on top of the footage."""
    size = view.size[0]
    radius = max(6, int(size * radius_fraction))
    mask = Image.new("L", view.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=255)
    alpha = np.minimum(np.array(view.getchannel("A")), np.array(mask))
    view.putalpha(Image.fromarray(alpha.astype("uint8")))
    ImageDraw.Draw(view).rounded_rectangle([0, 0, size - 1, size - 1], radius=radius,
                                           outline=(255, 255, 255, 150), width=2)


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
    dem_fetcher = TileFetcher(cfg, [DEM_TEMPLATE], "terrarium") if cfg.relief > 0 else None
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
                image = _frame(fetcher, center, cfg.zoom, coords, i, cfg.base_opacity, cfg, dem_fetcher)
                path = out_dir / f"map_{i:04d}.png"
                image.save(path)
                tracks.append(MapTrack([path], cfg.fps, False, [1.0]))
                continue

            clip_dir = out_dir / f"map_{i:04d}"
            clip_dir.mkdir(exist_ok=True)
            frames: list[Path] = []
            scales: list[float] = []
            n_move = max(2, int(cfg.move_seconds * cfg.fps))
            n_settle = max(1, int(cfg.settle_seconds * cfg.fps))

            for step in range(n_move):
                f = _ease(step / (n_move - 1))
                here = (previous[0] + (center[0] - previous[0]) * f,
                        previous[1] + (center[1] - previous[1]) * f)
                # pull back over the middle of the flight, then come back in
                zoom = cfg.zoom - cfg.move_zoom_out * math.sin(math.pi * f)
                # opacity and size swell together as the view sets off
                rise = min(1.0, f * 4)
                opacity = cfg.base_opacity + (cfg.move_opacity - cfg.base_opacity) * rise
                image = _frame(fetcher, here, int(round(zoom)), coords,
                               i if f > 0.5 else max(0, i - 1), opacity, cfg, dem_fetcher)
                path = clip_dir / f"f{step:03d}.png"
                image.save(path)
                frames.append(path)
                scales.append(1.0 + (cfg.move_grow - 1.0) * rise)

            for step in range(n_settle):   # shrink and fade back down on arrival
                f = _ease((step + 1) / n_settle)
                opacity = cfg.move_opacity + (cfg.base_opacity - cfg.move_opacity) * f
                image = _frame(fetcher, center, cfg.zoom, coords, i, opacity, cfg, dem_fetcher)
                path = clip_dir / f"s{step:03d}.png"
                image.save(path)
                frames.append(path)
                scales.append(cfg.move_grow + (1.0 - cfg.move_grow) * f)

            tracks.append(MapTrack(frames, cfg.fps, True, scales))
    finally:
        fetcher.close()
        if dem_fetcher is not None:
            dem_fetcher.close()

    animated = sum(1 for t in tracks if t.animated)
    log.info("rendered %d map views (%d animated moves, %d frames total) at zoom %d from %s tiles",
             len(tracks), animated, sum(len(t.frames) for t in tracks), cfg.zoom, cfg.tiles)
    if fetcher.failures:
        log.warning("%d tile request(s) failed - parts of the map may be blank", fetcher.failures)
    return tracks
