"""Phase 1 - scan the export folder and build a chronological timeline.

Metadata is read with a cascade of sources, best first:
  1. exiftool (via pyexiftool) - handles every format including HEIC/MOV
  2. an Immich / Takeout style `.json` sidecar next to the file
  3. ffprobe for videos, Pillow EXIF for photos
  4. filesystem mtime (last resort, flagged as `timestamp_source="mtime"`)

A file that fails every probe is still kept - it just sorts by filename.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dateutil import parser as dateparser

from models import MEDIA_EXTS, VIDEO_EXTS, GPS, Timeline, TimelineEntry

log = logging.getLogger(__name__)

try:  # HEIC support for Pillow
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover - optional
    pass


# ---------------------------------------------------------------- helpers

def _parse_timestamp(value: Any) -> str | None:
    """Normalise anything date-ish into an ISO8601 string."""
    if value is None:
        return None
    if isinstance(value, (int, float)):  # epoch seconds (or millis)
        ts = float(value)
        if ts > 1e11:
            ts /= 1000.0
        if ts <= 0:
            return None
        return datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None).isoformat()
    text = str(value).strip()
    if not text or text.startswith(("0000", "1970:01:01 00:00:00")):
        return None
    if text.isdigit() and len(text) >= 10:  # epoch as string
        return _parse_timestamp(int(text))
    # exiftool style "2024:06:12 14:30:05" -> "2024-06-12 14:30:05"
    m = re.match(r"^(\d{4}):(\d{2}):(\d{2})[ T](.*)$", text)
    if m:
        text = f"{m.group(1)}-{m.group(2)}-{m.group(3)} {m.group(4)}"
    try:
        dt = dateparser.parse(text)
    except (ValueError, OverflowError, TypeError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is not None:  # store naive local-ish time for stable sorting
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    if dt.year < 1990 or dt.year > 2100:
        return None
    return dt.isoformat()


# Camera and messenger naming conventions that carry the capture time.
_NAME_PATTERNS = (
    # 20250910_144855 / IMG_20250910_144855 / VID_... and PXL_20250910_144855123,
    # where Pixel appends milliseconds
    (re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})[_\-T](\d{2})(\d{2})(\d{2})(?:\d{1,3})?(?!\d)"), "second"),
    # signal-2025-09-10-144855
    (re.compile(r"(?<!\d)(20\d{2})[-.](\d{2})[-.](\d{2})[-_](\d{2})(\d{2})(\d{2})(?!\d)"), "second"),
    # 2025-09-10 14.48.55 / 2025-09-10_14-48-55
    (re.compile(r"(?<!\d)(20\d{2})[-.](\d{2})[-.](\d{2})[ _T](\d{2})[-.:](\d{2})[-.:](\d{2})(?!\d)"), "second"),
    # IMG-20260228-WA0003 (WhatsApp) - the day only
    (re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)"), "day"),
    (re.compile(r"(?<!\d)(20\d{2})-(\d{2})-(\d{2})(?!\d)"), "day"),
)


def timestamp_from_name(name: str) -> tuple[str | None, str | None]:
    """A capture time read out of the filename, and how precise it is.

    Phones name files after the moment they were taken - `20250910_144855.jpg`
    on Samsung, `PXL_20250910_144855123` on Pixel, `IMG-20250910-WA0007` from
    WhatsApp (day only). That is the only timestamp a WhatsApp forward or a
    stripped export still carries.
    """
    stem = Path(name).stem
    for pattern, precision in _NAME_PATTERNS:
        match = pattern.search(stem)
        if not match:
            continue
        parts = [int(g) for g in match.groups()]
        year, month, day = parts[0], parts[1], parts[2]
        hour, minute, second = (parts[3:6] + [0, 0, 0])[:3] if len(parts) > 3 else (0, 0, 0)
        try:
            when = datetime(year, month, day, hour, minute, second)
        except ValueError:
            continue
        if 1990 <= when.year <= 2100:
            return when.isoformat(), precision
    return None, None


def _valid_gps(lat: Any, lon: Any) -> GPS | None:
    try:
        lat_f, lon_f = float(lat), float(lon)
    except (TypeError, ValueError):
        return None
    if not (-90 <= lat_f <= 90) or not (-180 <= lon_f <= 180):
        return None
    if abs(lat_f) < 1e-7 and abs(lon_f) < 1e-7:  # null island == no fix
        return None
    return GPS(round(lat_f, 7), round(lon_f, 7))


_ISO6709 = re.compile(r"^([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)?")


def _parse_iso6709(value: str) -> GPS | None:
    """`+37.7749-122.4194+010.000/` - the QuickTime location string."""
    m = _ISO6709.match(str(value).strip())
    return _valid_gps(m.group(1), m.group(2)) if m else None


def _iso6709_altitude(value: str) -> float | None:
    """The third component of a QuickTime location string, in metres."""
    m = _ISO6709.match(str(value).strip())
    if not m or not m.group(3):
        return None
    return _valid_altitude(m.group(3))


def _valid_altitude(value: Any, below_sea: Any = None) -> float | None:
    """Metres above sea level, rejecting the nonsense some cameras write.

    An exact 0 is treated as "not recorded", not as sea level: ffmpeg writes
    `+0.000` into a QuickTime location string whenever it has no altitude, and
    phones do the same. Phase 5 then fills the real ground height from the
    terrain tiles, which is right for a genuine beach photo too.
    """
    try:
        metres = float(value)
    except (TypeError, ValueError):
        return None
    if abs(metres) < 0.5:
        return None
    if str(below_sea).strip() in ("1", "True") and metres > 0:
        metres = -metres          # GPSAltitudeRef 1 means below sea level
    if not (-500.0 <= metres <= 9000.0):
        return None
    return round(metres, 1)


def _signed_gps(value: Any, ref: Any) -> Any:
    """Apply an N/S/E/W reference to a (possibly unsigned) coordinate."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if ref and str(ref).strip().upper()[:1] in ("S", "W") and v > 0:
        v = -v
    return v


# ------------------------------------------------------------- discovery

def scan_files(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"input folder not found: {root}")
    files = [
        p
        for p in sorted(root.rglob("*"))
        if p.is_file()
        and p.suffix.lower() in MEDIA_EXTS
        and not p.name.startswith("._")  # macOS resource forks
    ]
    return files


# -------------------------------------------------------------- exiftool

def exiftool_available() -> bool:
    return shutil.which("exiftool") is not None


def read_exiftool(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """Batch-read metadata for every file. Returns {path: tags}."""
    if not paths or not exiftool_available():
        return {}
    try:
        import exiftool  # pyexiftool
    except ImportError:
        return {}

    out: dict[str, dict[str, Any]] = {}
    try:
        with exiftool.ExifToolHelper(common_args=["-G", "-n", "-a", "-u"]) as et:
            # chunked so one unreadable file cannot poison the whole batch
            for i in range(0, len(paths), 100):
                chunk = [str(p) for p in paths[i : i + 100]]
                try:
                    for tags in et.get_metadata(chunk):
                        src = tags.get("SourceFile")
                        if src:
                            out[os.path.abspath(src)] = tags
                except Exception as exc:  # noqa: BLE001 - keep going
                    log.warning("exiftool failed on %d files: %s", len(chunk), exc)
    except Exception as exc:  # noqa: BLE001 - exiftool missing/broken
        log.warning("exiftool unavailable (%s); falling back to ffprobe/Pillow", exc)
    return out


_TIME_TAGS = (
    "EXIF:DateTimeOriginal",
    "EXIF:CreateDate",
    "XMP:DateTimeOriginal",
    "QuickTime:CreationDate",
    "QuickTime:CreateDate",
    "QuickTime:MediaCreateDate",
    "Composite:SubSecDateTimeOriginal",
    "File:FileModifyDate",
)


def _from_exiftool(tags: dict[str, Any]) -> dict[str, Any]:
    info: dict[str, Any] = {}
    for tag in _TIME_TAGS:
        ts = _parse_timestamp(tags.get(tag))
        if ts:
            info["timestamp"] = ts
            info["timestamp_source"] = "mtime" if tag.startswith("File:") else "exif"
            break

    gps = _valid_gps(
        _signed_gps(tags.get("Composite:GPSLatitude", tags.get("EXIF:GPSLatitude")), tags.get("EXIF:GPSLatitudeRef")),
        _signed_gps(tags.get("Composite:GPSLongitude", tags.get("EXIF:GPSLongitude")), tags.get("EXIF:GPSLongitudeRef")),
    )
    if gps is None:
        for tag in ("QuickTime:GPSCoordinates", "Composite:GPSPosition", "XMP:GPSCoordinates"):
            if tags.get(tag):
                gps = _parse_iso6709(tags[tag]) or _valid_gps(*str(tags[tag]).split()[:2])
                if gps:
                    break
    if gps:
        info["gps"] = gps

    for w_tag, h_tag in (("File:ImageWidth", "File:ImageHeight"),
                         ("EXIF:ExifImageWidth", "EXIF:ExifImageHeight"),
                         ("QuickTime:ImageWidth", "QuickTime:ImageHeight")):
        w, h = tags.get(w_tag), tags.get(h_tag)
        if w and h:
            info["width"], info["height"] = int(w), int(h)
            break

    altitude = _valid_altitude(tags.get("Composite:GPSAltitude"))
    if altitude is None:
        altitude = _valid_altitude(tags.get("EXIF:GPSAltitude"), tags.get("EXIF:GPSAltitudeRef"))
    if altitude is None:
        for tag in ("QuickTime:GPSCoordinates", "Composite:GPSPosition"):
            if tags.get(tag):
                altitude = _iso6709_altitude(tags[tag])
                if altitude is not None:
                    break
    if altitude is not None:
        info["altitude"] = altitude

    model = tags.get("EXIF:Model") or tags.get("QuickTime:Model") or tags.get("XMP:Model")
    if model:
        make = tags.get("EXIF:Make") or ""
        info["camera"] = f"{make} {model}".strip() if str(make).lower() not in str(model).lower() else str(model)

    dur = tags.get("QuickTime:Duration") or tags.get("Composite:Duration")
    if dur:
        try:
            info["duration"] = float(dur)
        except (TypeError, ValueError):
            pass
    rot = tags.get("Composite:Rotation") or tags.get("QuickTime:Rotation")
    if rot is not None:
        try:
            info["rotation"] = int(float(rot))
        except (TypeError, ValueError):
            pass
    return info


# --------------------------------------------------------------- sidecar

def find_sidecar(path: Path) -> Path | None:
    """Immich/Takeout sidecars sit next to the file as `<name>.json`."""
    candidates = [
        path.with_suffix(path.suffix + ".json"),  # IMG_1.jpg.json
        path.with_suffix(".json"),                # IMG_1.json
        path.parent / f"{path.name}.supplemental-metadata.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _dig(data: dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def read_sidecar(path: Path) -> dict[str, Any]:
    """Tolerant reader for Immich asset JSON and Google Takeout JSON."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("unreadable sidecar %s: %s", path.name, exc)
        return {}
    if not isinstance(data, dict):
        return {}

    info: dict[str, Any] = {}
    for candidate in (
        _dig(data, "photoTakenTime", "timestamp"),
        _dig(data, "creationTime", "timestamp"),
        data.get("dateTimeOriginal"),
        _dig(data, "exifInfo", "dateTimeOriginal"),
        data.get("localDateTime"),
        data.get("fileCreatedAt"),
        data.get("DateTimeOriginal"),
        data.get("CreateDate"),
    ):
        ts = _parse_timestamp(candidate)
        if ts:
            info["timestamp"] = ts
            info["timestamp_source"] = "sidecar"
            break

    for lat, lon in (
        (_dig(data, "geoData", "latitude"), _dig(data, "geoData", "longitude")),
        (_dig(data, "geoDataExif", "latitude"), _dig(data, "geoDataExif", "longitude")),
        (_dig(data, "exifInfo", "latitude"), _dig(data, "exifInfo", "longitude")),
        (data.get("latitude"), data.get("longitude")),
        (data.get("GPSLatitude"), data.get("GPSLongitude")),
    ):
        gps = _valid_gps(lat, lon)
        if gps:
            info["gps"] = gps
            break

    for candidate in (_dig(data, "exifInfo", "altitude"), _dig(data, "geoData", "altitude"),
                      data.get("altitude"), data.get("GPSAltitude")):
        altitude = _valid_altitude(candidate)
        if altitude is not None:
            info["altitude"] = altitude
            break

    model = _dig(data, "exifInfo", "model") or data.get("model")
    if model:
        make = _dig(data, "exifInfo", "make") or ""
        info["camera"] = f"{make} {model}".strip()
    dur = _dig(data, "exifInfo", "duration") or data.get("duration")
    if isinstance(dur, str) and ":" in dur:  # "0:00:07.50"
        try:
            h, m, s = dur.split(":")
            dur = int(h) * 3600 + int(m) * 60 + float(s)
        except ValueError:
            dur = None
    if dur:
        try:
            info["duration"] = float(dur)
        except (TypeError, ValueError):
            pass
    return info


# --------------------------------------------------------------- ffprobe

def ffprobe(path: Path) -> dict[str, Any]:
    """Duration, dimensions, rotation and embedded GPS for a video file."""
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_entries", "format=duration:format_tags:stream=width,height,codec_type,nb_frames:stream_tags",
        "-show_streams", str(path),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
        data = json.loads(res.stdout or "{}")
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as exc:
        log.warning("ffprobe failed on %s: %s", path.name, exc)
        return {}

    info: dict[str, Any] = {}
    fmt = data.get("format", {}) or {}
    try:
        if fmt.get("duration"):
            info["duration"] = float(fmt["duration"])
    except (TypeError, ValueError):
        pass

    tags = {str(k).lower(): v for k, v in (fmt.get("tags") or {}).items()}
    for key in ("com.apple.quicktime.location.iso6709", "location", "location-eng"):
        if key in tags:
            gps = _parse_iso6709(tags[key])
            if gps:
                info["gps"] = gps
                altitude = _iso6709_altitude(tags[key])
                if altitude is not None:
                    info["altitude"] = altitude
                break
    for key in ("creation_time", "com.apple.quicktime.creationdate", "date"):
        ts = _parse_timestamp(tags.get(key))
        if ts:
            info["timestamp"] = ts
            info["timestamp_source"] = "ffprobe"
            break
    model = tags.get("com.apple.quicktime.model") or tags.get("model")
    if model:
        info["camera"] = str(model)

    for stream in data.get("streams", []) or []:
        if stream.get("codec_type") != "video":
            continue
        if stream.get("width") and stream.get("height"):
            info["width"], info["height"] = int(stream["width"]), int(stream["height"])
        rot = (stream.get("tags") or {}).get("rotate")
        side = stream.get("side_data_list") or []
        if rot is None and side:
            rot = side[0].get("rotation")
        if rot is not None:
            try:
                info["rotation"] = int(float(rot))
            except (TypeError, ValueError):
                pass
        if not info.get("duration") and stream.get("duration"):
            try:
                info["duration"] = float(stream["duration"])
            except (TypeError, ValueError):
                pass
        break
    return info


def _pillow_info(path: Path) -> dict[str, Any]:
    try:
        from PIL import Image, ExifTags

        with Image.open(path) as img:
            info: dict[str, Any] = {"width": img.width, "height": img.height}
            exif = img.getexif() or {}
            merged = dict(exif)
            try:  # DateTimeOriginal lives in the Exif sub-IFD, not the root
                merged.update(exif.get_ifd(ExifTags.IFD.Exif) or {})
            except Exception:  # noqa: BLE001
                pass
            tag_by_name = {ExifTags.TAGS.get(k, k): v for k, v in merged.items()}
            ts = _parse_timestamp(tag_by_name.get("DateTimeOriginal") or tag_by_name.get("DateTime"))
            if ts:
                info["timestamp"] = ts
                info["timestamp_source"] = "exif"
            if tag_by_name.get("Model"):
                info["camera"] = str(tag_by_name["Model"]).strip()
            try:
                gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo) or {}
            except Exception:  # noqa: BLE001
                gps_ifd = {}
            if gps_ifd:
                named = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}

                def _dms(v: Any) -> float | None:
                    try:
                        d, m, s = (float(x) for x in v)
                        return d + m / 60 + s / 3600
                    except (TypeError, ValueError):
                        return None

                lat, lon = _dms(named.get("GPSLatitude")), _dms(named.get("GPSLongitude"))
                gps = _valid_gps(
                    _signed_gps(lat, named.get("GPSLatitudeRef")),
                    _signed_gps(lon, named.get("GPSLongitudeRef")),
                )
                if gps:
                    info["gps"] = gps
                altitude = _valid_altitude(named.get("GPSAltitude"), named.get("GPSAltitudeRef"))
                if altitude is not None:
                    info["altitude"] = altitude
            return info
    except Exception as exc:  # noqa: BLE001 - unreadable image
        log.warning("Pillow could not read %s: %s", path.name, exc)
        return {}


# ------------------------------------------------------------------ main

def build_entry(path: Path, exif_tags: dict[str, Any] | None = None) -> TimelineEntry:
    kind = "video" if path.suffix.lower() in VIDEO_EXTS else "photo"
    info: dict[str, Any] = {}

    if exif_tags:
        info.update(_from_exiftool(exif_tags))

    sidecar = find_sidecar(path)
    if sidecar:
        for key, value in read_sidecar(sidecar).items():
            info.setdefault(key, value)  # exiftool wins, sidecar fills gaps

    if kind == "video":
        for key, value in ffprobe(path).items():
            info.setdefault(key, value)  # exiftool and the sidecar win where they had data
    elif not info.get("width") or not info.get("timestamp"):
        for key, value in _pillow_info(path).items():
            info.setdefault(key, value)

    # The filename often carries the capture time, and it survives edits and
    # exports that strip EXIF. Metadata still wins when it has a real date -
    # it is what the camera recorded - but a filename beats the file's mtime,
    # which is just when it was last copied.
    named, precision = timestamp_from_name(path.name)
    if named:
        info["filename_time"] = named
        current, source = info.get("timestamp"), info.get("timestamp_source")
        if not current or source in (None, "mtime", "ffprobe"):
            if not current or source != "ffprobe" or precision == "second":
                info["timestamp"] = named
                info["timestamp_source"] = "filename"
        elif precision == "second":
            try:
                drift = abs((dateparser.parse(current) - dateparser.parse(named)).total_seconds())
                if drift > 86400:
                    log.info("%s: metadata says %s but the name says %s - keeping the metadata",
                             path.name, current[:16], named[:16])
            except (ValueError, TypeError):
                pass

    if not info.get("timestamp"):
        try:
            info["timestamp"] = _parse_timestamp(path.stat().st_mtime)
            info["timestamp_source"] = "mtime"
        except OSError:
            pass

    rotation = info.pop("rotation", None)
    width, height = info.get("width"), info.get("height")
    if rotation and width and height and abs(int(rotation)) % 180 == 90:
        info["width"], info["height"] = height, width  # store display orientation

    return TimelineEntry(
        filepath=str(path.resolve()),
        type=kind,
        timestamp=info.get("timestamp"),
        gps=info.get("gps"),
        duration=float(info.get("duration") or 0.0),
        width=info.get("width"),
        height=info.get("height"),
        camera=info.get("camera"),
        altitude=info.get("altitude"),
        timestamp_source=info.get("timestamp_source"),
        extra={"filename_time": info["filename_time"]} if info.get("filename_time") else {},
    )


def _sort_key(entry: TimelineEntry) -> tuple[int, str, str]:
    """Chronological; undated files fall to the end, ordered by filename."""
    if entry.timestamp:
        return (0, entry.timestamp, entry.name.lower())
    return (1, "", entry.name.lower())


def ingest(input_dir: str | Path, previous: Timeline | None = None, progress_cb=None) -> Timeline:
    """Scan `input_dir` and return a chronologically sorted Timeline.

    If `previous` is given, per-file annotations already computed by later
    phases (captions, scores, selection) are carried over.
    """
    paths = scan_files(input_dir)
    log.info("found %d media files under %s", len(paths), input_dir)
    if not paths:
        return Timeline()

    exif_by_path = read_exiftool(paths)
    if not exif_by_path:
        log.warning("exiftool produced no metadata - using ffprobe/Pillow fallbacks")

    entries: list[TimelineEntry] = []
    for i, path in enumerate(paths):
        try:
            entry = build_entry(path, exif_by_path.get(os.path.abspath(str(path))))
        except Exception as exc:  # noqa: BLE001 - never lose the run over one file
            log.error("failed to ingest %s: %s", path.name, exc)
            continue
        if previous is not None:
            old = previous.by_path(entry.filepath)
            if old is not None:
                entry.caption = old.caption
                entry.caption_confidence = old.caption_confidence
                entry.transcript = old.transcript
                entry.quality_score = old.quality_score
                entry.blur_score = old.blur_score
                entry.brightness = old.brightness
                entry.selected = old.selected
                entry.place = old.place
                entry.drop_reason = old.drop_reason
                entry.group = old.group
                entry.extra = old.extra
        entries.append(entry)
        if progress_cb:
            progress_cb(i + 1, len(paths))

    entries.sort(key=_sort_key)
    return Timeline(entries)


def summarise(timeline: Timeline) -> dict[str, Any]:
    photos = sum(1 for e in timeline if e.type == "photo")
    videos = sum(1 for e in timeline if e.type == "video")
    dated = sum(1 for e in timeline if e.timestamp and e.timestamp_source != "mtime")
    located = sum(1 for e in timeline if e.has_gps())
    span = [e.timestamp for e in timeline if e.timestamp]
    return {
        "files": len(timeline),
        "photos": photos,
        "videos": videos,
        "with_exif_time": dated,
        "with_gps": located,
        "video_seconds": round(sum(e.duration for e in timeline if e.is_video), 1),
        "first": min(span) if span else None,
        "last": max(span) if span else None,
    }
