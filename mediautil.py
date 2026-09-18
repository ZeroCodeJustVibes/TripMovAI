"""Small shared helpers for reading pixels out of photos and videos.

Used by Phase 2 (quality metrics, CLIP), Phase 3 (BLIP-2 keyframes) and
Phase 6 (frame sizing). Everything here returns `None` instead of raising so a
single corrupt file can never take the pipeline down.
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

log = logging.getLogger(__name__)

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover - optional
    pass


def load_pil(path: str | Path, max_side: int | None = None) -> Image.Image | None:
    """Open a photo as an upright RGB image."""
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            if max_side and max(img.size) > max_side:
                scale = max_side / max(img.size)
                img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
            return img.copy()
    except Exception as exc:  # noqa: BLE001
        log.warning("cannot open image %s: %s", Path(path).name, exc)
        return None


def video_frame_pil(path: str | Path, at: float | None = None, max_side: int | None = None) -> Image.Image | None:
    """Grab one frame from a video (default: the midpoint) as a PIL image."""
    cap = None
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None
        if at is None:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            at = (count / fps) / 2 if fps > 0 and count > 0 else 0.0
        if at and at > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, at * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:  # seek failed - fall back to the first frame
            cap.set(cv2.CAP_PROP_POS_MSEC, 0)
            ok, frame = cap.read()
        if not ok or frame is None:
            return None
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if max_side and max(img.size) > max_side:
            scale = max_side / max(img.size)
            img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
        return img
    except Exception as exc:  # noqa: BLE001
        log.warning("cannot read frame from %s: %s", Path(path).name, exc)
        return None
    finally:
        if cap is not None:
            cap.release()


def representative_pil(path: str | Path, is_video: bool, max_side: int | None = 720) -> Image.Image | None:
    """One image that stands in for the whole file."""
    return video_frame_pil(path, None, max_side) if is_video else load_pil(path, max_side)


def to_gray(img: Image.Image, max_side: int = 640) -> np.ndarray:
    """Grayscale array at a normalised scale, so thresholds are resolution-independent."""
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.BILINEAR)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)


def laplacian_variance(gray: np.ndarray) -> float:
    """Focus measure - low values mean blurry."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_keyframes(path: str | Path, fps: float = 1.0, max_frames: int = 12,
                      max_side: int = 512) -> list[Image.Image]:
    """Sample frames from a video at `fps` via ffmpeg (Phase 3 captioning)."""
    frames: list[Image.Image] = []
    with tempfile.TemporaryDirectory(prefix="tripvid_kf_") as tmp:
        cmd = [
            "ffmpeg", "-v", "error", "-y", "-i", str(path),
            "-vf", f"fps={fps},scale='min({max_side},iw)':-2",
            "-frames:v", str(max_frames), "-q:v", "3",
            str(Path(tmp) / "kf_%03d.jpg"),
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=180, check=False)
        except (subprocess.SubprocessError, OSError) as exc:
            log.warning("keyframe extraction failed for %s: %s", Path(path).name, exc)
        for f in sorted(Path(tmp).glob("kf_*.jpg")):
            img = load_pil(f)
            if img is not None:
                frames.append(img)
    if not frames:  # ffmpeg produced nothing - at least try a single frame
        single = video_frame_pil(path, None, max_side)
        if single is not None:
            frames = [single]
    return frames


_VOLUME_RE = re.compile(r"(mean|max)_volume:\s*(-?\d+(?:\.\d+)?) dB")


def measure_audio(path: str | Path) -> dict[str, float] | None:
    """Mean (RMS) and peak level of a file's audio, in dBFS.

    Used to even out clips recorded at wildly different levels. ffmpeg's
    `volumedetect` is one quick pass and, unlike EBU R128 integrated loudness,
    stays meaningful on the two-second clips a phone produces.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
             "-map", "0:a:0", "-af", "volumedetect", "-f", "null", "-"],
            capture_output=True, text=True, timeout=300, check=False,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        log.warning("could not measure audio for %s: %s", Path(path).name, exc)
        return None

    found = {key: float(value) for key, value in _VOLUME_RE.findall(result.stderr or "")}
    if "mean" not in found:
        return None
    return {"mean_db": found["mean"], "max_db": found.get("max", 0.0)}


def has_audio_stream(path: str | Path) -> bool:
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
             "stream=index", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30, check=False,
        )
        return bool(res.stdout.strip())
    except (subprocess.SubprocessError, OSError):
        return False
