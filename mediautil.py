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


def release_memory() -> None:
    """Hand freed memory back to the operating system.

    Python releases the objects, but glibc keeps the arenas: after Phase 3
    loads a 15 GB checkpoint and a 6 GB Whisper model, that memory stays in the
    process's RSS for the rest of the run even though nothing references it.
    `malloc_trim` returns it, which is why an interrupted run that skips those
    phases looks so much leaner.
    """
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - torch may not be installed
        pass
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001 - not glibc, nothing to do
        pass


def extracted_audio(path: str | Path, cache_dir: str | Path | None = None) -> str | None:
    """The clip's soundtrack as a small WAV, cached.

    Reading audio through a `VideoFileClip` costs a second persistent ffmpeg
    reader per clip - with 25 videos in a cut that was 50 reader processes and
    tens of gigabytes of buffers. A WAV is cheap to open and cheap to seek.
    """
    source = Path(path)
    target_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "tripvid_audio"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{source.stem}.wav"
    if target.exists() and target.stat().st_size > 1024:
        return str(target)
    if not has_audio_stream(path):
        return None
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(path), "-map", "0:a:0",
           "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(target)]
    try:
        subprocess.run(cmd, capture_output=True, timeout=900, check=False)
    except (subprocess.SubprocessError, OSError) as exc:
        log.warning("could not extract audio from %s: %s", source.name, exc)
        return None
    return str(target) if target.exists() and target.stat().st_size > 1024 else None


def nvenc_available() -> bool:
    """Whether this ffmpeg can encode H.264 on the GPU."""
    global _NVENC
    if _NVENC is None:
        try:
            result = subprocess.run(["ffmpeg", "-v", "error", "-encoders"],
                                    capture_output=True, text=True, timeout=30, check=False)
            _NVENC = "h264_nvenc" in (result.stdout or "")
        except (subprocess.SubprocessError, OSError):
            _NVENC = False
    return bool(_NVENC)


_NVENC: bool | None = None
_PLAYABLE: dict[str, str | None] = {}


def _moviepy_opens(path: str | Path) -> bool:
    try:
        from moviepy import VideoFileClip

        clip = VideoFileClip(str(path), audio=False)
        clip.close()
        return True
    except Exception:  # noqa: BLE001 - that is exactly what we are testing for
        return False


def playable_video(path: str | Path, cache_dir: str | Path | None = None) -> str | None:
    """A path MoviePy can actually open, remuxing the container if it cannot.

    iPhone MOVs carry timecode and metadata streams alongside the picture, and
    often a second spatial-audio track. MoviePy's ffmpeg info parser trips over
    those streams and refuses the file - on a real library that was 14 of 73
    videos, silently dropped from the film. ffmpeg itself reads them fine, so
    the fix is to copy the picture and the first audio track into a clean
    container: no re-encoding, about a second per clip, cached on disk.

    Returns None only if even that fails, in which case the caller skips the
    clip as it would any unreadable file.
    """
    key = str(Path(path).resolve())
    if key in _PLAYABLE:
        return _PLAYABLE[key]

    if _moviepy_opens(path):
        _PLAYABLE[key] = str(path)
        return str(path)

    target_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "tripvid_remux"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{Path(path).stem}.mp4"

    if not target.exists():
        log.info("remuxing %s - its container upsets MoviePy", Path(path).name)
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(path),
               "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy",
               "-movflags", "+faststart", str(target)]
        try:
            subprocess.run(cmd, capture_output=True, timeout=900, check=False)
        except (subprocess.SubprocessError, OSError) as exc:
            log.error("could not remux %s: %s", Path(path).name, exc)

    if target.exists() and _moviepy_opens(target):
        _PLAYABLE[key] = str(target)
        return str(target)

    log.error("%s cannot be read for assembly even after remuxing", Path(path).name)
    _PLAYABLE[key] = None
    return None


def audio_duration(path: str | Path) -> float | None:
    """Length of an audio (or video) file in seconds, without decoding it."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=60, check=False)
        return float(result.stdout.strip())
    except (subprocess.SubprocessError, OSError, ValueError):
        log.warning("could not read the length of %s", Path(path).name)
        return None


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
