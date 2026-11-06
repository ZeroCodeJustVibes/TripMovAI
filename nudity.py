"""Find and blur explicit regions, so a stray frame does not end up on screen.

Detection runs once per clip in Phase 3 and is cached on the timeline as
*normalised* boxes, so the same result applies at any render size. Blurring
happens at render time, which keeps the originals untouched - nothing here
writes to the source files.

Only the explicitly exposed classes are acted on. A shirtless swimmer or a
sunbathing back is not what this is for, and blurring half a beach holiday
would be worse than the problem.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageFilter

log = logging.getLogger(__name__)

# NudeNet's labels for genuinely exposed anatomy. Everything else it reports -
# faces, feet, bellies, armpits, anything "covered" - is left alone.
EXPLICIT = frozenset({
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
})

DETECT_SIZE = 640          # what the detector sees; boxes come back normalised
_DETECTOR = None


@dataclass
class NudityConfig:
    threshold: float = 0.40     # detector confidence to act on
    grow: float = 0.12          # enlarge each box by this fraction
    strength: float = 0.11      # blur radius as a fraction of the box's size
    video_frames: int = 6       # keyframes sampled per video
    classes: frozenset = field(default_factory=lambda: EXPLICIT)


def _detector():
    """The NudeNet detector, loaded once. None if it is not installed."""
    global _DETECTOR
    if _DETECTOR is None:
        try:
            from nudenet import NudeDetector

            _DETECTOR = NudeDetector()
        except Exception as exc:  # noqa: BLE001 - optional dependency
            log.warning("nudenet unavailable (%s) - nothing will be blurred", exc)
            _DETECTOR = False
    return _DETECTOR or None


def detect_image(image: Image.Image, cfg: NudityConfig) -> list[list[float]]:
    """Normalised [x, y, w, h] boxes of explicit regions in one picture.

    The detector wants a file path and cannot read HEIC, so the frame is
    written out as a small JPEG first - which also bounds the work per image.
    """
    detector = _detector()
    if detector is None or image is None:
        return []

    scaled = image.copy()
    scaled.thumbnail((DETECT_SIZE, DETECT_SIZE), Image.LANCZOS)
    width, height = scaled.size
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as handle:
        try:
            scaled.convert("RGB").save(handle.name, quality=88)
            found = detector.detect(handle.name) or []
        except Exception as exc:  # noqa: BLE001 - a bad frame is not fatal
            log.warning("nudity detection failed: %s", exc)
            return []

    boxes = []
    for item in found:
        try:
            label = str(item.get("class", ""))
            score = float(item.get("score", 0.0))
            x, y, w, h = (float(v) for v in item["box"])
        except (KeyError, TypeError, ValueError):
            continue
        if label not in cfg.classes or score < cfg.threshold:
            continue
        boxes.append([max(0.0, x / width), max(0.0, y / height),
                      min(1.0, w / width), min(1.0, h / height)])
    return boxes


def scan(path: str | Path, is_video: bool, cfg: NudityConfig) -> dict:
    """Detection result for one clip, ready to cache on the timeline.

    A still keeps its boxes. A video only records *whether* anything was found:
    the subject moves, so a box from one keyframe would not cover the next -
    the whole picture is blurred for those.
    """
    from mediautil import extract_keyframes, load_pil

    if not is_video:
        boxes = detect_image(load_pil(path, max_side=DETECT_SIZE * 2), cfg)
        return {"boxes": boxes, "whole": False, "found": bool(boxes)}

    frames = extract_keyframes(path, 1.0, cfg.video_frames, DETECT_SIZE)
    for frame in frames:
        if detect_image(frame, cfg):
            return {"boxes": [], "whole": True, "found": True}
    return {"boxes": [], "whole": False, "found": False}


def blur_regions(image: Image.Image, boxes: list[list[float]], cfg: NudityConfig) -> Image.Image:
    """Blur the given normalised regions of a picture, in a copy."""
    if not boxes:
        return image
    out = image.copy()
    width, height = out.size
    for box in boxes:
        try:
            x, y, w, h = (float(v) for v in box)
        except (TypeError, ValueError):
            continue
        grow_x, grow_y = w * cfg.grow * width, h * cfg.grow * height
        left = max(0, int(x * width - grow_x))
        top = max(0, int(y * height - grow_y))
        right = min(width, int((x + w) * width + grow_x))
        bottom = min(height, int((y + h) * height + grow_y))
        if right - left < 4 or bottom - top < 4:
            continue
        patch = out.crop((left, top, right, bottom))
        radius = max(6.0, cfg.strength * max(patch.size))
        out.paste(patch.filter(ImageFilter.GaussianBlur(radius)), (left, top))
    return out


def blur_everything(image: Image.Image, cfg: NudityConfig) -> Image.Image:
    """Blur a whole picture - used for videos, where a box would not hold."""
    radius = max(8.0, cfg.strength * max(image.size) * 1.4)
    return image.filter(ImageFilter.GaussianBlur(radius))


def blur_array(frame, cfg: NudityConfig):
    """The same heavy blur for a video frame, cheaply.

    Down-sampling and scaling back up costs a fraction of a Gaussian at this
    size and is indistinguishable once it is this strong - which matters when
    it runs on every frame of a clip.
    """
    import numpy as np

    height, width = frame.shape[:2]
    factor = max(8, int(min(width, height) * cfg.strength * 1.8))
    small_w, small_h = max(1, width // factor), max(1, height // factor)
    small = Image.fromarray(frame).resize((small_w, small_h), Image.BILINEAR)
    return np.asarray(small.resize((width, height), Image.BILINEAR), dtype=frame.dtype)
