"""Pick the moments worth keeping from a long video.

Trimming to the first N seconds keeps whatever the camera happened to see when
recording started - usually a pocket, a lift queue or somebody saying "is it
on?". This scores a clip over its whole length and takes the best stretches
instead, up to a budget.

What counts as interesting: movement (something is happening), sharpness (the
camera settled), sensible exposure, loudness, and above all speech - if Whisper
heard someone talk, that is almost always the bit worth keeping.

Results are cached on the timeline, so the scoring happens once per video.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_FPS = 2.0          # frames per second scored
SAMPLE_WIDTH = 320        # they only need to be big enough to measure
MAX_SAMPLES = 900         # ~7.5 minutes of video at 2 fps


@dataclass
class HighlightConfig:
    budget: float = 15.0          # seconds to keep from any one video
    min_segment: float = 3.0      # shorter than this reads as a glitch
    max_segments: int = 3         # how many separate moments at most
    gap: float = 2.0              # keep picks this far apart
    whole_if_within: float = 1.05  # use it all only if barely over budget


def _scan_frames(path: str | Path, fps: float = SAMPLE_FPS) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Walk the sampled frames once, keeping only the measurements.

    The frames themselves are never held: at 2 fps a ten-minute clip is 1200
    frames, and stacking them costs hundreds of megabytes for numbers that fit
    in a few kilobytes.
    """
    from PIL import Image

    times, sharp, motion, brightness = [], [], [], []
    previous = None
    with tempfile.TemporaryDirectory(prefix="tripvid_scan_") as tmp:
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(path),
               "-vf", f"fps={fps},scale={SAMPLE_WIDTH}:-2,format=gray",
               "-frames:v", str(MAX_SAMPLES), str(Path(tmp) / "f_%05d.png")]
        try:
            subprocess.run(cmd, capture_output=True, timeout=600, check=False)
        except (subprocess.SubprocessError, OSError) as exc:
            log.warning("could not scan %s: %s", Path(path).name, exc)
            return np.empty(0), {}

        for index, f in enumerate(sorted(Path(tmp).glob("f_*.png"))):
            try:
                with Image.open(f) as img:
                    frame = np.asarray(img.convert("L"), dtype=np.float32)
            except Exception:  # noqa: BLE001 - skip an unreadable sample
                continue
            times.append(index / fps)
            sharp.append(float(np.var(np.diff(frame, axis=0))) + float(np.var(np.diff(frame, axis=1))))
            brightness.append(float(frame.mean()))
            motion.append(0.0 if previous is None else float(np.abs(frame - previous).mean()))
            previous = frame

    if not times:
        return np.empty(0), {}
    return np.asarray(times), {
        "sharp": np.asarray(sharp),
        "motion": np.asarray(motion),
        "brightness": np.asarray(brightness),
    }


def _audio_energy(path: str | Path, times: np.ndarray, rate: int = 8000) -> np.ndarray:
    """Loudness at each sample time, 0..1. Silent or unreadable audio -> zeros."""
    if not len(times):
        return np.empty(0)
    try:
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-ac", "1",
             "-ar", str(rate), "-f", "f32le", "-"],
            capture_output=True, timeout=600, check=False)
        audio = np.frombuffer(result.stdout, dtype=np.float32)
    except (subprocess.SubprocessError, OSError, ValueError):
        return np.zeros(len(times))
    if audio.size < rate // 4:
        return np.zeros(len(times))

    window = max(1, int(rate / SAMPLE_FPS))
    usable = (audio.size // window) * window
    rms = np.sqrt((audio[:usable].reshape(-1, window) ** 2).mean(axis=1) + 1e-12)
    loud = np.clip((20 * np.log10(rms) + 50.0) / 50.0, 0.0, 1.0)
    index = np.clip((times * SAMPLE_FPS).astype(int), 0, len(loud) - 1)
    return loud[index]


def _normalise(values: np.ndarray) -> np.ndarray:
    if not len(values):
        return values
    low, high = float(np.percentile(values, 5)), float(np.percentile(values, 95))
    if high - low < 1e-6:
        return np.full_like(values, 0.5)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def score_video(path: str | Path, speech: list[dict] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(times, interest score 0..1) across a clip."""
    times, measured = _scan_frames(path)
    if not len(times):
        return np.empty(0), np.empty(0)

    sharp, motion = measured["sharp"], measured["motion"]
    brightness = measured["brightness"]
    exposure = 1.0 - np.clip(np.abs(brightness - 128.0) / 128.0, 0.0, 1.0)
    loudness = _audio_energy(path, times)

    score = (0.34 * _normalise(motion) + 0.28 * _normalise(sharp)
             + 0.16 * exposure + 0.22 * loudness)

    # speech is the strongest signal there is: keep the bit where people talk
    for segment in speech or []:
        try:
            start, end = float(segment["start"]), float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        score[(times >= start - 0.5) & (times <= end + 0.5)] += 0.45

    # very dark or blown samples are not worth keeping whatever else they score
    score[(brightness < 22) | (brightness > 240)] *= 0.25

    smooth = max(1, int(SAMPLE_FPS * 1.5))
    score = np.convolve(score, np.ones(smooth) / smooth, mode="same")
    return times, score


def pick_segments(times: np.ndarray, score: np.ndarray, duration: float,
                  cfg: HighlightConfig) -> list[tuple[float, float]]:
    """The best few stretches, in the order they happen in the clip."""
    if not len(times) or duration <= 0:
        return []
    budget = min(cfg.budget, duration)
    wanted = max(1, min(cfg.max_segments, int(round(duration / 30.0)) or 1))
    length = max(cfg.min_segment, budget / wanted)
    if length > duration:
        return [(0.0, duration)]

    span = max(1, int(length * SAMPLE_FPS))
    if span >= len(score):
        return [(0.0, min(duration, budget))]
    kernel = np.ones(span) / span
    window_mean = np.convolve(score, kernel, mode="valid")      # score of each start point
    starts = times[: len(window_mean)]

    picked: list[tuple[float, float]] = []
    remaining = budget
    available = np.ones(len(window_mean), dtype=bool)
    while remaining >= cfg.min_segment - 1e-6 and available.any():
        take = min(length, remaining)
        index = int(np.argmax(np.where(available, window_mean, -np.inf)))
        start = float(min(starts[index], max(0.0, duration - take)))
        picked.append((start, min(duration, start + take)))
        remaining -= take
        blocked = (starts >= start - take - cfg.gap) & (starts <= start + take + cfg.gap)
        available &= ~blocked
        if len(picked) >= cfg.max_segments:
            break

    picked.sort()
    merged: list[tuple[float, float]] = []
    for start, end in picked:                  # join anything that ended up adjacent
        if merged and start - merged[-1][1] < 0.4:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def highlights_for(path: str | Path, duration: float, cfg: HighlightConfig,
                   speech: list[dict] | None = None) -> list[tuple[float, float]]:
    """Which stretches of this video to use. `[(0, duration)]` means all of it."""
    if duration <= 0:
        return []
    if duration <= cfg.budget * cfg.whole_if_within:
        return [(0.0, min(duration, cfg.budget))]

    times, score = score_video(path, speech)
    if not len(times):
        log.warning("could not score %s - using the opening %.0fs", Path(path).name, cfg.budget)
        return [(0.0, cfg.budget)]

    segments = pick_segments(times, score, duration, cfg)
    if not segments:
        return [(0.0, cfg.budget)]
    total = sum(end - start for start, end in segments)
    log.info("%s: %.0fs -> %d highlight(s), %.1fs (%s)", Path(path).name, duration,
             len(segments), total,
             ", ".join(f"{s:.0f}-{e:.0f}s" for s, e in segments))
    return segments
