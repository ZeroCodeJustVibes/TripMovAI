"""Music analysis - beats, energy and drops, so the cut can follow the track.

A slideshow whose cuts land anywhere in the bar feels accidental; the same
pictures cut on the beat feel edited. This module listens to the soundtrack
once and produces a `MusicMap`: where the beats and downbeats fall, how loud
and busy the track is over time, and where it drops.

Phase 6 uses that three ways: clip boundaries snap to beats, pacing follows the
energy (busier music, shorter shots), and a drop gets a hard cut instead of a
crossfade.

Analysis is cached next to the timeline, since it depends only on the music.
"""

from __future__ import annotations

import bisect
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

ANALYSIS_SR = 22050          # plenty for beat tracking, half the decoding cost
HOP = 512                    # ~23 ms at 22.05 kHz


@dataclass
class MusicMap:
    """What the soundtrack is doing, over time."""

    duration: float = 0.0
    tempo: float = 0.0
    beats: list[float] = field(default_factory=list)
    downbeats: list[float] = field(default_factory=list)
    drops: list[float] = field(default_factory=list)
    energy_times: list[float] = field(default_factory=list)
    energy: list[float] = field(default_factory=list)      # 0..1
    sources: list[str] = field(default_factory=list)

    @property
    def beat_interval(self) -> float:
        return 60.0 / self.tempo if self.tempo else 0.0

    def energy_at(self, when: float) -> float:
        """Loudness/busyness at a moment, 0 (quiet) to 1 (going off)."""
        if not self.energy:
            return 0.5
        index = min(len(self.energy) - 1,
                    max(0, bisect.bisect_left(self.energy_times, when)))
        return float(self.energy[index])

    def nearest_beat(self, when: float, tolerance: float,
                     prefer_downbeats: bool = False) -> float | None:
        """The beat closest to `when`, or None if none is near enough."""
        grid = self.downbeats if (prefer_downbeats and self.downbeats) else self.beats
        if not grid:
            return None
        index = bisect.bisect_left(grid, when)
        candidates = [grid[i] for i in (index - 1, index, index + 1) if 0 <= i < len(grid)]
        if not candidates:
            return None
        best = min(candidates, key=lambda b: abs(b - when))
        return best if abs(best - when) <= tolerance else None

    def is_drop(self, when: float, window: float = 0.35) -> bool:
        return any(abs(drop - when) <= window for drop in self.drops)

    def busy_threshold(self, percentile: float = 78.0) -> float:
        """The energy level this particular track counts as "going". Relative,
        not absolute: a mellow acoustic track never reaches 0.75, and would
        otherwise never get a single hard cut."""
        if not self.energy:
            return 1.0
        return float(np.percentile(np.asarray(self.energy), percentile))

    # -- persistence ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "duration": round(self.duration, 3),
            "tempo": round(self.tempo, 2),
            "beats": [round(b, 3) for b in self.beats],
            "downbeats": [round(b, 3) for b in self.downbeats],
            "drops": [round(d, 3) for d in self.drops],
            "energy_times": [round(t, 3) for t in self.energy_times],
            "energy": [round(e, 4) for e in self.energy],
            "sources": self.sources,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MusicMap":
        return cls(
            duration=float(data.get("duration") or 0.0),
            tempo=float(data.get("tempo") or 0.0),
            beats=[float(b) for b in data.get("beats") or []],
            downbeats=[float(b) for b in data.get("downbeats") or []],
            drops=[float(d) for d in data.get("drops") or []],
            energy_times=[float(t) for t in data.get("energy_times") or []],
            energy=[float(e) for e in data.get("energy") or []],
            sources=[str(s) for s in data.get("sources") or []],
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load_if_matching(cls, path: str | Path, sources: list[str]) -> "MusicMap | None":
        """Reuse a cached analysis only if it was made from the same tracks."""
        p = Path(path)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        cached = cls.from_dict(data)
        return cached if cached.sources == list(sources) and cached.beats else None


# ------------------------------------------------------------- analysis

def _downbeats(beats: list[float], strength: np.ndarray, times: np.ndarray,
               per_bar: int = 4) -> list[float]:
    """Pick the phase of the bar whose beats hit hardest, then take every 4th.

    Proper downbeat tracking needs a model; for cutting pictures it is enough
    to know which of the four offsets consistently carries the accent.
    """
    if len(beats) < per_bar * 2 or not len(strength):
        return list(beats[::per_bar])
    scores = []
    for offset in range(per_bar):
        picked = beats[offset::per_bar]
        indexes = np.clip(np.searchsorted(times, picked), 0, len(strength) - 1)
        scores.append(float(np.mean(strength[indexes])) if len(indexes) else 0.0)
    best = int(np.argmax(scores))
    return list(beats[best::per_bar])


def _find_drops(energy: np.ndarray, times: np.ndarray, beats: list[float],
                lead: float = 3.0, rise: float = 0.22) -> list[float]:
    """Moments where the track opens up: a quiet stretch followed by a loud one.

    Compares the average energy of the seconds before and after each candidate;
    a big jump is a drop. Snapped to the nearest beat, because that is where a
    cut has to land anyway.
    """
    if len(energy) < 8 or not len(times):
        return []
    opening = 5.0        # the track coming in is not a drop
    step = float(np.median(np.diff(times))) or 0.023
    span = max(1, int(lead / step))
    drops: list[float] = []
    for index in range(span, len(energy) - span):
        before = float(np.mean(energy[index - span:index]))
        after = float(np.mean(energy[index:index + span]))
        if after - before >= rise and after > 0.55:
            when = float(times[index])
            if when < opening:
                continue
            if drops and when - drops[-1] < lead * 1.5:
                continue
            if beats:
                nearest = min(beats, key=lambda b: abs(b - when))
                if abs(nearest - when) <= 0.4:
                    when = nearest
            drops.append(when)
    return drops


def analyse(paths: list[str], cache_path: str | Path | None = None,
            max_seconds: float = 900.0) -> MusicMap | None:
    """Listen to the soundtrack and describe it. None if analysis is unavailable.

    Several files are analysed as one continuous bed, matching how Phase 6
    plays them, so beat times stay valid across a playlist.
    """
    if not paths:
        return None
    sources = [str(Path(p).resolve()) for p in paths]
    if cache_path:
        cached = MusicMap.load_if_matching(cache_path, sources)
        if cached:
            log.info("reusing cached music analysis (%.0f BPM, %d beats)",
                     cached.tempo, len(cached.beats))
            return cached

    try:
        import librosa
    except ImportError:
        log.warning("librosa is not installed - the cut will not follow the music")
        return None

    try:
        chunks, total = [], 0.0
        for path in paths:
            samples, _ = librosa.load(path, sr=ANALYSIS_SR, mono=True)
            chunks.append(samples)
            total += len(samples) / ANALYSIS_SR
            if total >= max_seconds:
                break
        if not chunks:
            return None
        audio = np.concatenate(chunks)[: int(max_seconds * ANALYSIS_SR)]

        onset = librosa.onset.onset_strength(y=audio, sr=ANALYSIS_SR, hop_length=HOP)
        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset, sr=ANALYSIS_SR, hop_length=HOP, units="frames")
        beats = [float(t) for t in librosa.frames_to_time(beat_frames, sr=ANALYSIS_SR,
                                                          hop_length=HOP)]
        times = librosa.frames_to_time(np.arange(len(onset)), sr=ANALYSIS_SR, hop_length=HOP)

        rms = librosa.feature.rms(y=audio, hop_length=HOP)[0]
        loudness = librosa.amplitude_to_db(np.maximum(rms, 1e-8), ref=np.max)
        loud = np.clip((loudness + 45.0) / 45.0, 0.0, 1.0)          # -45 dB .. 0 dB -> 0..1
        busy = onset / (np.percentile(onset, 97) or 1.0)
        busy = np.clip(busy[: len(loud)], 0.0, 1.0)
        energy = np.clip(0.65 * loud[: len(busy)] + 0.35 * busy, 0.0, 1.0)
        smooth = max(3, int(1.5 / (float(np.median(np.diff(times))) or 0.023)))
        energy = np.convolve(energy, np.ones(smooth) / smooth, mode="same")

        tempo_value = float(np.atleast_1d(tempo)[0])
        music = MusicMap(
            duration=len(audio) / ANALYSIS_SR,
            tempo=tempo_value,
            beats=beats,
            downbeats=_downbeats(beats, onset, times),
            drops=_find_drops(energy, times, beats),
            energy_times=[float(t) for t in times[::4]],
            energy=[float(e) for e in energy[::4]],
            sources=sources,
        )
    except Exception as exc:  # noqa: BLE001 - a bad decode must not stop the film
        log.error("music analysis failed (%s) - the cut will not follow the music", exc)
        return None

    log.info("music: %.0f BPM, %d beats, %d downbeats, %d drop(s) over %.0fs",
             music.tempo, len(music.beats), len(music.downbeats), len(music.drops),
             music.duration)
    if cache_path:
        try:
            music.save(cache_path)
        except OSError as exc:
            log.warning("could not cache the music analysis: %s", exc)
    return music


def pacing_factor(music: MusicMap, when: float, strength: float,
                  low: float = 0.72, high: float = 1.34) -> float:
    """How long a shot at `when` should run, relative to normal.

    Busy passages get shorter shots and quiet ones get longer, which is most of
    what makes a cut feel like it belongs to the track. `strength` scales the
    whole effect from off (0) to full (1).
    """
    energy = music.energy_at(when)
    factor = high + (low - high) * max(0.0, min(1.0, energy))
    return 1.0 + (factor - 1.0) * max(0.0, min(1.0, strength))
