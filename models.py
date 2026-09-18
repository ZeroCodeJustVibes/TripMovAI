"""Shared data structures for the trip video pipeline.

Every phase reads and writes the same `timeline.json`, so the entry schema here
is the contract between phases. Unknown keys found on disk are preserved, which
keeps older timelines usable after the schema grows.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Iterator

PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".heic"}
VIDEO_EXTS = {".mp4", ".mov"}
MEDIA_EXTS = PHOTO_EXTS | VIDEO_EXTS


@dataclass
class GPS:
    lat: float
    lon: float

    def to_dict(self) -> dict[str, float]:
        return {"lat": self.lat, "lon": self.lon}

    @staticmethod
    def from_any(value: Any) -> "GPS | None":
        if value is None:
            return None
        if isinstance(value, GPS):
            return value
        if isinstance(value, dict):
            lat, lon = value.get("lat"), value.get("lon")
            if lat is None or lon is None:
                return None
            try:
                return GPS(float(lat), float(lon))
            except (TypeError, ValueError):
                return None
        return None


@dataclass
class TimelineEntry:
    """One media file, annotated progressively by each phase."""

    filepath: str
    type: str  # "photo" | "video"
    timestamp: str | None = None  # ISO8601
    gps: GPS | None = None
    duration: float = 0.0

    # Phase 3
    caption: str | None = None
    caption_confidence: float | None = None
    transcript: str | None = None

    # Phase 2
    quality_score: float | None = None
    blur_score: float | None = None
    brightness: float | None = None
    selected: bool | None = None
    drop_reason: str | None = None
    group: int | None = None

    # Phase 5 (reverse geocoded from gps, cached so it is looked up once)
    place: str | None = None

    # Phase 1 extras
    width: int | None = None
    height: int | None = None
    camera: str | None = None
    timestamp_source: str | None = None

    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    # -- convenience ------------------------------------------------------
    @property
    def is_video(self) -> bool:
        return self.type == "video"

    @property
    def name(self) -> str:
        return os.path.basename(self.filepath)

    def has_gps(self) -> bool:
        return self.gps is not None

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        extra = d.pop("extra", {}) or {}
        d["gps"] = self.gps.to_dict() if self.gps else None
        d.update(extra)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TimelineEntry":
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        kwargs = {k: v for k, v in d.items() if k in known}
        kwargs["gps"] = GPS.from_any(d.get("gps"))
        kwargs.setdefault("filepath", d.get("filepath", ""))
        kwargs.setdefault("type", d.get("type", "photo"))
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(extra=extra, **kwargs)


class Timeline:
    """An ordered collection of `TimelineEntry`, persisted as timeline.json."""

    def __init__(self, entries: Iterable[TimelineEntry] | None = None):
        self.entries: list[TimelineEntry] = list(entries or [])

    # -- container protocol ----------------------------------------------
    def __iter__(self) -> Iterator[TimelineEntry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> TimelineEntry:
        return self.entries[i]

    # -- lookups ----------------------------------------------------------
    def by_path(self, filepath: str) -> TimelineEntry | None:
        for e in self.entries:
            if e.filepath == filepath:
                return e
        # fall back to basename so a moved export still matches
        base = os.path.basename(filepath)
        for e in self.entries:
            if e.name == base:
                return e
        return None

    @property
    def selected(self) -> list[TimelineEntry]:
        return [e for e in self.entries if e.selected]

    # -- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump([e.to_dict() for e in self.entries], fh, indent=2, ensure_ascii=False)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | Path) -> "Timeline":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):  # tolerate {"entries": [...]}
            data = data.get("entries", [])
        return cls(TimelineEntry.from_dict(d) for d in data)

    @classmethod
    def load_if_exists(cls, path: str | Path) -> "Timeline | None":
        p = Path(path)
        if not p.exists():
            return None
        try:
            return cls.load(p)
        except (json.JSONDecodeError, OSError):
            return None


@dataclass
class ChapterMarker:
    at_clip: int
    title: str

    def to_dict(self) -> dict[str, Any]:
        return {"at_clip": self.at_clip, "title": self.title}


@dataclass
class NarrativePlan:
    """Phase 4 output: the edit decision list."""

    ordered_clips: list[str] = field(default_factory=list)
    captions: dict[str, str] = field(default_factory=dict)
    transition_hints: dict[str, str] = field(default_factory=dict)
    chapter_markers: list[ChapterMarker] = field(default_factory=list)
    mood: str = ""

    VALID_TRANSITIONS = ("crossfade", "cut", "fadein")

    def transition_for(self, filepath: str, default: str = "crossfade") -> str:
        hint = (self.transition_hints.get(filepath) or default).strip().lower()
        return hint if hint in self.VALID_TRANSITIONS else default

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordered_clips": self.ordered_clips,
            "captions": self.captions,
            "transition_hints": self.transition_hints,
            "chapter_markers": [c.to_dict() for c in self.chapter_markers],
            "mood": self.mood,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NarrativePlan":
        markers = []
        for m in d.get("chapter_markers") or []:
            if isinstance(m, dict) and "title" in m:
                try:
                    markers.append(ChapterMarker(int(m.get("at_clip", 0)), str(m["title"])))
                except (TypeError, ValueError):
                    continue
        return cls(
            ordered_clips=[str(p) for p in d.get("ordered_clips") or []],
            captions={str(k): str(v) for k, v in (d.get("captions") or {}).items() if v},
            transition_hints={str(k): str(v) for k, v in (d.get("transition_hints") or {}).items()},
            chapter_markers=markers,
            mood=str(d.get("mood") or ""),
        )

    @classmethod
    def chronological(cls, timeline: Timeline) -> "NarrativePlan":
        """Fallback plan: keep timeline order, no captions, gentle crossfades."""
        paths = [e.filepath for e in timeline.selected]
        return cls(ordered_clips=paths, transition_hints={p: "crossfade" for p in paths})

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)

    @classmethod
    def load_if_exists(cls, path: str | Path) -> "NarrativePlan | None":
        p = Path(path)
        if not p.exists():
            return None
        try:
            with open(p, "r", encoding="utf-8") as fh:
                return cls.from_dict(json.load(fh))
        except (json.JSONDecodeError, OSError):
            return None
