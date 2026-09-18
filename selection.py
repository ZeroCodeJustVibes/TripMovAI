"""Phase 2 - cut 200+ files down to the best ~50, automatically.

Named `selection.py`, not `select.py`: CPython ships a builtin module called
`select`, and builtins outrank sys.path, so a file of that name can never be
imported.

Three steps, in order:
  A. quality gate      - drop blurry, under/over-exposed and too-short clips
  B. deduplication     - CLIP embeddings, keep the sharpest of each near-dup set
  C. balanced pick     - group by place (DBSCAN on GPS, else time gaps),
                         then take a quality-weighted spread from each group

Everything is written back onto the timeline (`selected`, `quality_score`,
`blur_score`, `brightness`, `drop_reason`, `group`).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from mediautil import laplacian_variance, representative_pil, to_gray
from models import Timeline, TimelineEntry

log = logging.getLogger(__name__)

DARK_LIMIT = 30.0
BRIGHT_LIMIT = 225.0
MIN_VIDEO_SECONDS = 2.0
DEDUP_SIMILARITY = 0.88
GROUP_SHARE_CAP = 0.30  # no single location may own more than 30% of the cut
CLIP_MODEL = "openai/clip-vit-base-patch32"


@dataclass
class SelectionConfig:
    max_clips: int = 50
    blur_threshold: float = 80.0
    dedup_similarity: float = DEDUP_SIMILARITY
    min_video_seconds: float = MIN_VIDEO_SECONDS
    group_share_cap: float = GROUP_SHARE_CAP
    device: str = "cuda"


# ------------------------------------------------------ A. quality gate

def score_entry(entry: TimelineEntry, cfg: SelectionConfig) -> None:
    """Fill blur_score / brightness / quality_score, and drop_reason if bad."""
    if entry.is_video and entry.duration and entry.duration < cfg.min_video_seconds:
        entry.drop_reason = f"video shorter than {cfg.min_video_seconds:g}s"
        entry.quality_score = 0.0
        return

    img = representative_pil(entry.filepath, entry.is_video)
    if img is None:
        entry.drop_reason = "unreadable"
        entry.quality_score = 0.0
        return

    gray = to_gray(img)
    entry.blur_score = round(laplacian_variance(gray), 2)
    entry.brightness = round(float(gray.mean()), 2)
    if entry.width is None or entry.height is None:
        entry.width, entry.height = img.size

    # exposure first: a near-black frame is "too dark", not "blurry", even
    # though it also has almost no high-frequency detail left to measure
    if entry.brightness < DARK_LIMIT:
        entry.drop_reason = f"too dark ({entry.brightness:.0f})"
    elif entry.brightness > BRIGHT_LIMIT:
        entry.drop_reason = f"blown out ({entry.brightness:.0f})"
    elif entry.blur_score < cfg.blur_threshold:
        entry.drop_reason = f"blurry ({entry.blur_score:.0f} < {cfg.blur_threshold:g})"
    else:
        entry.drop_reason = None

    # 0..1: sharpness (log-scaled, saturating) blended with exposure centring
    sharp = min(1.0, math.log1p(max(entry.blur_score, 0.0)) / math.log1p(1500.0))
    exposure = 1.0 - min(1.0, abs(entry.brightness - 128.0) / 128.0)
    score = 0.7 * sharp + 0.3 * exposure
    if entry.is_video:
        score += 0.05  # videos add variety, give them a nudge
    entry.quality_score = round(min(1.0, score), 4)


def quality_gate(timeline: Timeline, cfg: SelectionConfig, progress_cb: Callable | None = None) -> list[TimelineEntry]:
    survivors: list[TimelineEntry] = []
    for i, entry in enumerate(timeline):
        try:
            score_entry(entry, cfg)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
            log.error("quality scoring failed for %s: %s", entry.name, exc)
            entry.drop_reason = "scoring error"
            entry.quality_score = 0.0
        entry.selected = False
        if entry.drop_reason is None:
            survivors.append(entry)
        if progress_cb:
            progress_cb(i + 1, len(timeline))
    log.info("quality gate: %d/%d kept", len(survivors), len(timeline))
    return survivors


# -------------------------------------------------------- B. dedup (CLIP)

def clip_embeddings(entries: Sequence[TimelineEntry], device: str = "cuda",
                    progress_cb: Callable | None = None) -> np.ndarray | None:
    """L2-normalised CLIP image embeddings, shape (n, d). None if CLIP fails."""
    if not entries:
        return None
    try:
        import torch
        from transformers import CLIPImageProcessor, CLIPModel
    except ImportError as exc:
        log.warning("CLIP unavailable (%s) - skipping deduplication", exc)
        return None

    try:
        model = CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval()
        processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL)
    except Exception as exc:  # noqa: BLE001 - offline / download failure
        log.warning("could not load %s (%s) - skipping deduplication", CLIP_MODEL, exc)
        return None

    vectors: list[np.ndarray] = []
    batch_size = 32
    try:
        with torch.no_grad():
            for start in range(0, len(entries), batch_size):
                chunk = entries[start : start + batch_size]
                images, index = [], []
                for i, e in enumerate(chunk):
                    img = representative_pil(e.filepath, e.is_video, max_side=384)
                    if img is not None:
                        images.append(img)
                        index.append(i)
                out = np.zeros((len(chunk), model.config.projection_dim), dtype=np.float32)
                if images:
                    inputs = processor(images=images, return_tensors="pt").to(device)
                    feats = model.get_image_features(**inputs)
                    # transformers>=5 returns the vision output object, with the
                    # projected embedding in `pooler_output`; older ones a tensor
                    feats = getattr(feats, "pooler_output", getattr(feats, "image_embeds", feats))
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    feats = feats.float().cpu().numpy()
                    for slot, vec in zip(index, feats):
                        out[slot] = vec
                vectors.append(out)
                if progress_cb:
                    progress_cb(min(start + batch_size, len(entries)), len(entries))
    except Exception as exc:  # noqa: BLE001
        log.warning("CLIP inference failed (%s) - skipping deduplication", exc)
        return None
    finally:
        try:
            import torch

            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    return np.vstack(vectors) if vectors else None


def deduplicate(entries: list[TimelineEntry], cfg: SelectionConfig,
                progress_cb: Callable | None = None) -> list[TimelineEntry]:
    """Collapse near-identical shots, keeping the sharpest of each set."""
    if len(entries) < 2:
        return entries
    embeddings = clip_embeddings(entries, cfg.device, progress_cb)
    if embeddings is None:
        return entries

    sim = embeddings @ embeddings.T
    order = sorted(range(len(entries)), key=lambda i: (entries[i].timestamp or "", entries[i].name))

    # Greedy clustering against cluster representatives rather than transitive
    # union-find: a slow pan across a valley is a chain of similar neighbours,
    # and chaining would collapse the whole pan into a single shot.
    clusters: list[list[int]] = []
    reps: list[int] = []
    for i in order:
        if not embeddings[i].any():
            clusters.append([i])
            reps.append(i)
            continue
        best, best_sim = -1, 0.0
        for c, rep in enumerate(reps):
            if embeddings[rep].any() and sim[i, rep] > best_sim:
                best, best_sim = c, float(sim[i, rep])
        if best >= 0 and best_sim >= cfg.dedup_similarity:
            clusters[best].append(i)
        else:
            clusters.append([i])
            reps.append(i)

    kept: list[TimelineEntry] = []
    dropped = 0
    def _best(indexes: list[int]) -> int:
        return max(indexes, key=lambda i: (entries[i].blur_score or 0.0,
                                           entries[i].quality_score or 0.0,
                                           entries[i].duration or 0.0))

    for members in clusters:
        # A video is never dropped in favour of a photo - it carries motion and
        # sound a still cannot - but duplicate videos of the same thing still
        # collapse to the best one. Immich exports routinely contain a
        # "name+1.mp4" twin of every clip, and both would otherwise be used.
        videos = [i for i in members if entries[i].is_video]
        photos = [i for i in members if not entries[i].is_video]
        winners = set()
        if videos:
            winners.add(_best(videos))
        if photos:
            winners.add(_best(photos))
        for i in members:
            if i in winners:
                kept.append(entries[i])
            else:
                entries[i].drop_reason = "near-duplicate"
                dropped += 1
    kept.sort(key=lambda e: (e.timestamp or "", e.name))
    log.info("deduplication: dropped %d near-duplicates, %d remain", dropped, len(kept))
    return kept


# ------------------------------------------------- C. balanced selection

def assign_groups(entries: list[TimelineEntry], eps_km: float = 1.0, time_gap_hours: float = 2.0) -> int:
    """Group entries by place (GPS/DBSCAN) or, without GPS, by time gaps."""
    located = [e for e in entries if e.has_gps()]
    n_groups = 0

    if len(located) >= 3:
        try:
            from sklearn.cluster import DBSCAN

            coords = np.radians([[e.gps.lat, e.gps.lon] for e in located])
            labels = DBSCAN(eps=eps_km / 6371.0, min_samples=2, metric="haversine").fit_predict(coords)
            next_label = int(labels.max()) + 1 if len(labels) else 0
            for e, label in zip(located, labels):
                if label < 0:  # DBSCAN noise - give each outlier its own group
                    e.group = next_label
                    next_label += 1
                else:
                    e.group = int(label)
            n_groups = next_label
        except Exception as exc:  # noqa: BLE001
            log.warning("GPS clustering failed (%s) - falling back to time gaps", exc)

    # anything still ungrouped (no GPS, or clustering failed) -> time-gap groups
    ungrouped = [e for e in entries if e.group is None]
    if ungrouped:
        ungrouped.sort(key=lambda e: (e.timestamp or "", e.name))
        label = n_groups
        previous: str | None = None
        for e in ungrouped:
            if previous and e.timestamp:
                try:
                    from dateutil import parser as dateparser

                    gap = (dateparser.parse(e.timestamp) - dateparser.parse(previous)).total_seconds() / 3600.0
                    if gap > time_gap_hours:
                        label += 1
                except (ValueError, TypeError):
                    pass
            e.group = label
            previous = e.timestamp or previous
        n_groups = label + 1
    return n_groups


def _spread_pick(candidates: list[TimelineEntry], quota: int) -> list[TimelineEntry]:
    """Take `quota` entries spread over the group's time range, best of each slice."""
    if quota <= 0 or not candidates:
        return []
    if quota >= len(candidates):
        return list(candidates)
    ordered = sorted(candidates, key=lambda e: (e.timestamp or "", e.name))
    picked: list[TimelineEntry] = []
    n = len(ordered)
    for k in range(quota):
        lo, hi = int(k * n / quota), max(int((k + 1) * n / quota), int(k * n / quota) + 1)
        window = [e for e in ordered[lo:hi] if e not in picked]
        if window:
            picked.append(max(window, key=lambda e: e.quality_score or 0.0))
    # a short window may have yielded nothing - top up by quality
    if len(picked) < quota:
        for e in sorted(ordered, key=lambda e: -(e.quality_score or 0.0)):
            if e not in picked:
                picked.append(e)
            if len(picked) == quota:
                break
    return picked


def balanced_select(entries: list[TimelineEntry], cfg: SelectionConfig) -> list[TimelineEntry]:
    if not entries:
        return []
    assign_groups(entries)
    groups: dict[int, list[TimelineEntry]] = {}
    for e in entries:
        groups.setdefault(e.group if e.group is not None else 0, []).append(e)

    target = min(cfg.max_clips, len(entries))
    chosen: list[TimelineEntry] = [e for e in entries if e.is_video]  # videos always ride along
    if len(chosen) > target:
        chosen = sorted(chosen, key=lambda e: -(e.quality_score or 0.0))[:target]
    chosen_set = set(id(e) for e in chosen)

    cap = max(1, int(math.floor(cfg.group_share_cap * target)))
    remaining = target - len(chosen)
    quotas: dict[int, int] = {}
    total_photos = sum(1 for e in entries if not e.is_video)

    for label, members in groups.items():
        photos = [e for e in members if not e.is_video]
        share = (len(photos) / total_photos) if total_photos else 0.0
        already = sum(1 for e in members if id(e) in chosen_set)
        quota = int(round(share * remaining))
        quotas[label] = max(0, min(quota, cap - already, len(photos)))

    # redistribute whatever the 30% cap and rounding left over
    def _shortfall() -> int:
        return remaining - sum(quotas.values())

    for _ in range(8):
        short = _shortfall()
        if short <= 0:
            break
        grew = False
        for label, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            if short <= 0:
                break
            photos = sum(1 for e in members if not e.is_video)
            already = sum(1 for e in members if id(e) in chosen_set)
            headroom = min(photos - quotas[label], cap - already - quotas[label])
            if headroom > 0:
                take = min(headroom, short)
                quotas[label] += take
                short -= take
                grew = True
        if not grew:
            break
    if _shortfall() < 0:  # trim overshoot, biggest quota first
        excess = -_shortfall()
        for label in sorted(quotas, key=lambda k: -quotas[k]):
            while excess > 0 and quotas[label] > 0:
                quotas[label] -= 1
                excess -= 1

    for label, members in groups.items():
        photos = [e for e in members if not e.is_video and id(e) not in chosen_set]
        chosen.extend(_spread_pick(photos, quotas.get(label, 0)))

    chosen.sort(key=lambda e: (e.timestamp or "", e.name))
    return chosen


# -------------------------------------------------------------- pipeline

def run(timeline: Timeline, cfg: SelectionConfig, progress: dict[str, Callable] | None = None) -> Timeline:
    """Run steps A-C and write `selected` back onto the timeline."""
    progress = progress or {}
    for e in timeline:
        e.selected = False
        e.drop_reason = None
        e.group = None

    survivors = quality_gate(timeline, cfg, progress.get("quality"))
    survivors = deduplicate(survivors, cfg, progress.get("clip"))
    chosen = balanced_select(survivors, cfg)

    for e in chosen:
        e.selected = True
        e.drop_reason = None
    for e in timeline:
        if not e.selected and e.drop_reason is None:
            e.drop_reason = "not needed for target clip count"

    log.info("selected %d clips (%d videos) from %d files",
             len(chosen), sum(1 for e in chosen if e.is_video), len(timeline))
    return timeline


def stats(timeline: Timeline) -> dict[str, object]:
    chosen = timeline.selected
    reasons: dict[str, int] = {}
    for e in timeline:
        if not e.selected and e.drop_reason:
            key = e.drop_reason.split(" (")[0]
            reasons[key] = reasons.get(key, 0) + 1
    return {
        "selected": len(chosen),
        "videos": sum(1 for e in chosen if e.is_video),
        "groups": len({e.group for e in chosen if e.group is not None}),
        "dropped_by": reasons,
    }
