"""Phase 4 - ask an LLM (via OpenRouter) to edit the selected clips into a story.

The model gets the annotated timeline and returns an edit decision list: clip
order, a handful of short captions, transition hints and chapter markers.

The pipeline never depends on this succeeding - a parse failure, a missing API
key or a network error all fall back to chronological order with no captions.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

from models import NarrativePlan, Timeline, TimelineEntry

log = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-haiku-4-5"

SYSTEM_PROMPT = """You are a sharp, funny video editor cutting a 3-5 minute trip recap.

You receive a JSON list of clips in chronological order. Each clip has: id, file, type
(photo/video), time, place (a location cluster id, coordinates, and often the name of the
town or area), a machine-generated caption, and for videos a speech transcript.

IMPORTANT about `caption`: it comes from an image model that is frequently wrong - it
mistakes a lake for a desert, invents ferris wheels, and describes stock photos. Treat it
as a loose hint about mood and subject, never as fact. The `place` name and the time are
reliable; the caption is not.

Return ONLY a single JSON object - no markdown fences, no commentary - with exactly these keys:
{
  "ordered_clips": [<every clip id, in the order they should appear>],
  "captions": {"<clip id>": "<caption>"},
  "transition_hints": {"<clip id>": "crossfade" | "cut" | "fadein"},
  "chapter_markers": [{"at_clip": <index into ordered_clips>, "title": "<2-3 words>"}],
  "mood": "<a few words describing the overall feel>",
  "day_titles": {"<YYYY-MM-DD>": "<title for that whole day>"},
  "closing_quote": "<the line the film ends on>"
}

Rules:
- ordered_clips MUST contain every clip id exactly once. Keep broadly chronological order;
  only reorder to open strong, group a place together, or end on a memorable shot.
- captions: the voice is dry, warm and funny - the caption a friend writes over a holiday
  photo, not a travel brochure and not a description of what is visible. Think understated
  observation, gentle self-mockery, mock-heroic understatement of an ordinary moment.
  * Maximum 6 words. No final period. No emoji. No hashtags. No exclamation marks.
  * The exact number of captions to write is given below. Spread them through the film
    rather than bunching them, and leave the quiet, beautiful shots to breathe.
  * Because the machine caption may be wrong, never assert a specific object, animal,
    landmark or activity unless the place name or transcript backs it up. Write lines that
    stay funny even if the image is not quite what the caption claims.
  * Good: "Breakfast, eventually", "Nobody admits being lost", "Third pass, still uphill",
    "The weather had opinions", "Worth every step, allegedly".
  * Bad: "A beautiful mountain lake" (flat description), "LOL so random" (try-hard),
    "The majestic ferris wheel" (trusting a caption that may be hallucinated).
- transition_hints: "crossfade" for calm or related shots, "cut" for energy or a sharp
  change of subject, "fadein" for the first clip of a new place or chapter.
- chapter_markers: 2-5 of them, at indexes where the trip moves to a new place. Use the
  place name when there is one; these are only a fallback, since the video already labels
  locations from GPS.
- mood: a short phrase, e.g. "uplifting mountain adventure".
- day_titles: one for EVERY date listed below, in the same dry, funny voice as the
  captions. It titles the whole day, so it should catch that day's arc or its running
  joke rather than any single clip - what someone would say about the day afterwards.
  Maximum 5 words, no final period, no emoji, no "Day 1" prefix (the film adds that).
  Good: "Everything uphill, obviously", "The day of the fog", "Nobody packed enough socks".
- closing_quote: the line the film ends on, over the last shot. Warm rather than
  punchy - the one that makes them want to go back. Maximum 9 words, no attribution.
"""

CAPTION_TARGET_RULE = """
Caption EXACTLY {target} of the {count} clips - no fewer. That is the requested density,
so pick the {target} clips with the most comic potential and write a line for each.
"""

TOP_UP_PROMPT = """You wrote {have} captions but {target} were asked for.

Here are the clips you left uncaptioned. Write captions for {needed} of them, in the same
dry, funny voice, maximum 6 words each, no emoji, no final period. Remember the machine
captions are unreliable, so do not assert specific objects or landmarks.

Reply with ONLY a JSON object of the form {{"captions": {{"<clip id>": "<caption>"}}}}.

{clips}
"""

RETRY_SUFFIX = """

Your previous reply was not valid JSON. Reply with the raw JSON object ONLY.
Start your reply with { and end it with }. No prose, no markdown fences."""


@dataclass
class NarrativeConfig:
    label_fraction: float = 0.25      # share of clips that should carry a caption
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str = OPENROUTER_BASE_URL
    timeout: float = 120.0
    max_tokens: int = 8000
    temperature: float = 0.6


# --------------------------------------------------------------- payload

def _clip_payload(timeline: Timeline) -> tuple[list[dict[str, Any]], dict[str, TimelineEntry]]:
    """Compact, id-addressed view of the selected clips for the model.

    Short ids (`c0`, `c1`, ...) instead of absolute paths keep the prompt small
    and stop the model from inventing or mangling file paths.
    """
    payload, lookup = [], {}
    for i, e in enumerate(timeline.selected):
        cid = f"c{i}"
        lookup[cid] = e
        item: dict[str, Any] = {"id": cid, "file": e.name, "type": e.type}
        if e.timestamp:
            item["time"] = e.timestamp
        if e.group is not None:
            item["place"] = e.group
        if e.place:
            item["place_name"] = e.place
        if e.has_gps():
            item["coords"] = [round(e.gps.lat, 4), round(e.gps.lon, 4)]
        if e.is_video:
            item["duration"] = round(e.duration, 1)
        if e.caption:
            item["caption"] = e.caption
        if e.transcript:
            item["speech"] = e.transcript[:300]
        payload.append(item)
    return payload, lookup


def day_key(entry) -> str | None:
    """The calendar date a clip belongs to, as YYYY-MM-DD."""
    if not entry.timestamp:
        return None
    return str(entry.timestamp)[:10]


def trip_days(timeline: Timeline) -> list[str]:
    """Every date the selected clips fall on, in order."""
    seen = []
    for entry in timeline.selected:
        key = day_key(entry)
        if key and key not in seen:
            seen.append(key)
    return sorted(seen)


def _day_digest(timeline: Timeline) -> list[dict[str, Any]]:
    """A per-day summary for the model: where it was and how much happened."""
    days: dict[str, dict[str, Any]] = {}
    for entry in timeline.selected:
        key = day_key(entry)
        if not key:
            continue
        day = days.setdefault(key, {"date": key, "clips": 0, "videos": 0,
                                    "places": [], "captions": []})
        day["clips"] += 1
        day["videos"] += int(entry.is_video)
        if entry.place and entry.place not in day["places"]:
            day["places"].append(entry.place)
        if entry.caption and len(day["captions"]) < 6:
            day["captions"].append(entry.caption)
    return [days[k] for k in sorted(days)]


def _trip_context(timeline: Timeline) -> str:
    times = [e.timestamp for e in timeline.selected if e.timestamp]
    if not times:
        return ""
    try:
        first, last = datetime.fromisoformat(min(times)), datetime.fromisoformat(max(times))
    except ValueError:
        return ""
    days = (last.date() - first.date()).days + 1
    return f"The trip spans {days} day(s), from {first:%d %b %Y} to {last:%d %b %Y}."


# ---------------------------------------------------------------- client

def _extract_json(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a reply that may be wrapped in prose/fences."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise json.JSONDecodeError("no JSON object in reply", text, 0)


def call_openrouter(cfg: NarrativeConfig, messages: list[dict[str, str]],
                    json_mode: bool = True) -> str:
    key = cfg.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    body: dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    response = httpx.post(
        f"{cfg.base_url.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "X-Title": "trip-video-generator",
        },
        json=body,
        timeout=cfg.timeout,
    )
    response.raise_for_status()
    data = response.json()
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"unexpected OpenRouter response: {str(data)[:400]}") from exc


# ------------------------------------------------------------ validation

def _to_plan(raw: dict[str, Any], lookup: dict[str, TimelineEntry], timeline: Timeline) -> NarrativePlan:
    """Map the model's clip ids back to real paths, repairing what is off."""
    selected_paths = [e.filepath for e in timeline.selected]

    def resolve(cid: Any) -> str | None:
        key = str(cid).strip()
        if key in lookup:
            return lookup[key].filepath
        for entry in lookup.values():  # the model sometimes answers with filenames
            if key in (entry.name, entry.filepath):
                return entry.filepath
        return None

    ordered, seen = [], set()
    for cid in raw.get("ordered_clips") or []:
        path = resolve(cid)
        if path and path not in seen:
            ordered.append(path)
            seen.add(path)
    dropped = [p for p in selected_paths if p not in seen]
    if dropped:  # the model forgot some clips - append them chronologically
        log.warning("narrative plan omitted %d clip(s); appending them in order", len(dropped))
        ordered.extend(dropped)

    captions: dict[str, str] = {}
    for cid, text in (raw.get("captions") or {}).items():
        path = resolve(cid)
        if not path or not text:
            continue
        words = str(text).strip().strip('"').rstrip(".").split()
        if words:
            captions[path] = " ".join(words[:6])

    hints: dict[str, str] = {}
    for cid, hint in (raw.get("transition_hints") or {}).items():
        path = resolve(cid)
        if path and str(hint).strip().lower() in NarrativePlan.VALID_TRANSITIONS:
            hints[path] = str(hint).strip().lower()

    plan = NarrativePlan.from_dict({
        "ordered_clips": ordered,
        "captions": captions,
        "transition_hints": hints,
        "chapter_markers": raw.get("chapter_markers"),
        "mood": raw.get("mood"),
        "day_titles": raw.get("day_titles"),
        "closing_quote": raw.get("closing_quote"),
    })
    plan.day_titles = {k: " ".join(v.split()[:5]).rstrip(".")
                       for k, v in plan.day_titles.items() if v}
    plan.closing_quote = " ".join(plan.closing_quote.split()[:9]).rstrip(".")
    plan.chapter_markers = [c for c in plan.chapter_markers if 0 <= c.at_clip < len(ordered)]
    return plan


# -------------------------------------------------------------- pipeline

def plan_narrative(timeline: Timeline, cfg: NarrativeConfig) -> NarrativePlan:
    """Ask the LLM for an edit plan; fall back to chronological on any failure."""
    if not timeline.selected:
        return NarrativePlan()

    clips, lookup = _clip_payload(timeline)
    target = max(1, round(cfg.label_fraction * len(clips)))
    system_prompt = SYSTEM_PROMPT + CAPTION_TARGET_RULE.format(target=target, count=len(clips))
    days = _day_digest(timeline)
    user_message = (
        f"{_trip_context(timeline)}\n\n"
        f"The trip runs over these {len(days)} day(s) - write a day_title for each date:\n"
        f"{json.dumps(days, ensure_ascii=False, indent=1)}\n\n"
        f"Here are the {len(clips)} selected clips:\n"
        f"{json.dumps(clips, ensure_ascii=False, indent=1)}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    json_mode = True
    for attempt in (1, 2):
        try:
            reply = call_openrouter(cfg, messages, json_mode=json_mode)
            raw = _extract_json(reply)
            plan = _to_plan(raw, lookup, timeline)
            if len(plan.captions) < target:
                _top_up_captions(plan, cfg, lookup, timeline, target, json_mode)
            log.info("narrative plan: %d clips, %d/%d captions, mood=%r",
                     len(plan.ordered_clips), len(plan.captions), target, plan.mood)
            return plan
        except json.JSONDecodeError as exc:
            log.warning("attempt %d: model did not return valid JSON (%s)", attempt, exc)
            if attempt == 1:
                messages = [
                    {"role": "system", "content": system_prompt + RETRY_SUFFIX},
                    {"role": "user", "content": user_message},
                ]
                continue
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:300] if exc.response is not None else ""
            status = exc.response.status_code if exc.response is not None else 0
            # not every model accepts response_format - retry once without it
            if json_mode and attempt == 1 and status in (400, 404, 422):
                log.warning("model rejected JSON mode (%s) - retrying without it", status)
                json_mode = False
                continue
            log.error("OpenRouter returned %s: %s", status, detail)
            break
        except Exception as exc:  # noqa: BLE001 - network, auth, anything
            log.error("narrative planning failed: %s", exc)
            break

    log.warning("falling back to chronological order with no captions")
    return NarrativePlan.chronological(timeline)


def _top_up_captions(plan: NarrativePlan, cfg: NarrativeConfig, lookup: dict[str, TimelineEntry],
                     timeline: Timeline, target: int, json_mode: bool) -> None:
    """Ask once more for the shortfall when the model under-delivers captions.

    `--amount-labels` is a floor, not a hint, and models routinely write three
    captions when asked for twelve. One extra call is cheaper than shipping a
    film with a third of the jokes missing.
    """
    remaining = [(cid, e) for cid, e in lookup.items() if e.filepath not in plan.captions]
    needed = min(target - len(plan.captions), len(remaining))
    if needed <= 0:
        return

    listing = json.dumps([
        {"id": cid, "file": e.name, "caption": e.caption or "", "place_name": e.place or ""}
        for cid, e in remaining
    ], ensure_ascii=False, indent=1)
    prompt = TOP_UP_PROMPT.format(have=len(plan.captions), target=target,
                                  needed=needed, clips=listing)
    try:
        reply = call_openrouter(cfg, [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ], json_mode=json_mode)
        extra = _extract_json(reply).get("captions") or {}
    except Exception as exc:  # noqa: BLE001 - a short plan beats no plan
        log.warning("caption top-up failed (%s) - keeping %d captions", exc, len(plan.captions))
        return

    added = 0
    for cid, text in extra.items():
        entry = lookup.get(str(cid).strip())
        if entry is None or entry.filepath in plan.captions or not text:
            continue
        words = str(text).strip().strip('"').rstrip(".").split()
        if words:
            plan.captions[entry.filepath] = " ".join(words[:6])
            added += 1
    log.info("caption top-up added %d caption(s)", added)


# ------------------------------------------------------------ interactive

def review_captions(plan: NarrativePlan, timeline: Timeline) -> NarrativePlan:
    """`--interactive`: accept / edit / delete each caption, and add missing ones."""
    if not plan.ordered_clips:
        return plan
    print("\nCaption review - Enter keeps, new text replaces, '-' deletes, 'q' finishes.\n")
    for index, path in enumerate(plan.ordered_clips):
        entry = timeline.by_path(path)
        current = plan.captions.get(path, "")
        label = entry.name if entry else os.path.basename(path)
        hint = f"  ({entry.caption})" if entry and entry.caption and not current else ""
        try:
            answer = input(f"[{index + 1}/{len(plan.ordered_clips)}] {label}{hint}\n    caption [{current}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n(caption review cancelled)")
            break
        if answer.lower() == "q":
            break
        if not answer:
            continue
        if answer == "-":
            plan.captions.pop(path, None)
        else:
            plan.captions[path] = " ".join(answer.split()[:6])
    return plan
