"""Phase 6 - cut the clips, captions, map and music into the finished film.

Visual layout per clip: the photo or video is fitted (letterboxed, never
cropped) into the frame, the map frame is composited according to `--map-mode`,
and a caption - if the narrative plan wrote one - sits in the bottom third.

Audio is built by hand rather than left to MoviePy so volumes stay predictable:
one music bed, ducked to 15% under clips where Whisper heard speech, plus the
original audio of each video clip (full for speech, 40% otherwise).
"""

from __future__ import annotations

import logging
import math
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from mediautil import extracted_audio, nvenc_available, playable_video, representative_pil
from models import NarrativePlan, Timeline, TimelineEntry

log = logging.getLogger(__name__)

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]


@dataclass
class AssembleConfig:
    photo_duration: float = 3.5
    max_video_duration: float = 15.0   # total seconds used from any one video
    use_highlights: bool = True        # play the best stretches, not the opening
    remux_dir: Path | None = None      # where repaired containers are cached
    audio_dir: Path | None = None      # where extracted clip audio is cached

    # --- rendering ------------------------------------------------------
    chunk_size: int = 8                # shots per independently rendered run
    workers: int = 4                   # how many runs render at once
    crossfade: float = 0.5
    map_mode: str = "overlay"          # overlay | side | none
    map_opacity: float = 0.35
    # The UI column: wide enough that a full-height landscape shot sits exactly
    # beside it, and the map fills it rather than floating as a small badge.
    column_fraction: float = 0.24      # of frame width
    map_width_fraction: float = 0.24   # the map fills the column
    pair_portraits: bool = True        # show two tall photos side by side
    pair_duration_scale: float = 1.25  # a spread earns a little more time
    pair_lookahead: int = 6            # how far ahead to find a partner

    # --- chapters, finale and motion ------------------------------------
    day_cards: bool = True             # a title card at the start of each day
    day_card_duration: float = 4.0
    day_card_images: int = 3           # up to this many photos on a day card
    day_card_gallery_height: float = 0.34   # of frame height
    day_card_map_fraction: float = 0.11     # of frame width - smaller than the film's
    day_card_title: float = 0.072           # title size, of frame height
    closing_card: bool = True          # quote + the whole route + corner photos
    closing_card_duration: float = 6.0
    closing_card_images: int = 4       # small photos, one per corner
    closing_map_fraction: float = 0.52  # of frame height - the map is the hero here
    closing_quote_size: float = 0.056   # of frame height
    closing_corner_width: float = 0.15  # of frame width
    animate_cards: bool = True         # ease each picture in and out
    anim_in: float = 0.55
    anim_out: float = 0.45
    anim_from: float = 0.90            # entering, the picture rises from behind
    anim_to: float = 1.05              # leaving, it drifts past the viewer

    # --- composition ----------------------------------------------------
    # One grid for everything: the footage sits on a rounded card inset by
    # `frame_margin`, and every piece of UI hangs off `ui_margin`, which is
    # fixed to the frame so the map and caption never shift between clips.
    frame_margin: float = 0.024        # of frame width, around the media card
    ui_margin: float = 0.045           # of frame width, for map and caption
    card_radius: float = 0.014         # of frame width
    card_border: tuple[int, int, int, int] = (255, 255, 255, 64)
    card_border_width: float = 0.0013  # of frame width
    card_shadow_alpha: int = 150
    backdrop_blur: float = 0.022       # of frame width
    backdrop_dim: float = 0.52         # 1.0 keeps it as bright as the photo
    backdrop_saturation: float = 0.72
    side_panel_fraction: float = 0.30  # of frame width, side mode
    resolution: tuple[int, int] | None = None
    fps: int = 30
    codec: str = "libx264"
    encoder: str = "auto"              # auto | x264 | nvenc
    audio_codec: str = "aac"
    bitrate: str | None = None
    preset: str = "medium"
    threads: int = 0
    music_volume: float = 1.0
    music_crossfade: float = 2.5      # overlap between consecutive songs
    target_duration: float | None = None   # stretch/squeeze photos to this length

    # --- cutting to the music -------------------------------------------
    beat_sync: bool = True             # land cuts on beats
    beat_sync_strength: float = 0.7    # how far pacing follows the energy, 0..1
    beat_tolerance: float = 0.32       # seconds a cut may move to reach a beat
    drop_punch: float = 0.80           # entrance scale on a drop (smaller = harder hit)
    min_photo_duration: float = 1.6
    max_photo_duration: float = 9.0
    # Ducking, not muting: the bed stays present under speech. -8 dB is enough
    # for a voice to sit on top while the track is still clearly playing;
    # the old -16.5 dB left the music barely there.
    duck_volume: float = 0.40
    ambient_volume: float = 0.45
    duck_fade: float = 0.4

    # loudness matching: phone clips arrive anywhere from -25 to -64 dBFS
    normalize_audio: bool = True
    clip_target_db: float = -18.0     # where a clip's own audio should sit
    music_target_db: float = -20.0    # the bed, a couple of dB under the clips
    max_gain_db: float = 18.0         # never amplify hiss into a roar
    silence_floor_db: float = -45.0   # below this a clip is silence, not quiet
    headroom_db: float = 1.0
    show_location: bool = True
    ffmpeg_params: list[str] = field(default_factory=lambda: ["-movflags", "+faststart"])


# ------------------------------------------------------------ frame setup

def target_resolution(timeline: Timeline, entries: Sequence[TimelineEntry] | None = None) -> tuple[int, int]:
    """The most common resolution among the clips, rounded to even numbers."""
    pool = entries if entries is not None else (timeline.selected or list(timeline))
    sizes = Counter((e.width, e.height) for e in pool if e.width and e.height)
    if not sizes:
        log.warning("no resolution metadata - defaulting to 1920x1080")
        return (1920, 1080)
    (width, height), _ = sizes.most_common(1)[0]
    width, height = int(width), int(height)
    if width * height > 3840 * 2160:  # keep 4K exports off a 10 GB card's back
        scale = math.sqrt((3840 * 2160) / (width * height))
        width, height = int(width * scale), int(height * scale)
    return (width - width % 2, height - height % 2)


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1 scales this
    except TypeError:
        return ImageFont.load_default()


@dataclass
class Layout:
    """Where everything goes, derived once per clip from the frame and media.

    `panels` holds one rect per picture - two when a pair of portrait photos
    share the clip - and `card` is their bounding box. The UI column is fixed,
    so the map and caption never shift between clips.
    """

    frame: tuple[int, int]
    panels: list[tuple[int, int, int, int]]
    card: tuple[int, int, int, int]
    radius: int
    border: int
    ui_left: int
    ui_top: int
    ui_bottom: int
    ui_right: int
    gap: int

    @property
    def ui_width(self) -> int:
        return max(80, self.ui_right - self.ui_left)


def layout_for(media_sizes, frame_size: tuple[int, int], cfg: AssembleConfig,
               with_map: bool = True) -> Layout:
    """Lay out one or two pictures beside the UI column.

    With a map on screen the column is reserved first and the picture takes
    everything left over - which is what makes a full-height landscape shot sit
    flush beside the map instead of the map shrinking to a badge. Without a map
    there is nothing to reserve, so the picture uses the whole frame and the
    caption sits on it.
    """
    if isinstance(media_sizes, tuple):
        media_sizes = [media_sizes]
    media_sizes = [m for m in media_sizes if m and m[0] > 0 and m[1] > 0] or [(4, 3)]

    width, height = frame_size
    margin = round(cfg.frame_margin * width)
    ui = round(cfg.ui_margin * width)
    gap = round(cfg.ui_margin * width * 0.4)
    area_x, area_y, area_w, area_h = _video_area(frame_size, cfg)

    column = round(cfg.column_fraction * width) if (with_map and cfg.map_mode == "overlay") else 0
    picture_x = area_x + margin + (column + gap if column else 0)
    picture_w = max(80, area_x + area_w - margin - picture_x)
    picture_h = max(60, area_h - 2 * margin)

    aspects = [m[0] / m[1] for m in media_sizes]
    inner_gap = gap if len(aspects) > 1 else 0
    panel_h = min(picture_h, (picture_w - inner_gap) / sum(aspects))
    widths = [max(8, round(panel_h * a)) for a in aspects]
    panel_h = max(8, round(panel_h))

    total_w = sum(widths) + inner_gap
    x = picture_x + (picture_w - total_w) // 2
    y = area_y + (area_h - panel_h) // 2
    panels, cursor = [], x
    for w in widths:
        panels.append((cursor, y, w, panel_h))
        cursor += w + inner_gap
    card = (x, y, total_w, panel_h)

    if column:
        ui_left, ui_top, ui_bottom = area_x + margin, area_y + margin, area_y + area_h - margin
        ui_right = ui_left + column
    else:                      # no column: the UI sits on the picture itself
        ui_left = max(area_x + ui, x + gap)
        ui_top = max(ui, y + gap)
        ui_bottom = min(height - ui, y + panel_h - gap)
        ui_right = min(area_x + area_w - ui, x + total_w - gap)

    return Layout(
        frame=frame_size,
        panels=panels,
        card=card,
        radius=max(6, round(cfg.card_radius * width)),
        border=max(1, round(cfg.card_border_width * width)),
        ui_left=ui_left,
        ui_top=ui_top,
        ui_bottom=ui_bottom,
        ui_right=ui_right,
        gap=gap,
    )


def _rounded_mask(size: tuple[int, int], radius: int) -> Image.Image:
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size[0] - 1, size[1] - 1],
                                           radius=radius, fill=255)
    return mask


def backdrop(source: Image.Image, layout: Layout, cfg: AssembleConfig) -> Image.Image:
    """Blurred, dimmed cover-fill of the shot itself, plus the card's shadow.

    Letterboxing a portrait phone clip onto flat black leaves two dead bars;
    filling them with a defocused copy of the same frame reads as depth and
    keeps the colour of the shot.
    """
    width, height = layout.frame
    scale = max(width / source.width, height / source.height)
    scaled = source.resize((max(1, int(source.width * scale + 1)),
                            max(1, int(source.height * scale + 1))), Image.LANCZOS)
    left, top = (scaled.width - width) // 2, (scaled.height - height) // 2
    plate = scaled.crop((left, top, left + width, top + height)).convert("RGB")
    plate = plate.filter(ImageFilter.GaussianBlur(max(2.0, cfg.backdrop_blur * width)))
    plate = ImageEnhance.Color(plate).enhance(cfg.backdrop_saturation)
    plate = ImageEnhance.Brightness(plate).enhance(cfg.backdrop_dim)

    spread = max(4, int(layout.radius * 0.9))
    shadow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    for x, y, w, h in layout.panels:
        shadow_draw.rounded_rectangle(
            [x - spread // 2, y - spread // 3, x + w + spread // 2, y + h + spread],
            radius=layout.radius + spread // 2, fill=(0, 0, 0, cfg.card_shadow_alpha))
    shadow = shadow.filter(ImageFilter.GaussianBlur(spread))
    return Image.alpha_composite(plate.convert("RGBA"), shadow)


def card_border_overlay(layout: Layout, cfg: AssembleConfig) -> Image.Image:
    """A hairline around the media card, drawn above whatever fills it."""
    overlay = Image.new("RGBA", layout.frame, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for x, y, w, h in layout.panels:
        draw.rounded_rectangle([x, y, x + w - 1, y + h - 1], radius=layout.radius,
                               outline=cfg.card_border, width=layout.border)
    return overlay


def _fit(size: tuple[int, int], box: tuple[int, int]) -> tuple[int, int]:
    """Largest size fitting inside `box`, preserving aspect (even numbers)."""
    scale = min(box[0] / size[0], box[1] / size[1])
    w, h = max(2, int(size[0] * scale)), max(2, int(size[1] * scale))
    return (w - w % 2, h - h % 2)


# --------------------------------------------------------------- captions

def _wrap(text: str, font: ImageFont.FreeTypeFont, max_width: int,
          draw: ImageDraw.ImageDraw, max_lines: int | None = None) -> list[str]:
    """Greedy word wrap. Without `max_lines` nothing is dropped - callers that
    need a hard limit should go through `fit_text`, which shrinks first."""
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=font) <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:max_lines] if max_lines else lines


def _ellipsize(line: str, font: ImageFont.FreeTypeFont, max_width: int,
               draw: ImageDraw.ImageDraw) -> str:
    if draw.textlength(line, font=font) <= max_width:
        return line
    while line and draw.textlength(line + "…", font=font) > max_width:
        line = line[:-1]
    return (line.rstrip() + "…") if line else ""


def fit_text(text: str, max_width: int, max_lines: int, draw: ImageDraw.ImageDraw,
             start_size: int, min_size: int) -> tuple[ImageFont.FreeTypeFont, list[str]]:
    """Lay out `text` so it genuinely fits, shrinking the type rather than
    clipping it.

    Wrapping alone silently drops whatever does not fit in the last line, which
    is how "Matrei in Osttirol · 2,177 m" lost its unit and a long day title
    lost its ending. This steps the size down until the text fits the box, and
    only ellipsizes if even the smallest size cannot hold it.
    """
    size = max(start_size, min_size)
    while size > min_size:
        font = _load_font(size)
        lines = _wrap(font=font, text=text, max_width=max_width, draw=draw)
        if len(lines) <= max_lines and all(
                draw.textlength(line, font=font) <= max_width for line in lines):
            return font, lines
        size -= max(1, round(size * 0.06))

    font = _load_font(min_size)
    lines = _wrap(text, font, max_width, draw)[:max_lines]
    if lines:
        lines[-1] = _ellipsize(lines[-1], font, max_width, draw)
    return font, [line for line in lines if line]


def location_label(entry: TimelineEntry) -> str | None:
    """"Kitzbuehel · 1,878 m", falling back to raw coordinates without a name."""
    if entry.place:
        label = entry.place
    elif entry.has_gps():
        label = f"{entry.gps.lat:.4f}, {entry.gps.lon:.4f}"
    else:
        return None
    if entry.altitude is not None:
        label = f"{label} · {entry.altitude:,.0f} m"
    return label


def _draw_pin_icon(draw: ImageDraw.ImageDraw, x: float, y: float, size: float,
                   color=(235, 92, 48, 245)) -> float:
    """A small map pin, drawn rather than typed - the bundled fonts have no
    emoji glyphs, so a literal marker character would render as a blank box.
    Returns the width it used."""
    r = size * 0.34
    cx, cy = x + r, y + r
    draw.ellipse([cx - r - 1, cy - r - 1, cx + r + 1, cy + r + 1], fill=(0, 0, 0, 120))
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    draw.polygon([(cx - r * 0.62, cy + r * 0.55), (cx + r * 0.62, cy + r * 0.55),
                  (cx, cy + r * 1.85)], fill=color)
    draw.ellipse([cx - r * 0.34, cy - r * 0.34, cx + r * 0.34, cy + r * 0.34],
                 fill=(255, 255, 255, 235))
    return r * 2 + size * 0.22


def caption_overlay(text: str, layout: Layout, cfg: AssembleConfig,
                    location: str | None = None, eyebrow: str | None = None,
                    emphasis: float = 1.0) -> Image.Image | None:
    """One card, bottom-left: the caption with the location line beneath it.

    Caption and location used to be separate floating elements with different
    treatments. Sharing a container - same corner radius and hairline as the
    media card - is what makes the frame read as arranged rather than stacked.
    """
    if not text and not location and not eyebrow:
        return None
    width, height = layout.frame
    overlay = Image.new("RGBA", layout.frame, (0, 0, 0, 0))
    measure = ImageDraw.Draw(overlay)

    pad_x, pad_y = round(height * 0.026), round(height * 0.020)
    inner_w = max(120, min(layout.ui_width, round(width * 0.52)) - 2 * pad_x)

    caption_font, lines = (None, [])
    if text:
        caption_font, lines = fit_text(text, inner_w, 3, measure,
                                       max(15, round(height * 0.036 * emphasis)),
                                       max(11, round(height * 0.022)))
    if caption_font is None:
        caption_font = _load_font(max(15, round(height * 0.036 * emphasis)))
    caption_lh = round((caption_font.getbbox("Ag")[3] - caption_font.getbbox("Ag")[1]) * 1.5)
    eyebrow_font = _load_font(max(11, round(height * 0.024))) if eyebrow else None
    eyebrow_h = round((eyebrow_font.getbbox("Ag")[3] - eyebrow_font.getbbox("Ag")[1]) * 1.9) if eyebrow else 0
    icon = max(11.0, height * 0.030)

    content_w = max([measure.textlength(line, font=caption_font) for line in lines] or [0])

    # "Matrei in Osttirol · 2,177 m" is wider than the column at full size, and
    # a clipped elevation reads as a bug, so step the location line down until
    # it fits rather than letting it run off the card.
    loc_font, loc_w = None, 0.0
    if location:
        for points in range(round(height * 0.027), max(10, round(height * 0.017)) - 1, -1):
            loc_font = _load_font(points)
            loc_w = icon + height * 0.008 + measure.textlength(location, font=loc_font)
            if loc_w <= inner_w:
                break
    loc_h = (loc_font.getbbox("Ag")[3] - loc_font.getbbox("Ag")[1]) if loc_font else 0
    content_w = max(content_w, loc_w)

    card_w = round(min(content_w + 2 * pad_x, layout.ui_width))
    if eyebrow_font is not None:
        content_w = max(content_w, measure.textlength(eyebrow, font=eyebrow_font))
    card_h = pad_y * 2 + caption_lh * len(lines) + eyebrow_h
    if location:
        card_h += round(max(loc_h, icon) * 1.15) + (round(height * 0.012) if lines else 0)
    card_x, card_y = layout.ui_left, layout.ui_bottom - card_h

    card = Image.new("RGBA", (card_w, card_h), (0, 0, 0, 0))
    card_draw = ImageDraw.Draw(card)
    radius = max(6, round(layout.radius * 0.8))
    card_draw.rounded_rectangle([0, 0, card_w - 1, card_h - 1], radius=radius, fill=(12, 14, 18, 168))
    card_draw.rounded_rectangle([0, 0, card_w - 1, card_h - 1], radius=radius,
                                outline=cfg.card_border, width=max(1, layout.border))

    y = pad_y
    if eyebrow_font is not None:
        card_draw.text((pad_x, y), eyebrow, font=eyebrow_font, fill=(235, 92, 48, 245))
        y += eyebrow_h
    for line in lines:
        card_draw.text((pad_x, y), line, font=caption_font, fill=(255, 255, 255, 244))
        y += caption_lh
    if location:
        if lines:
            y += round(height * 0.012)
            card_draw.line([pad_x, y - round(height * 0.007), card_w - pad_x, y - round(height * 0.007)],
                           fill=(255, 255, 255, 38), width=1)
        used = _draw_pin_icon(card_draw, pad_x, y, icon)
        card_draw.text((pad_x + used, y), location, font=loc_font, fill=(235, 238, 245, 224))

    overlay.alpha_composite(card, (card_x, card_y))
    return overlay


# -------------------------------------------------------------- map frame

def map_geometry(layout: Layout, cfg: AssembleConfig) -> tuple[int, tuple[int, int]]:
    """(parked display size, top-left position) for the map in the current mode.

    Overlay mode keeps it small in the **top-left**, on the same rail as the
    caption below it, so the two line up on one left edge. It grows from that
    anchor while the view travels, so the corner it occupies never moves.
    Opacity is already baked into the PNG by Phase 5.
    """
    width, height = layout.frame
    if cfg.map_mode == "side":
        panel = int(width * cfg.side_panel_fraction)
        size = min(panel - int(width * 0.03), height - int(height * 0.06))
        return size, (width - panel + (panel - size) // 2, (height - size) // 2)
    size = min(layout.ui_width, round(width * cfg.map_width_fraction))
    return size, (layout.ui_left, layout.ui_top)


def map_overlay(map_path: Path, layout: Layout, cfg: AssembleConfig) -> Image.Image | None:
    """A still map frame placed on a transparent full-frame layer."""
    try:
        map_img = Image.open(map_path).convert("RGBA")
    except Exception as exc:  # noqa: BLE001
        log.warning("cannot read map frame %s: %s", map_path, exc)
        return None
    size, position = map_geometry(layout, cfg)
    overlay = Image.new("RGBA", layout.frame, (0, 0, 0, 0))
    overlay.alpha_composite(map_img.resize((size, size), Image.LANCZOS), position)
    return overlay


def map_animation_clip(track, layout: Layout, cfg: AssembleConfig, duration: float):
    """A moving, growing map as a positioned clip, holding its last frame.

    The clip's canvas is fixed at the largest size the map reaches, and each
    frame is drawn into it scaled and anchored top-left. MoviePy needs every
    frame of a clip to be the same shape, so the swell has to happen *inside*
    a constant canvas rather than by resizing the clip itself.

    Only clips where the trip changes location get one; everywhere else the map
    is a still that gets flattened into the photo, which is far cheaper.
    """
    from moviepy import VideoClip

    size, position = map_geometry(layout, cfg)
    canvas = max(size, int(size * track.max_scale))
    canvas = min(canvas, min(layout.frame) - 2)   # never larger than the frame

    rgb, alpha = [], []
    for index, path in enumerate(track.frames):
        try:
            frame = Image.open(path).convert("RGBA")
        except Exception as exc:  # noqa: BLE001
            log.warning("cannot read map frame %s: %s", path, exc)
            continue
        drawn = max(8, min(canvas, int(round(size * track.scale_at(index)))))
        slot = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
        slot.alpha_composite(frame.resize((drawn, drawn), Image.LANCZOS), (0, 0))
        rgb.append(np.array(slot.convert("RGB")))
        alpha.append(np.array(slot.getchannel("A"), dtype=np.float32) / 255.0)
    if not rgb:
        return None

    fps, last = max(track.fps, 1.0), len(rgb) - 1

    def index(t: float) -> int:
        return min(int(t * fps), last)

    clip = VideoClip(lambda t: rgb[index(t)], duration=duration)
    clip = clip.with_mask(VideoClip(lambda t: alpha[index(t)], is_mask=True, duration=duration))
    return clip.with_position(position)


# ------------------------------------------------------------ clip builder

def _video_area(frame_size: tuple[int, int], cfg: AssembleConfig) -> tuple[int, int, int, int]:
    """(x, y, w, h) of the region the footage may occupy."""
    width, height = frame_size
    if cfg.map_mode == "side":
        usable = int(width * (1 - cfg.side_panel_fraction))
        return (0, 0, usable, height)
    return (0, 0, width, height)


def day_card_layout(sizes: list[tuple[int, int]], frame_size: tuple[int, int],
                    cfg: AssembleConfig, text_height: int = 0) -> dict:
    """Geometry for a day card: photo band, centred text, small map beneath.

    A day card is a different beat from the clips around it, so it gets its own
    composition rather than the film's column: the title is the hero and sits
    in the middle. Everything stacks down one centre line - the map included -
    because a map parked in a corner collides with a title the moment that
    title needs a third line.
    """
    width, height = frame_size
    margin = round(cfg.frame_margin * width)
    gap = round(cfg.ui_margin * width * 0.4)

    aspects = [w / h for w, h in sizes] or [4 / 3]
    row_gap = gap if len(aspects) > 1 else 0
    max_h = height * cfg.day_card_gallery_height
    max_w = width - 2 * margin - row_gap * (len(aspects) - 1)
    panel_h = round(min(max_h, max_w / sum(aspects)))
    widths = [max(8, round(panel_h * a)) for a in aspects]
    row_w = sum(widths) + row_gap * (len(widths) - 1)

    map_size = round(width * cfg.day_card_map_fraction)
    text_gap = round(height * 0.042)
    map_gap = round(height * 0.030)
    text_height = text_height or round(height * 0.22)

    block = panel_h + text_gap + text_height + map_gap + map_size
    top = max(margin, round((height - block) / 2))
    # if a long title pushes the stack past the frame, take it off the photos
    overflow = (top + block) - (height - margin)
    if overflow > 0:
        panel_h = max(round(height * 0.16), panel_h - overflow)
        widths = [max(8, round(panel_h * a)) for a in aspects]
        row_w = sum(widths) + row_gap * (len(widths) - 1)
        block = panel_h + text_gap + text_height + map_gap + map_size
        top = max(margin, round((height - block) / 2))

    panels, cursor = [], round((width - row_w) / 2)
    for w in widths:
        panels.append((cursor, top, w, panel_h))
        cursor += w + row_gap

    text_top = top + panel_h + text_gap
    return {
        "panels": panels,
        "row": (round((width - row_w) / 2), top, row_w, panel_h),
        "text_top": text_top,
        "text_width": round(width * 0.66),
        "map": (round((width - map_size) / 2), text_top + text_height + map_gap,
                map_size, map_size),
        "radius": max(6, round(cfg.card_radius * width)),
        "border": max(1, round(cfg.card_border_width * width)),
    }


def day_card_text_height(shot: "Shot", text_width: int, frame_size: tuple[int, int],
                         cfg: AssembleConfig) -> int:
    """How tall the title block actually is, so the stack can be laid out
    around it rather than assuming a fixed height."""
    width, height = frame_size
    draw = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    total = 0
    if shot.eyebrow:
        font = _load_font(max(12, round(height * 0.026)))
        total += round((font.getbbox("Ag")[3] - font.getbbox("Ag")[1]) * 1.42) + round(height * 0.012)
    if shot.caption:
        font, lines = fit_text(shot.caption, text_width, 3, draw,
                               max(20, round(height * cfg.day_card_title)),
                               max(16, round(height * 0.034)))
        total += round((font.getbbox("Ag")[3] - font.getbbox("Ag")[1]) * 1.42) * len(lines)
    if shot.location:
        font = _load_font(max(12, round(height * 0.030)))
        total += round(height * 0.014) + round((font.getbbox("Ag")[3] - font.getbbox("Ag")[1]) * 1.42)
    return total


def day_card_text(shot: "Shot", geometry: dict, frame_size: tuple[int, int],
                  cfg: AssembleConfig) -> Image.Image:
    """The centred title block: eyebrow, title, then date and place."""
    width, height = frame_size
    overlay = Image.new("RGBA", frame_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    centre = width / 2
    y = geometry["text_top"]

    def centred(text: str, font, fill, shadow=(0, 0, 0, 170)) -> int:
        w = draw.textlength(text, font=font)
        line_h = round((font.getbbox("Ag")[3] - font.getbbox("Ag")[1]) * 1.42)
        draw.text((centre - w / 2 + 2, y + 2), text, font=font, fill=shadow)
        draw.text((centre - w / 2, y), text, font=font, fill=fill)
        return line_h

    if shot.eyebrow:
        eyebrow_font = _load_font(max(12, round(height * 0.026)))
        spaced = " ".join(shot.eyebrow.upper())      # letterspaced, like a chapter mark
        y += centred(spaced, eyebrow_font, (235, 92, 48, 250)) + round(height * 0.012)

    if shot.caption:
        title_font, title_lines = fit_text(shot.caption, geometry["text_width"], 3, draw,
                                           max(20, round(height * cfg.day_card_title)),
                                           max(16, round(height * 0.034)))
        for line in title_lines:
            y += centred(line, title_font, (255, 255, 255, 250))

    if shot.location:
        y += round(height * 0.014)
        sub_font, sub_lines = fit_text(shot.location, geometry["text_width"], 1, draw,
                                       max(12, round(height * 0.030)),
                                       max(10, round(height * 0.020)))
        shot = replace(shot, location=sub_lines[0] if sub_lines else shot.location)
        rule = round(width * 0.05)
        draw.line([centre - rule, y - round(height * 0.009), centre + rule, y - round(height * 0.009)],
                  fill=(255, 255, 255, 70), width=1)
        centred(shot.location, sub_font, (232, 236, 244, 226))
    return overlay


def _joined_segments(clip, segments: list[tuple[float, float]], duration: float):
    """One clip from the chosen stretches, in order."""
    from moviepy import concatenate_videoclips

    available = clip.duration or duration
    pieces = [clip.subclipped(max(0.0, start), min(end, available))
              for start, end in segments
              if min(end, available) - max(0.0, start) > 0.1]
    if not pieces:
        return clip.subclipped(0, min(duration, available))
    joined = pieces[0] if len(pieces) == 1 else concatenate_videoclips(pieces, method="chain")
    return joined.subclipped(0, min(duration, joined.duration or duration))


def _ease_out(t: float) -> float:
    return 1.0 - (1.0 - max(0.0, min(1.0, t))) ** 3


def _scaled_position(rect: tuple[int, int, int, int], factor: float) -> tuple[int, int]:
    """Keep a scaled card centred on the same point it occupies at rest."""
    x, y, w, h = rect
    return (round(x + w * (1 - factor) / 2), round(y + h * (1 - factor) / 2))


def _segment(below: Image.Image, card, above: Image.Image, rect, frame_size,
             duration: float, scale_from: float, scale_to: float, ease):
    """One animated window: the card scales over a static backdrop and UI."""
    from moviepy import CompositeVideoClip, ImageClip

    def factor(t: float) -> float:
        progress = ease(t / duration) if duration > 0 else 1.0
        return scale_from + (scale_to - scale_from) * progress

    moving = (card.with_duration(duration)
              .resized(lambda t: factor(t))
              .with_position(lambda t: _scaled_position(rect, factor(t))))
    return CompositeVideoClip([
        ImageClip(np.array(below.convert("RGB"))).with_duration(duration),
        moving,
        ImageClip(np.array(above), transparent=True).with_duration(duration),
    ], size=frame_size).with_duration(duration)


def build_day_card(shot: "Shot", frame_size: tuple[int, int], cfg: AssembleConfig,
                   animate: bool = True):
    """A day's title card: photo band, centred title, small map."""
    from moviepy import ImageClip, concatenate_videoclips

    limit = int(max(frame_size) * 1.2)
    images = [img for img in (representative_pil(e.filepath, e.is_video, max_side=limit)
                              for e in shot.gallery[:cfg.day_card_images]) if img is not None]
    if not images:
        raise ValueError("no usable image for the day card")

    probe = day_card_layout([i.size for i in images], frame_size, cfg)
    text_height = day_card_text_height(shot, probe["text_width"], frame_size, cfg)
    geometry = day_card_layout([i.size for i in images], frame_size, cfg, text_height)
    radius, border = geometry["radius"], geometry["border"]

    plate = backdrop(images[0], Layout(frame=frame_size, panels=geometry["panels"],
                                       card=geometry["row"], radius=radius, border=border,
                                       ui_left=0, ui_top=0, ui_bottom=frame_size[1],
                                       ui_right=frame_size[0], gap=0), cfg)

    # the photo band is one card, so the whole row eases in together
    row_x, row_y, row_w, row_h = geometry["row"]
    band = Image.new("RGBA", (row_w, row_h), (0, 0, 0, 0))
    for image, (x, y, w, h) in zip(images, geometry["panels"]):
        panel = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        panel.paste(image.resize((w, h), Image.LANCZOS).convert("RGBA"), (0, 0),
                    _rounded_mask((w, h), radius))
        ImageDraw.Draw(panel).rounded_rectangle([0, 0, w - 1, h - 1], radius=radius,
                                                outline=cfg.card_border, width=border)
        band.alpha_composite(panel, (x - row_x, y - row_y))

    above = day_card_text(shot, geometry, frame_size, cfg)
    if shot.map_still is not None:
        try:
            map_x, map_y, map_w, _ = geometry["map"]
            map_img = Image.open(shot.map_still).convert("RGBA").resize((map_w, map_w), Image.LANCZOS)
            above.alpha_composite(map_img, (map_x, map_y))
        except Exception as exc:  # noqa: BLE001
            log.warning("cannot place the day map: %s", exc)

    still = plate.copy()
    still.alpha_composite(band, (row_x, row_y))
    still = Image.alpha_composite(still, above)
    rest = ImageClip(np.array(still.convert("RGB")))

    windows = (min(cfg.anim_in, shot.duration * 0.3), min(cfg.anim_out, shot.duration * 0.25))
    if not animate or sum(windows) >= shot.duration * 0.8:
        return rest.with_duration(shot.duration)

    in_dur, out_dur = windows
    band_clip = ImageClip(np.array(band), transparent=True)
    rect = (row_x, row_y, row_w, row_h)
    return concatenate_videoclips([
        _segment(plate, band_clip, above, rect, frame_size, in_dur, cfg.anim_from, 1.0, _ease_out),
        rest.with_duration(shot.duration - in_dur - out_dur),
        _segment(plate, band_clip, above, rect, frame_size, out_dur, 1.0, cfg.anim_to, lambda t: t),
    ], method="chain").with_duration(shot.duration)


def build_closing_card(shot: "Shot", frame_size: tuple[int, int], cfg: AssembleConfig,
                       animate: bool = True):
    """The finale: the whole route big in the middle, the quote and the trip's
    numbers under it, and a photo tucked into each corner."""
    from moviepy import ImageClip, concatenate_videoclips

    width, height = frame_size
    margin = round(cfg.frame_margin * width)
    radius = max(6, round(cfg.card_radius * width))
    border = max(1, round(cfg.card_border_width * width))
    limit = int(max(frame_size) * 1.2)

    images = [img for img in (representative_pil(e.filepath, e.is_video, max_side=limit)
                              for e in shot.gallery[:cfg.closing_card_images]) if img is not None]
    backdrop_source = images[0] if images else representative_pil(
        shot.entry.filepath, shot.entry.is_video, max_side=limit)
    if backdrop_source is None:
        raise ValueError("no usable image for the closing card")

    map_size = round(min(height * cfg.closing_map_fraction, width * 0.42))
    map_rect = (round((width - map_size) / 2), 0, map_size, map_size)

    scratch = ImageDraw.Draw(Image.new("RGBA", frame_size))
    text_width = round(width * 0.66)
    quote_font, quote_lines = fit_text(shot.caption or "", text_width, 3, scratch,
                                       max(20, round(height * cfg.closing_quote_size)),
                                       max(16, round(height * 0.030)))
    quote_lh = round((quote_font.getbbox("Ag")[3] - quote_font.getbbox("Ag")[1]) * 1.45)
    stats_font, stats_lines = fit_text(shot.location or "", text_width, 1, scratch,
                                       max(16, round(height * 0.040)),
                                       max(12, round(height * 0.024)))
    stats_h = round((stats_font.getbbox("Ag")[3] - stats_font.getbbox("Ag")[1]) * 1.9) if stats_lines else 0

    text_h = quote_lh * len(quote_lines) + stats_h + round(height * 0.02)
    block = map_size + round(height * 0.045) + text_h
    top = max(margin, round((height - block) / 2))
    map_rect = (map_rect[0], top, map_size, map_size)

    layout = Layout(frame=frame_size, panels=[map_rect], card=map_rect, radius=radius,
                    border=border, ui_left=0, ui_top=0, ui_bottom=height,
                    ui_right=width, gap=0)
    plate = backdrop(backdrop_source, layout, cfg)

    # one photo per corner, small enough to frame the middle rather than crowd it
    corner_w = round(width * cfg.closing_corner_width)
    corner_h = round(height * 0.19)
    for index, image in enumerate(images[:4]):
        w, h = _fit(image.size, (corner_w, corner_h))
        x = margin if index % 2 == 0 else width - margin - w
        y = margin if index < 2 else height - margin - h
        card = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        card.paste(image.resize((w, h), Image.LANCZOS).convert("RGBA"), (0, 0),
                   _rounded_mask((w, h), max(4, round(radius * 0.7))))
        ImageDraw.Draw(card).rounded_rectangle([0, 0, w - 1, h - 1], radius=max(4, round(radius * 0.7)),
                                               outline=cfg.card_border, width=border)
        plate.alpha_composite(card, (x, y))

    above = Image.new("RGBA", frame_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(above)
    centre = width / 2
    y = top + map_size + round(height * 0.045)
    for line in quote_lines:
        w = draw.textlength(line, font=quote_font)
        draw.text((centre - w / 2 + 2, y + 2), line, font=quote_font, fill=(0, 0, 0, 175))
        draw.text((centre - w / 2, y), line, font=quote_font, fill=(255, 255, 255, 250))
        y += quote_lh
    if stats_lines:
        y += round(height * 0.012)
        line = stats_lines[0]
        w = draw.textlength(line, font=stats_font)
        rule = round(width * 0.055)
        draw.line([centre - rule, y - round(height * 0.010), centre + rule, y - round(height * 0.010)],
                  fill=(235, 92, 48, 220), width=max(2, round(height * 0.003)))
        draw.text((centre - w / 2 + 2, y + 2), line, font=stats_font, fill=(0, 0, 0, 175))
        draw.text((centre - w / 2, y), line, font=stats_font, fill=(245, 214, 200, 244))

    map_card = None
    if shot.map_still is not None:
        try:
            map_card = Image.open(shot.map_still).convert("RGBA").resize((map_size, map_size),
                                                                         Image.LANCZOS)
        except Exception as exc:  # noqa: BLE001
            log.warning("cannot read the closing map: %s", exc)

    still = plate.copy()
    if map_card is not None:
        still.alpha_composite(map_card, (map_rect[0], map_rect[1]))
    still = Image.alpha_composite(still, above)
    rest = ImageClip(np.array(still.convert("RGB")))

    windows = (min(cfg.anim_in, shot.duration * 0.3), min(cfg.anim_out, shot.duration * 0.25))
    if map_card is None or not animate or sum(windows) >= shot.duration * 0.8:
        return rest.with_duration(shot.duration)

    in_dur, out_dur = windows
    map_clip = ImageClip(np.array(map_card), transparent=True)
    return concatenate_videoclips([
        _segment(plate, map_clip, above, map_rect, frame_size, in_dur, cfg.anim_from, 1.0, _ease_out),
        rest.with_duration(shot.duration - in_dur - out_dur),
        _segment(plate, map_clip, above, map_rect, frame_size, out_dur, 1.0, cfg.anim_to, lambda t: t),
    ], method="chain").with_duration(shot.duration)


def build_clip(entry: TimelineEntry, duration: float, frame_size: tuple[int, int],
               cfg: AssembleConfig, overlays_for: Callable[[Layout], list],
               resources: list | None = None, overlay_clips_for: Callable[[Layout], list] | None = None,
               companion: TimelineEntry | None = None, animate: bool = True,
               entrance: float | None = None):
    """One finished, silent shot: backdrop, picture card(s), then the UI.

    The still middle of a photo shot is flattened into a single image, so
    MoviePy emits one static layer instead of recompositing the same stack for
    every one of ~100 identical frames. Only the entrance and exit windows are
    animated - the picture rises from behind and drifts past on the way out -
    which keeps the carousel motion cheap: roughly one second of compositing
    per shot rather than all of it.

    Any opened `VideoFileClip` is appended to `resources`: closing the returned
    composite does not close its children, so the caller has to hold on to the
    reader and close it once the render is done.
    """
    from moviepy import CompositeVideoClip, ImageClip, VideoFileClip, concatenate_videoclips

    limit = int(max(frame_size) * 1.5)   # ample to downscale from
    source = representative_pil(entry.filepath, entry.is_video, max_side=limit)
    if source is None:
        raise ValueError("unreadable media")
    second = representative_pil(companion.filepath, False, max_side=limit) if companion else None

    sizes = [source.size] + ([second.size] if second else [])
    layout = layout_for(sizes, frame_size, cfg, with_map=cfg.map_mode == "overlay")
    plate = backdrop(source, layout, cfg)
    overlays = [o for o in overlays_for(layout) if o is not None]
    overlay_clips = [c for c in (overlay_clips_for(layout) if overlay_clips_for else []) if c is not None]

    def framed(image: Image.Image, rect: tuple[int, int, int, int]) -> Image.Image:
        """A picture as its own rounded, bordered card."""
        x, y, w, h = rect
        card = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        card.paste(image.resize((w, h), Image.LANCZOS).convert("RGBA"), (0, 0),
                   _rounded_mask((w, h), layout.radius))
        ImageDraw.Draw(card).rounded_rectangle([0, 0, w - 1, h - 1], radius=layout.radius,
                                               outline=cfg.card_border, width=layout.border)
        return card

    if second is not None and len(layout.panels) > 1:
        x, y, w, h = layout.panels[1]
        plate.alpha_composite(framed(second, layout.panels[1]), (x, y))

    above = Image.new("RGBA", frame_size, (0, 0, 0, 0))
    for overlay in overlays:
        above = Image.alpha_composite(above, overlay)

    rect = layout.panels[0]
    card_x, card_y, card_w, card_h = rect
    mask = _rounded_mask((card_w, card_h), layout.radius)
    windows = (min(cfg.anim_in, duration * 0.3), min(cfg.anim_out, duration * 0.25))
    animate = animate and not overlay_clips and sum(windows) < duration * 0.8
    anim_from = cfg.anim_from if entrance is None else entrance

    if not entry.is_video:
        card_image = framed(source, rect)
        still = plate.copy()
        still.alpha_composite(card_image, (card_x, card_y))
        still = Image.alpha_composite(still, above)
        rest = ImageClip(np.array(still.convert("RGB")))
        if overlay_clips:
            # a flying map rides above the flattened still; the picture holds
            # steady so the two motions do not compete
            return CompositeVideoClip(
                [rest.with_duration(duration)]
                + [clip.with_duration(duration) for clip in overlay_clips],
                size=frame_size).with_duration(duration)
        if not animate:
            return rest.with_duration(duration)

        in_dur, out_dur = windows
        card_clip = ImageClip(np.array(card_image), transparent=True)
        return concatenate_videoclips([
            _segment(plate, card_clip, above, rect, frame_size, in_dur,
                     anim_from, 1.0, _ease_out),
            rest.with_duration(duration - in_dur - out_dur),
            _segment(plate, card_clip, above, rect, frame_size, out_dur,
                     1.0, cfg.anim_to, lambda t: t),
        ], method="chain").with_duration(duration)

    readable = playable_video(entry.filepath, cfg.remux_dir)
    if readable is None:
        raise ValueError("unreadable video")
    source = VideoFileClip(readable, audio=False)
    if resources is not None:
        resources.append(source)
    # play the highlight stretches, exactly as the audio path does, so picture
    # and sound stay together
    video = _joined_segments(source, video_segments(entry, cfg), duration)
    footage = video.resized((card_w, card_h)).with_duration(duration)
    rounded = ImageClip(np.asarray(mask, dtype=np.float32) / 255.0, is_mask=True).with_duration(duration)
    border = ImageClip(np.array(card_border_overlay(layout, cfg)), transparent=True)

    layers = [ImageClip(np.array(plate.convert("RGB"))).with_duration(duration),
              footage.with_mask(rounded).with_position((card_x, card_y)),
              border.with_duration(duration),
              ImageClip(np.array(above), transparent=True).with_duration(duration)]
    layers.extend(clip.with_duration(duration) for clip in overlay_clips)
    composite = CompositeVideoClip(layers, size=frame_size).with_duration(duration)
    if not animate:
        return composite

    # a video card gets the same motion, applied to its own composite window
    in_dur, out_dur = windows

    def window(sub_start: float, sub_dur: float, scale_from: float, scale_to: float, ease):
        piece = video.subclipped(sub_start, sub_start + sub_dur).resized((card_w, card_h))
        piece = piece.with_mask(ImageClip(np.asarray(mask, dtype=np.float32) / 255.0,
                                          is_mask=True).with_duration(sub_dur))
        return _segment(plate, piece, above, rect, frame_size, sub_dur, scale_from, scale_to, ease)

    try:
        return concatenate_videoclips([
            window(0.0, in_dur, anim_from, 1.0, _ease_out),
            composite.subclipped(in_dur, duration - out_dur),
            window(duration - out_dur, out_dur, 1.0, cfg.anim_to, lambda t: t),
        ], method="chain").with_duration(duration)
    except Exception as exc:  # noqa: BLE001 - short or awkward clip: skip the motion
        log.debug("no entrance animation for %s: %s", entry.name, exc)
        return composite


@dataclass
class Shot:
    """One thing on screen: a normal clip, a day title card, or the finale."""

    entry: TimelineEntry
    duration: float
    companion: TimelineEntry | None = None
    caption: str = ""
    location: str | None = None
    eyebrow: str | None = None          # "DAY 3" above the title
    plan_index: int | None = None       # into plan.ordered_clips, for the map
    map_still: Path | None = None       # a card's own map instead of a track
    map_track: object | None = None     # the flying map for this shot, if any
    slice_start: float = 0.0            # when a chunk boundary splits this shot
    slice_end: float | None = None
    transition: str = "crossfade"
    kind: str = "clip"                  # clip | day | closing
    gallery: list = field(default_factory=list)   # day card: 1-3 photos
    punch: bool = False                 # lands on a drop: hit it harder


def day_of(entry: TimelineEntry) -> str | None:
    return str(entry.timestamp)[:10] if entry.timestamp else None


def hero_for(entries: list[TimelineEntry]) -> TimelineEntry | None:
    """The shot that should represent a day (or the trip) on a title card.

    Landscape photos win: a title card wants the full picture area, and a
    portrait one would leave it half empty. Quality breaks the tie.
    """
    photos = [e for e in entries if not e.is_video]
    pool = photos or entries
    if not pool:
        return None
    landscape = [e for e in pool if not is_portrait(e)]
    return max(landscape or pool, key=lambda e: e.quality_score or 0.0)


def day_gallery(entries: list[TimelineEntry], limit: int = 3) -> list[TimelineEntry]:
    """One to three photos to stand for a day, spread across it.

    Taking the three best would often mean three frames of the same hour; one
    from each third of the day tells the story of the day instead.
    """
    photos = sorted((e for e in entries if not e.is_video),
                    key=lambda e: (e.timestamp or "", e.name))
    if not photos:
        photos = sorted(entries, key=lambda e: (e.timestamp or "", e.name))
    if len(photos) <= limit:
        return photos[:limit]

    chosen = []
    for slot in range(limit):
        lo = round(slot * len(photos) / limit)
        hi = max(lo + 1, round((slot + 1) * len(photos) / limit))
        window = [e for e in photos[lo:hi] if e not in chosen]
        if window:
            chosen.append(max(window, key=lambda e: e.quality_score or 0.0))
    return chosen or photos[:limit]


def format_day_label(date_key: str, index: int) -> tuple[str, str]:
    """("DAY 3", "Fri 27 Feb") - the eyebrow and the date under the title."""
    try:
        when = datetime.strptime(date_key, "%Y-%m-%d")
        return f"DAY {index}", when.strftime("%a %d %b")
    except ValueError:
        return f"DAY {index}", date_key


def trip_summary(timeline: Timeline, coords: list[tuple[float, float]] | None) -> str:
    """"6 days · 5 places · 214 km" for the closing card."""
    days = {day_of(e) for e in timeline.selected if day_of(e)}
    places = {e.place for e in timeline.selected if e.place}
    parts = []
    if days:
        parts.append(f"{len(days)} day{'s' if len(days) != 1 else ''}")
    if places:
        parts.append(f"{len(places)} place{'s' if len(places) != 1 else ''}")
    if coords and len(coords) > 1:
        from map_render import _haversine_km

        distance = sum(_haversine_km(a, b) for a, b in zip(coords, coords[1:]))
        if distance >= 1:
            parts.append(f"{distance:,.0f} km")
    return " · ".join(parts)


def is_portrait(entry: TimelineEntry, threshold: float = 0.88) -> bool:
    return bool(entry.width and entry.height and entry.width / entry.height < threshold)


def pair_portraits(entries: list, cfg: AssembleConfig, animated: set[int]) -> list:
    """Fold portrait photos into shared clips, two side by side.

    A single tall photo in a wide frame leaves most of the picture area empty;
    two fill it and read as a spread. Partners do not have to be adjacent -
    portraits are usually scattered between landscape shots - so each one looks
    a few clips ahead for the next portrait *in the same place*, which keeps
    the location label and the map honest while barely disturbing the order.

    Only photos pair: two video decoders and two soundtracks in one clip is not
    worth it. A clip whose map is flying keeps its own slot in the sequence.
    """
    if not cfg.pair_portraits:
        return [(i, path, entry, None) for i, path, entry in entries]

    def pairable(index: int, entry: TimelineEntry) -> bool:
        return not entry.is_video and is_portrait(entry) and index not in animated

    def same_place(a: TimelineEntry, b: TimelineEntry) -> bool:
        """Compatible enough to share one location label.

        Resolved names, not GPS clusters: DBSCAN splits a week in one valley
        into twenty clusters that are all the same town, and matching on
        cluster id blocks almost every pair. A clip with no fix at all adopts
        its partner's label - it had none of its own to contradict.
        """
        if a.place and b.place:
            return a.place == b.place
        if a.place or b.place:
            return True
        return a.group is None or b.group is None or a.group == b.group

    result, used = [], set()
    for position, (plan_index, path, entry) in enumerate(entries):
        if position in used:
            continue
        companion = None
        if pairable(plan_index, entry):
            for ahead in range(position + 1, min(position + 1 + cfg.pair_lookahead, len(entries))):
                if ahead in used:
                    continue
                next_index, _, next_entry = entries[ahead]
                if pairable(next_index, next_entry) and same_place(entry, next_entry):
                    companion = next_entry
                    used.add(ahead)
                    break
        result.append((plan_index, path, entry, companion))
    return result


def plan_shots(timeline: Timeline, plan: NarrativePlan, cfg: AssembleConfig,
               map_tracks: Sequence | None, work_dir: Path | None = None) -> list[Shot]:
    """Turn the ordered clips into the shot list, chapters and finale included.

    Day cards open each calendar day with its own title, hero image and that
    day's leg of the route; the closing card ends on the quote, a hero shot and
    the whole track. Both are ordinary shots as far as the renderer cares.
    """
    import map_render

    paths = plan.ordered_clips or [e.filepath for e in timeline.selected]
    resolved = [(i, path, timeline.by_path(path)) for i, path in enumerate(paths)]
    missing = [path for _, path, entry in resolved if entry is None]
    if missing:
        log.warning("%d planned clip(s) are not in the timeline and will be skipped", len(missing))
    resolved = [(i, path, entry) for i, path, entry in resolved if entry is not None]
    if not resolved:
        return []

    animated = {i for i, t in enumerate(map_tracks or []) if getattr(t, "animated", False)}
    paired = pair_portraits(resolved, cfg, animated)

    coords = map_render.route_for(timeline, [path for _, path, _ in resolved]) if map_tracks else []
    maps_dir = Path(work_dir or "./tmp") / "map_frames"
    # which points on the route belong to each day, for the day card's map
    day_indexes: dict[str, list[int]] = {}
    for position, (_, _, entry) in enumerate(resolved):
        key = day_of(entry)
        if key:
            day_indexes.setdefault(key, []).append(position)

    shots: list[Shot] = []
    seen_days: set[str] = set()

    for plan_index, path, entry, companion in paired:
        day = day_of(entry)
        if cfg.day_cards and day and day not in seen_days:
            seen_days.add(day)
            members = [e for _, _, e, _ in paired if day_of(e) == day]
            gallery = day_gallery(members, cfg.day_card_images)
            hero = hero_for(members) or entry
            eyebrow, date_label = format_day_label(day, len(seen_days))
            title = plan.day_titles.get(day) or date_label
            subtitle = date_label
            places = [e.place for e in members if e.place]
            if places:
                subtitle = f"{date_label} · {max(set(places), key=places.count)}"
            day_map = None
            if coords and day in day_indexes:
                day_map = map_render.render_track_map(
                    coords, day_indexes[day], maps_dir / f"day_{day}.png",
                    map_render.MapConfig(out_dir=maps_dir, cache_dir=Path(maps_dir).parent / "tile_cache",
                                         size=700))
            shots.append(Shot(entry=gallery[0] if gallery else hero,
                              duration=cfg.day_card_duration, caption=title,
                              location=subtitle, eyebrow=eyebrow, map_still=day_map,
                              transition="fadein", kind="day",
                              gallery=gallery or [hero]))

        shots.append(Shot(entry=entry, duration=clip_duration(entry, cfg) *
                          (cfg.pair_duration_scale if companion else 1.0),
                          companion=companion,
                          caption=plan.captions.get(path, "") or
                          (plan.captions.get(companion.filepath, "") if companion else ""),
                          location=location_label(entry) if cfg.show_location else None,
                          plan_index=plan_index,
                          transition=plan.transition_for(path, "crossfade")))

    if cfg.closing_card and (plan.closing_quote or coords):
        hero = hero_for([e for _, _, e, _ in paired]) or paired[-1][2]
        full_map = None
        if coords:
            full_map = map_render.render_track_map(
                coords, None, maps_dir / "trip_overview.png",
                map_render.MapConfig(out_dir=maps_dir, cache_dir=Path(maps_dir).parent / "tile_cache",
                                     size=700))
        shots.append(Shot(entry=hero, duration=cfg.closing_card_duration,
                          caption=plan.closing_quote or "That was the trip",
                          location=trip_summary(timeline, coords) or None,
                          map_still=full_map, transition="crossfade", kind="closing",
                          gallery=day_gallery([e for _, _, e, _ in paired],
                                              cfg.closing_card_images)))

    log.info("shot list: %d clips, %d day card(s)%s", len(paired), len(seen_days),
             ", closing card" if shots and shots[-1].kind == "closing" else "")
    return shots


def fit_to_duration(shots: list["Shot"], cfg: AssembleConfig) -> float:
    """Stretch or squeeze the photo shots so the film lands on `target_duration`.

    Videos keep their own length - trimming them further would cut mid-action -
    and so do the title cards, so the whole adjustment lands on the stills.
    Returns the photo duration actually used, clamped to something watchable:
    if the target cannot be reached within those bounds, the film comes out as
    close as it can and says so rather than flashing frames past.
    """
    if not cfg.target_duration or not shots:
        return cfg.photo_duration

    overlap = cfg.crossfade * max(0, sum(1 for s in shots[1:] if s.transition == "crossfade"))
    fixed = sum(shot.duration for shot in shots
                if shot.kind != "clip" or shot.entry.is_video)
    photo_shots = [shot for shot in shots if shot.kind == "clip" and not shot.entry.is_video]
    weight = sum(cfg.pair_duration_scale if shot.companion else 1.0 for shot in photo_shots)
    if weight <= 0:
        log.warning("no photo shots to stretch - keeping the natural length")
        return cfg.photo_duration

    wanted = (cfg.target_duration + overlap - fixed) / weight
    photo = max(cfg.min_photo_duration, min(cfg.max_photo_duration, wanted))
    if abs(photo - wanted) > 0.05:
        reachable = fixed + photo * weight - overlap
        log.warning("cannot reach %.0fs with %d photo shot(s): %.1fs each would be needed, "
                    "using %.1fs for about %.0fs",
                    cfg.target_duration, len(photo_shots), wanted, photo, reachable)
    else:
        log.info("photo duration %.2fs to land on %.0fs", photo, cfg.target_duration)

    for shot in photo_shots:
        shot.duration = photo * (cfg.pair_duration_scale if shot.companion else 1.0)
    return photo


def choose_transitions(shots: list["Shot"], music, cfg: AssembleConfig) -> int:
    """Pick cuts or dissolves from what the music is doing at each join.

    Runs *before* the length fit on purpose: a hard cut has no crossfade
    overlap, so switching a dissolve to a cut lengthens the film. Deciding
    first means `fit_to_duration` counts the overlaps that will actually exist.
    """
    if not (music and music.beats and cfg.beat_sync and shots):
        return 0

    busy = music.busy_threshold()
    changed, start = 0, 0.0
    for index, shot in enumerate(shots):
        nxt = shots[index + 1] if index + 1 < len(shots) else None
        overlap = cfg.crossfade if (nxt and nxt.transition == "crossfade") else 0.0
        appears = start + shot.duration - overlap
        if nxt is not None and nxt.kind == "clip":
            if music.is_drop(appears):
                nxt.transition, nxt.punch = "cut", True   # change picture on the drop
                changed += 1
            elif music.energy_at(appears) >= busy:
                nxt.transition = "cut"                    # busy: cut, do not dissolve
                changed += 1
            elif music.energy_at(appears) < busy * 0.8:
                nxt.transition = "crossfade"              # quiet: let it breathe
        start += shot.duration - overlap
    return changed


def sync_to_music(shots: list["Shot"], music, cfg: AssembleConfig) -> int:
    """Re-time the cut so it belongs to the track.

    Two passes, after the transitions are settled and the length is fitted:

    1. **Pacing** - shots in busy passages run shorter and shots in quiet ones
       longer, then everything is rescaled so the film keeps the length it was
       given. This is most of what makes a cut feel musical.
    2. **Snapping** - each cut point is nudged onto the nearest beat (a
       downbeat before a title card, which wants the bar) when it is within
       `beat_tolerance`. Videos are only ever shortened to reach one, never
       stretched past the footage that exists.

    Returns how many cuts ended up on a beat.
    """
    if not (music and music.beats and cfg.beat_sync and shots):
        return 0

    flexible = [s for s in shots if s.kind == "clip" and not s.entry.is_video]
    if flexible and cfg.beat_sync_strength > 0:
        import music as music_module

        before = sum(s.duration for s in flexible)
        at = 0.0
        for shot in shots:
            if shot in flexible:
                shot.duration *= music_module.pacing_factor(music, at, cfg.beat_sync_strength)
            at += shot.duration
        after = sum(s.duration for s in flexible)
        if after > 0:                      # keep the overall length we were asked for
            for shot in flexible:
                shot.duration = max(cfg.min_photo_duration,
                                    min(cfg.max_photo_duration, shot.duration * before / after))

    interval = music.beat_interval or 0.5
    tolerance = min(cfg.beat_tolerance, interval * 0.5)
    snapped, start = 0, 0.0

    for index, shot in enumerate(shots):
        nxt = shots[index + 1] if index + 1 < len(shots) else None
        overlap = cfg.crossfade if (nxt and nxt.transition == "crossfade") else 0.0
        cut_at = start + shot.duration - overlap      # when the next picture appears
        beat = music.nearest_beat(cut_at, tolerance,
                                  prefer_downbeats=bool(nxt and nxt.kind != "clip"))
        if beat is not None:
            wanted = beat + overlap - start
            if shot.entry.is_video and shot.kind == "clip":
                # only ever trim a video to the beat; stretching it would freeze
                wanted = min(wanted, shot.duration)
                low, high = 1.2, shot.duration
            elif shot.kind != "clip":
                low, high = shot.duration * 0.75, shot.duration * 1.3
            else:
                low, high = cfg.min_photo_duration, cfg.max_photo_duration
            adjusted = max(low, min(high, wanted))
            if abs(adjusted - shot.duration) <= tolerance + 1e-6:
                shot.duration = adjusted
                snapped += 1
        start += shot.duration - overlap

    log.info("beat sync: %d/%d cuts on the beat at %.0f BPM, %d drop(s) hit",
             snapped, len(shots), music.tempo, sum(1 for s in shots if s.punch))
    return snapped


def video_segments(entry: TimelineEntry, cfg: AssembleConfig) -> list[tuple[float, float]]:
    """Which stretches of a video to play, from Phase 3's highlight scan.

    Without a cached scan (`--skip-understand` on a fresh timeline, or the scan
    disabled) this falls back to the opening seconds, which is what the film
    did before highlights existed.
    """
    cached = entry.extra.get("highlights") if cfg.use_highlights else None
    segments = []
    for item in cached or []:
        try:
            start, end = float(item[0]), float(item[1])
        except (TypeError, ValueError, IndexError):
            continue
        if end - start > 0.2:
            segments.append((max(0.0, start), min(end, entry.duration or end)))
    if segments:
        budget, kept = cfg.max_video_duration, []
        for start, end in segments:          # respect the budget even if cached wider
            if budget <= 0.1:
                break
            take = min(end - start, budget)
            kept.append((start, start + take))
            budget -= take
        return kept
    natural = entry.duration or cfg.max_video_duration
    return [(0.0, max(0.5, min(natural, cfg.max_video_duration)))]


def clip_duration(entry: TimelineEntry, cfg: AssembleConfig) -> float:
    if not entry.is_video:
        return max(0.5, cfg.photo_duration)
    return max(0.5, sum(end - start for start, end in video_segments(entry, cfg)))


# ------------------------------------------------------------------ audio

def audio_levels(entry: TimelineEntry) -> dict[str, float] | None:
    """Cached loudness for a clip, measured on demand if Phase 3 was skipped."""
    mean = entry.extra.get("audio_mean_db")
    if mean is not None:
        return {"mean_db": float(mean), "max_db": float(entry.extra.get("audio_max_db", 0.0))}
    from mediautil import measure_audio

    levels = measure_audio(entry.filepath)
    if levels:
        entry.extra["audio_mean_db"] = round(levels["mean_db"], 2)
        entry.extra["audio_max_db"] = round(levels["max_db"], 2)
    return levels


def normalized_volume(levels: dict[str, float] | None, role_volume: float,
                      target_db: float, cfg: AssembleConfig) -> float:
    """Volume factor that brings a clip to `target_db`, then applies its role.

    The peak guard is applied to the *combined* figure, so a clip that is only
    quiet on average but has loud transients is lifted as far as its peaks
    allow and no further - no clipping, no limiter needed. Clips under the
    silence floor are left alone: boosting them 40 dB only amplifies hiss.
    """
    if not cfg.normalize_audio or not levels:
        return role_volume
    if levels["mean_db"] < cfg.silence_floor_db:
        return role_volume

    role_db = 20 * math.log10(max(role_volume, 1e-6))
    gain_db = max(min(target_db - levels["mean_db"], cfg.max_gain_db), -cfg.max_gain_db)
    total_db = min(gain_db + role_db, -cfg.headroom_db - levels["max_db"])
    return float(10 ** (total_db / 20))


def _gain_envelope(windows: list[tuple[float, float]], level: float, fade: float, total: float):
    """A time -> gain function: 1.0 normally, `level` inside each window."""
    def gain(t):
        times = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.ones_like(times)
        for start, end in windows:
            if end <= start:
                continue
            ramp_in = np.clip((times - (start - fade)) / max(fade, 1e-6), 0.0, 1.0)
            ramp_out = np.clip(((end + fade) - times) / max(fade, 1e-6), 0.0, 1.0)
            depth = np.minimum(ramp_in, ramp_out)
            out = np.minimum(out, 1.0 - (1.0 - level) * depth)
        return out
    return gain


AUDIO_SUFFIXES = {".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus", ".wma"}


def _natural_key(path: Path) -> tuple:
    """Sort "2 - song" before "10 - song".

    Plain lexicographic order puts track 10 before track 2, which quietly
    reorders a playlist. Splitting the digits out and comparing them as numbers
    gives the order the files were named in.
    """
    parts = re.split(r"(\d+)", path.stem.lower())
    return tuple(int(part) if part.isdigit() else part for part in parts)


def music_files(sources) -> list[str]:
    """Expand what was passed to --music into an ordered list of audio files.

    Accepts a single file, several files (played in the order given), or a
    directory (played in natural filename order), so a playlist is just
    `--music ./playlist/` or `--music intro.mp3 main.mp3`.
    """
    if isinstance(sources, (str, Path)):
        sources = [sources]
    files: list[str] = []
    for source in sources or []:
        path = Path(source)
        if path.is_dir():
            tracks = [f for f in path.iterdir()
                      if f.is_file() and f.suffix.lower() in AUDIO_SUFFIXES]
            files.extend(str(f) for f in sorted(tracks, key=_natural_key))
        elif path.is_file():
            files.append(str(path))
        else:
            log.warning("music file not found: %s", path)
    return files


def music_bed_length(paths: list[str], cfg: AssembleConfig) -> float:
    """How long the soundtrack runs once the songs are crossfaded together.

    Used when no `--duration` is given: the film then lasts exactly as long as
    the music, instead of the music being looped or cut to fit the pictures.
    """
    from mediautil import audio_duration

    lengths = [d for d in (audio_duration(p) for p in paths) if d and d > 0]
    if not lengths:
        return 0.0
    fade = min(cfg.music_crossfade, min(lengths) / 3) if len(lengths) > 1 else 0.0
    return sum(lengths) - fade * (len(lengths) - 1)


def build_music_bed(paths: list[str], total: float, cfg: AssembleConfig):
    """One continuous bed from one or more songs, matched in level and
    crossfaded into each other, then looped or trimmed to `total`.

    Each song is normalised on its own before the joins: a playlist mixes
    masters from different years, and without that step one track arrives twice
    as loud as the one before it.
    """
    from moviepy import AudioFileClip, CompositeAudioClip, afx

    from mediautil import measure_audio

    songs, opened = [], []
    for path in paths:
        try:
            song = AudioFileClip(path)
        except Exception as exc:  # noqa: BLE001 - unreadable track, skip it
            log.error("could not load music %s: %s", path, exc)
            continue
        opened.append(song)
        volume = cfg.music_volume
        if cfg.normalize_audio:
            volume = normalized_volume(measure_audio(path), cfg.music_volume,
                                       cfg.music_target_db, cfg)
            log.info("music %s: %+.1f dB", Path(path).name,
                     20 * math.log10(max(volume, 1e-6)))
        songs.append(song.with_effects([afx.MultiplyVolume(volume)]))
    if not songs:
        return None

    fade = min(cfg.music_crossfade, min(s.duration for s in songs) / 3)
    placed, cursor = [], 0.0
    for index, song in enumerate(songs):
        piece = song
        effects = []
        if index > 0 and fade > 0:
            effects.append(afx.AudioFadeIn(fade))
        if index < len(songs) - 1 and fade > 0:
            effects.append(afx.AudioFadeOut(fade))
        if effects:
            piece = piece.with_effects(effects)
        placed.append(piece.with_start(cursor))
        cursor += song.duration - (fade if index < len(songs) - 1 else 0.0)

    bed = CompositeAudioClip(placed).with_duration(cursor) if len(placed) > 1 else placed[0]
    if bed.duration < total:
        bed = bed.with_effects([afx.AudioLoop(duration=total)])
    return bed.subclipped(0, min(total, bed.duration)), opened


def _joined_audio(audio, segments: list[tuple[float, float]], duration: float):
    """The clip's own sound, following the same stretches as its picture."""
    from moviepy import afx, concatenate_audioclips

    available = audio.duration or duration
    pieces = []
    for start, end in segments:
        low, high = max(0.0, start), min(end, available)
        if high - low <= 0.1:
            continue
        piece = audio.subclipped(low, high)
        fade = min(0.06, (high - low) / 4)     # tiny fades so joins do not click
        pieces.append(piece.with_effects([afx.AudioFadeIn(fade), afx.AudioFadeOut(fade)]))
    if not pieces:
        return audio.subclipped(0, min(duration, available))
    joined = pieces[0] if len(pieces) == 1 else concatenate_audioclips(pieces)
    return joined.subclipped(0, min(duration, joined.duration or duration))


def build_audio(music_paths, total: float, duck_windows: list[tuple[float, float]],
                clip_audio: list, cfg: AssembleConfig):
    """Music bed (looped, normalised, ducked) mixed with the clips' own audio."""
    from moviepy import CompositeAudioClip, afx

    tracks = []
    paths = music_files(music_paths)
    if not paths:
        log.error("no usable music files in %s", music_paths)
    else:
        try:
            music, _opened = build_music_bed(paths, total, cfg)
            effects = [afx.AudioFadeIn(1.0), afx.AudioFadeOut(min(2.5, total / 4))]
            music = music.with_effects(effects)
            if duck_windows:
                gain = _gain_envelope(duck_windows, cfg.duck_volume, cfg.duck_fade, total)

                def duck(get_frame, t):
                    frame = np.asarray(get_frame(t), dtype=float)
                    factor = gain(t)
                    if frame.ndim == 2:            # (n_samples, channels)
                        return frame * factor[:, None]
                    return frame * float(factor[0])

                music = music.transform(duck, keep_duration=True)
            tracks.append(music)
        except Exception as exc:  # noqa: BLE001 - unreadable music must not be fatal
            log.error("could not build the music bed: %s", exc)

    tracks.extend(a for a in clip_audio if a is not None)
    if not tracks:
        return None
    return CompositeAudioClip(tracks).with_duration(total)


# --------------------------------------------------------------- assembly

def schedule_shots(shots: list["Shot"], cfg: AssembleConfig) -> list[tuple["Shot", float, float]]:
    """(shot, start, duration) for the whole film.

    One source of truth for timing: the chunk renderer lays out pictures from
    it and the soundtrack is placed against it, so the two cannot drift.
    """
    schedule, position = [], 0.0
    for index, shot in enumerate(shots):
        overlap = cfg.crossfade if (shot.transition == "crossfade" and index) else 0.0
        start = max(0.0, position - overlap)
        schedule.append((shot, start, shot.duration))
        position = start + shot.duration
    return schedule


def chunk_shots(shots: list["Shot"], cfg: AssembleConfig) -> list[list["Shot"]]:
    """Split the film into independently renderable runs.

    A boundary can fall anywhere, including in the middle of a crossfade: the
    shot that straddles it appears in both chunks, sliced. The outgoing chunk
    carries its first `crossfade` seconds - so the dissolve completes inside
    that file - and the incoming chunk carries the rest, starting on a cut.
    Both chunks build the shot identically and then slice it, so the join is
    frame-exact and no transition is lost. Splitting only at hard cuts would
    mean no parallelism at all on a film that dissolves throughout.
    """
    if cfg.chunk_size <= 0 or len(shots) <= cfg.chunk_size:
        return [list(shots)]

    chunks: list[list[Shot]] = []
    current: list[Shot] = []
    for index, shot in enumerate(shots):
        if current and len(current) >= cfg.chunk_size and index:
            if shot.transition == "crossfade":
                overlap = min(cfg.crossfade, shot.duration / 2)
                current.append(replace(shot, slice_end=overlap))
                chunks.append(current)
                current = [replace(shot, slice_start=overlap, transition="cut")]
                continue
            chunks.append(current)
            current = []
        current.append(shot)
    if current:
        chunks.append(current)
    return chunks


def _render_chunk(job: dict) -> str:
    """Render one run of shots to a silent file. Runs in a worker process."""
    import logging as worker_logging

    from moviepy import CompositeVideoClip, vfx

    worker_logging.basicConfig(level=worker_logging.ERROR)
    shots: list[Shot] = job["shots"]
    cfg: AssembleConfig = job["cfg"]
    frame_size = job["frame_size"]
    codec, preset, extra = job["encoder"]

    clips, resources, position = [], [], 0.0
    for index, shot in enumerate(shots):
        duration = shot.duration
        sliced = shot.slice_start > 0 or shot.slice_end is not None
        piece = (shot.slice_end if shot.slice_end is not None else duration) - shot.slice_start
        track = shot.map_track
        animated = bool(track and getattr(track, "animated", False))

        def overlays_for(layout, shot=shot, track=track, animated=animated):
            built = []
            if shot.map_still is not None:
                built.append(map_overlay(Path(shot.map_still), layout, cfg))
            elif track is not None and not animated and track.first is not None:
                built.append(map_overlay(Path(track.first), layout, cfg))
            if shot.caption or shot.location or shot.eyebrow:
                built.append(caption_overlay(shot.caption, layout, cfg, shot.location,
                                             shot.eyebrow, 1.12 if shot.kind != "clip" else 1.0))
            return built

        def clips_for(layout, track=track, animated=animated, duration=duration):
            return [map_animation_clip(track, layout, cfg, duration)] if animated else []

        try:
            if shot.kind == "day":
                clip = build_day_card(shot, frame_size, cfg, animate=cfg.animate_cards)
            elif shot.kind == "closing":
                clip = build_closing_card(shot, frame_size, cfg, animate=cfg.animate_cards)
            else:
                clip = build_clip(shot.entry, duration, frame_size, cfg, overlays_for,
                                  resources, clips_for, shot.companion,
                                  animate=cfg.animate_cards and not animated,
                                  entrance=cfg.drop_punch if shot.punch else None)
        except Exception as exc:  # noqa: BLE001 - skip the file, keep the film
            log.error("skipping %s during assembly: %s", shot.entry.name, exc)
            continue

        if sliced:
            clip = clip.subclipped(shot.slice_start, shot.slice_end)

        overlap = cfg.crossfade if (shot.transition == "crossfade" and clips) else 0.0
        start = max(0.0, position - overlap)
        if overlap > 0:
            clip = clip.with_effects([vfx.CrossFadeIn(overlap)])
        elif shot.transition == "fadein" or (job["first"] and not clips):
            clip = clip.with_effects([vfx.FadeIn(min(0.6, duration / 2))])
        clips.append(clip.with_start(start))
        position = start + piece

    if not clips:
        raise RuntimeError("every clip in this chunk failed to build")

    video = CompositeVideoClip(clips, size=frame_size).with_duration(position)
    if job["last"]:
        video = video.with_effects([vfx.FadeOut(min(1.2, position / 8))])

    try:
        with tempfile.TemporaryDirectory(prefix="tripvid_chunk_") as scratch:
            video.write_videofile(job["output"], fps=cfg.fps, codec=codec, preset=preset,
                                  audio=False, threads=cfg.threads or None,
                                  ffmpeg_params=extra, temp_audiofile_path=scratch,
                                  logger=None)
    finally:
        for item in clips + resources:
            try:
                item.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            video.close()
        except Exception:  # noqa: BLE001
            pass
        from mediautil import release_memory

        release_memory()
    return job["output"]


def concat_chunks(paths: list[str], output: str) -> bool:
    """Join the rendered runs without re-encoding them."""
    listing = Path(output).with_suffix(".txt")
    listing.write_text("".join(f"file '{Path(p).resolve()}'\n" for p in paths), encoding="utf-8")
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
           "-i", str(listing), "-c", "copy", "-movflags", "+faststart", output]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, check=False)
    except (subprocess.SubprocessError, OSError) as exc:
        log.error("could not join the rendered chunks: %s", exc)
        return False
    finally:
        listing.unlink(missing_ok=True)
    if result.returncode != 0:
        log.error("joining the chunks failed: %s", (result.stderr or "")[:300])
        return False
    return True


def mux_audio(video_path: str, audio_path: str, output: str) -> bool:
    """Put the soundtrack onto the finished picture, copying both streams."""
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", video_path, "-i", audio_path,
           "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", "-shortest",
           "-movflags", "+faststart", output]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, check=False)
    except (subprocess.SubprocessError, OSError) as exc:
        log.error("could not attach the soundtrack: %s", exc)
        return False
    if result.returncode != 0:
        log.error("attaching the soundtrack failed: %s", (result.stderr or "")[:300])
        return False
    return True


def encoder_settings(cfg: AssembleConfig, parallel: int = 1) -> tuple[str, str | None, list[str]]:
    """(codec, preset, extra ffmpeg args) for the chosen encoder.

    NVENC moves encoding onto the GPU, which sits idle during assembly, and
    frees the CPU for the compositing that is the real bottleneck. Consumer
    cards allow a handful of concurrent sessions, so it is only used when few
    workers are rendering at once.
    """
    wanted = cfg.encoder
    if wanted == "auto":
        wanted = "nvenc" if (nvenc_available() and parallel <= 4) else "x264"
    if wanted == "nvenc" and nvenc_available():
        return "h264_nvenc", None, ["-preset", "p4", "-rc", "vbr", "-cq", "23",
                                    "-b:v", "0", "-movflags", "+faststart"]
    if wanted == "nvenc":
        log.warning("h264_nvenc is not available in this ffmpeg - using x264")
    return cfg.codec, cfg.preset, list(cfg.ffmpeg_params)


def assemble(timeline: Timeline, plan: NarrativePlan, music_path, output_path: str,
             map_tracks: Sequence | None, cfg: AssembleConfig,
             progress_cb: Callable | None = None, work_dir: Path | None = None,
             music=None) -> str:
    """Render the whole video and write it to `output_path`."""
    from moviepy import AudioFileClip, afx

    if work_dir is not None:
        cfg = replace(cfg,
                      remux_dir=cfg.remux_dir or Path(work_dir) / "remux",
                      audio_dir=cfg.audio_dir or Path(work_dir) / "clip_audio")
    frame_size = cfg.resolution or target_resolution(timeline)
    log.info("assembling at %dx%d", *frame_size)

    # plan.chapter_markers is kept in narrative.json for reference but is no
    # longer drawn: the day cards and the location line carry that job.
    use_map = cfg.map_mode != "none" and bool(map_tracks)

    shots = plan_shots(timeline, plan, cfg, map_tracks if use_map else None, work_dir)
    if not shots:
        raise RuntimeError("nothing to assemble - no selected clips")
    pairs = sum(1 for shot in shots if shot.companion is not None)
    if pairs:
        log.info("pairing %d portrait photo(s) side by side", pairs * 2)

    # transitions first (they decide the overlaps), then the length, then the
    # fine timing - each step depends on the one before it
    choose_transitions(shots, music, cfg)
    fit_to_duration(shots, cfg)
    sync_to_music(shots, music, cfg)

    # attach each shot's map, so a worker process needs nothing else
    for shot in shots:
        if use_map and shot.map_still is None and shot.plan_index is not None \
                and shot.plan_index < len(map_tracks):
            shot.map_track = map_tracks[shot.plan_index]

    schedule = schedule_shots(shots, cfg)
    total = schedule[-1][1] + schedule[-1][2]

    # --- the soundtrack, built once against the global timeline -----------
    clip_audio, duck_windows = [], []
    for shot, start, duration in schedule:
        if not (shot.entry.is_video and shot.kind == "clip"):
            continue
        try:
            wav = extracted_audio(shot.entry.filepath, cfg.audio_dir)
            if not wav:
                continue
            track = AudioFileClip(wav)
            audio = _joined_audio(track, video_segments(shot.entry, cfg), duration)
            speaking = bool(shot.entry.transcript)
            volume = normalized_volume(audio_levels(shot.entry),
                                       1.0 if speaking else cfg.ambient_volume,
                                       cfg.clip_target_db, cfg)
            clip_audio.append(audio.with_effects([
                afx.MultiplyVolume(volume),
                afx.AudioFadeIn(min(0.25, duration / 4)),
                afx.AudioFadeOut(min(0.25, duration / 4)),
            ]).with_start(start))
            if speaking:
                duck_windows.append((start, start + duration))
        except Exception as exc:  # noqa: BLE001 - a silent clip is not fatal
            log.warning("no usable audio from %s: %s", shot.entry.name, exc)

    # --- the picture, in independently rendered runs ----------------------
    chunks = chunk_shots(shots, cfg)
    scratch = Path(work_dir or Path(output_path).parent) / "chunks"
    scratch.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(cfg.workers, len(chunks)))
    settings = encoder_settings(cfg, workers)
    log.info("rendering %.0fs as %d chunk(s) on %d worker(s), %s",
             total, len(chunks), workers, settings[0])

    jobs = [{"shots": chunk, "cfg": cfg, "frame_size": frame_size,
             "output": str(scratch / f"chunk_{i:03d}.mp4"), "encoder": settings,
             "first": i == 0, "last": i == len(chunks) - 1}
            for i, chunk in enumerate(chunks)]

    done = 0
    if workers > 1 and len(jobs) > 1:
        import concurrent.futures as futures
        import multiprocessing

        context = multiprocessing.get_context("spawn")
        with futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            running = {pool.submit(_render_chunk, job): job for job in jobs}
            for future in futures.as_completed(running):
                job = running[future]
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001 - retry here, in-process
                    log.warning("chunk %s failed in a worker (%s) - retrying locally",
                                Path(job["output"]).stem, exc)
                    _render_chunk(job)
                done += 1
                if progress_cb:
                    progress_cb(done, len(jobs))
    else:
        for job in jobs:
            _render_chunk(job)
            done += 1
            if progress_cb:
                progress_cb(done, len(jobs))

    rendered = [job["output"] for job in jobs if Path(job["output"]).exists()]
    if not rendered:
        raise RuntimeError("no chunk rendered - nothing to write")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    silent = str(scratch / "picture.mp4")
    if len(rendered) == 1:
        shutil.copyfile(rendered[0], silent)
    elif not concat_chunks(rendered, silent):
        raise RuntimeError("could not join the rendered chunks")

    # --- soundtrack onto picture, both stream-copied ----------------------
    audio = build_audio(music_path, total, duck_windows, clip_audio, cfg)
    written = False
    if audio is not None:
        audio_path = str(scratch / "soundtrack.m4a")
        try:
            audio.write_audiofile(audio_path, codec=cfg.audio_codec, fps=44100, logger=None)
            written = mux_audio(silent, audio_path, output_path)
        except Exception as exc:  # noqa: BLE001 - ship the picture rather than nothing
            log.error("could not write the soundtrack (%s) - keeping the silent cut", exc)
        finally:
            try:
                audio.close()
            except Exception:  # noqa: BLE001
                pass
            for track in clip_audio:
                try:
                    track.close()
                except Exception:  # noqa: BLE001
                    pass
    if not written:
        shutil.move(silent, output_path)

    for leftover in scratch.glob("chunk_*.mp4"):
        leftover.unlink(missing_ok=True)
    Path(silent).unlink(missing_ok=True)
    from mediautil import release_memory

    release_memory()
    log.info("wrote %s (%.1fs, %d ducked window(s))", output_path, total, len(duck_windows))
    return output_path
