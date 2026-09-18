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
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mediautil import load_pil
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
    max_video_duration: float = 12.0
    crossfade: float = 0.5
    map_mode: str = "overlay"          # overlay | side | none
    map_opacity: float = 0.35
    map_width_fraction: float = 0.28   # of frame width, overlay mode
    side_panel_fraction: float = 0.30  # of frame width, side mode
    resolution: tuple[int, int] | None = None
    fps: int = 30
    codec: str = "libx264"
    audio_codec: str = "aac"
    bitrate: str | None = None
    preset: str = "medium"
    threads: int = 0
    music_volume: float = 1.0
    duck_volume: float = 0.15
    ambient_volume: float = 0.40
    duck_fade: float = 0.4
    show_chapters: bool = True
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


def _fit(size: tuple[int, int], box: tuple[int, int]) -> tuple[int, int]:
    """Largest size fitting inside `box`, preserving aspect (even numbers)."""
    scale = min(box[0] / size[0], box[1] / size[1])
    w, h = max(2, int(size[0] * scale)), max(2, int(size[1] * scale))
    return (w - w % 2, h - h % 2)


# --------------------------------------------------------------- captions

def _wrap(text: str, font: ImageFont.FreeTypeFont, max_width: int, draw: ImageDraw.ImageDraw) -> list[str]:
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
    return lines[:2]


def caption_overlay(text: str, frame_size: tuple[int, int], header: str | None = None,
                    text_band: tuple[int, int] | None = None) -> Image.Image | None:
    """Transparent full-frame overlay: caption bottom-left, header top-left.

    `text_band` is the horizontal span the text may use, so neither line runs
    under the map - which sits bottom-right in overlay mode and fills the right
    panel in side mode.
    """
    if not text and not header:
        return None
    width, height = frame_size
    overlay = Image.new("RGBA", frame_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    x0, x1 = text_band or (0, width)
    band = max(160, x1 - x0)
    left = x0 + int(width * 0.045)

    if text:
        font = _load_font(max(16, int(height * 0.042)))   # relative to frame height
        pad_x, pad_y = int(height * 0.024), int(height * 0.018)
        lines = _wrap(text, font, band - int(width * 0.09) - pad_x * 2, draw)
        line_h = int((font.getbbox("Ag")[3] - font.getbbox("Ag")[1]) * 1.55)
        text_w = max(int(draw.textlength(line, font=font)) for line in lines)
        bar_w = min(text_w + pad_x * 2, band - int(width * 0.05))
        bar_h = line_h * len(lines) + pad_y * 2
        bar_top = height - bar_h - int(height * 0.075)

        bar = Image.new("RGBA", (bar_w, bar_h), (0, 0, 0, 0))
        ImageDraw.Draw(bar).rounded_rectangle([0, 0, bar_w - 1, bar_h - 1],
                                              radius=int(bar_h * 0.24), fill=(0, 0, 0, 150))
        overlay.alpha_composite(bar, (left, bar_top))

        y = bar_top + pad_y
        for line in lines:
            draw.text((left + pad_x, y), line, font=font, fill=(255, 255, 255, 242))
            y += line_h

    if header:
        hfont = _load_font(max(18, int(height * 0.052)))
        y = int(height * 0.065)
        draw.text((left + 2, y + 2), header, font=hfont, fill=(0, 0, 0, 150))
        draw.text((left, y), header, font=hfont, fill=(255, 255, 255, 238))
        box = draw.textbbox((left, y), header, font=hfont)
        draw.line([left, box[3] + 9, box[2], box[3] + 9],
                  fill=(235, 92, 48, 235), width=max(2, int(height * 0.004)))
    return overlay


# -------------------------------------------------------------- map frame

def map_geometry(frame_size: tuple[int, int], cfg: AssembleConfig) -> tuple[int, tuple[int, int]]:
    """(display size, top-left position) for the map in the current mode.

    Overlay mode puts it bottom-**right**: the caption now owns the bottom-left
    corner. Opacity is already baked into the rendered PNG by Phase 5, which
    varies it while the view is travelling.
    """
    width, height = frame_size
    if cfg.map_mode == "side":
        panel = int(width * cfg.side_panel_fraction)
        size = min(panel - int(width * 0.03), height - int(height * 0.06))
        return size, (width - panel + (panel - size) // 2, (height - size) // 2)
    size = int(width * cfg.map_width_fraction)
    margin = int(width * 0.022)
    return size, (width - size - margin, height - size - margin)


def map_overlay(map_path: Path, frame_size: tuple[int, int], cfg: AssembleConfig) -> Image.Image | None:
    """A still map frame placed on a transparent full-frame layer."""
    try:
        map_img = Image.open(map_path).convert("RGBA")
    except Exception as exc:  # noqa: BLE001
        log.warning("cannot read map frame %s: %s", map_path, exc)
        return None
    size, position = map_geometry(frame_size, cfg)
    overlay = Image.new("RGBA", frame_size, (0, 0, 0, 0))
    overlay.alpha_composite(map_img.resize((size, size), Image.LANCZOS), position)
    return overlay


def map_animation_clip(track, frame_size: tuple[int, int], cfg: AssembleConfig, duration: float):
    """A moving map as a small positioned clip, holding its last frame.

    Only clips where the trip changes location get one; everywhere else the map
    is a still that gets flattened into the photo, which is far cheaper.
    """
    from moviepy import VideoClip

    size, position = map_geometry(frame_size, cfg)
    rgb, alpha = [], []
    for path in track.frames:
        try:
            frame = Image.open(path).convert("RGBA").resize((size, size), Image.LANCZOS)
        except Exception as exc:  # noqa: BLE001
            log.warning("cannot read map frame %s: %s", path, exc)
            continue
        rgb.append(np.array(frame.convert("RGB")))
        alpha.append(np.array(frame.getchannel("A"), dtype=np.float32) / 255.0)
    if not rgb:
        return None

    fps, last = max(track.fps, 1.0), len(rgb) - 1

    def index(t: float) -> int:
        return min(int(t * fps), last)

    clip = VideoClip(lambda t: rgb[index(t)], duration=duration)
    clip = clip.with_mask(VideoClip(lambda t: alpha[index(t)], is_mask=True, duration=duration))
    return clip.with_position(position)


# ------------------------------------------------------------ clip builder

def caption_band(frame_size: tuple[int, int], cfg: AssembleConfig, has_map: bool) -> tuple[int, int]:
    """Horizontal span left over for captions once the map has its corner."""
    width, _ = frame_size
    if not has_map or cfg.map_mode == "none":
        return (0, width)
    if cfg.map_mode == "side":
        return (0, int(width * (1 - cfg.side_panel_fraction)))
    size, position = map_geometry(frame_size, cfg)
    return (0, position[0] - int(width * 0.02))


def _video_area(frame_size: tuple[int, int], cfg: AssembleConfig) -> tuple[int, int, int, int]:
    """(x, y, w, h) of the region the footage may occupy."""
    width, height = frame_size
    if cfg.map_mode == "side":
        usable = int(width * (1 - cfg.side_panel_fraction))
        return (0, 0, usable, height)
    return (0, 0, width, height)


def build_clip(entry: TimelineEntry, duration: float, frame_size: tuple[int, int],
               overlays: list[Image.Image], cfg: AssembleConfig,
               resources: list | None = None, overlay_clips: list | None = None):
    """One finished, silent clip: footage + map + caption, fitted to the frame.

    A photo is flattened into a single still in PIL - background, image and
    overlays - so MoviePy has one static layer to emit instead of recompositing
    the same stack for every one of ~100 identical frames. Videos genuinely
    change per frame, so those stay a real composite.

    Any opened `VideoFileClip` is appended to `resources`: closing the returned
    composite does not close its children, so the caller has to hold on to the
    reader and close it once the render is done.
    """
    from moviepy import ColorClip, CompositeVideoClip, ImageClip, VideoFileClip

    area_x, area_y, area_w, area_h = _video_area(frame_size, cfg)

    overlay_clips = [c for c in (overlay_clips or []) if c is not None]

    if not entry.is_video and not overlay_clips:
        image = load_pil(entry.filepath)
        if image is None:
            raise ValueError("unreadable photo")
        fitted = image.resize(_fit(image.size, (area_w, area_h)), Image.LANCZOS)
        canvas = Image.new("RGB", frame_size, (0, 0, 0))
        canvas.paste(fitted, (area_x + (area_w - fitted.width) // 2,
                              area_y + (area_h - fitted.height) // 2))
        canvas = canvas.convert("RGBA")
        for overlay in overlays:
            if overlay is not None:
                canvas.alpha_composite(overlay)
        return ImageClip(np.array(canvas.convert("RGB"))).with_duration(duration)

    if entry.is_video:
        source = VideoFileClip(entry.filepath, audio=False)
        if resources is not None:
            resources.append(source)
        if source.duration and duration < source.duration:
            source = source.subclipped(0, duration)
        footage = source.resized(_fit(source.size, (area_w, area_h))).with_duration(duration)
    else:  # a photo that needs live layers on top, so it stays a composite
        image = load_pil(entry.filepath)
        if image is None:
            raise ValueError("unreadable photo")
        image = image.resize(_fit(image.size, (area_w, area_h)), Image.LANCZOS)
        footage = ImageClip(np.array(image)).with_duration(duration)

    footage = footage.with_position((
        area_x + (area_w - footage.size[0]) // 2,
        area_y + (area_h - footage.size[1]) // 2,
    ))

    layers = [ColorClip(frame_size, color=(0, 0, 0)).with_duration(duration), footage]
    for overlay in overlays:
        if overlay is not None:
            layers.append(ImageClip(np.array(overlay), transparent=True).with_duration(duration))
    layers.extend(clip.with_duration(duration) for clip in overlay_clips)
    return CompositeVideoClip(layers, size=frame_size).with_duration(duration)


def clip_duration(entry: TimelineEntry, cfg: AssembleConfig) -> float:
    if not entry.is_video:
        return max(0.5, cfg.photo_duration)
    duration = entry.duration or cfg.max_video_duration
    return max(0.5, min(duration, cfg.max_video_duration))


# ------------------------------------------------------------------ audio

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


def build_audio(music_path: str, total: float, duck_windows: list[tuple[float, float]],
                clip_audio: list, cfg: AssembleConfig):
    """Music bed (looped, ducked) mixed with the video clips' own audio."""
    from moviepy import AudioFileClip, CompositeAudioClip, afx

    tracks = []
    try:
        music = AudioFileClip(music_path)
        if music.duration < total:
            music = music.with_effects([afx.AudioLoop(duration=total)])
        music = music.subclipped(0, min(total, music.duration))
        effects = [afx.MultiplyVolume(cfg.music_volume), afx.AudioFadeIn(1.0),
                   afx.AudioFadeOut(min(2.5, total / 4))]
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
        log.error("could not load music %s: %s", music_path, exc)

    tracks.extend(a for a in clip_audio if a is not None)
    if not tracks:
        return None
    return CompositeAudioClip(tracks).with_duration(total)


# --------------------------------------------------------------- assembly

def assemble(timeline: Timeline, plan: NarrativePlan, music_path: str, output_path: str,
             map_tracks: Sequence | None, cfg: AssembleConfig,
             progress_cb: Callable | None = None,
             place_headers: dict[int, str] | None = None) -> str:
    """Render the whole video and write it to `output_path`."""
    from moviepy import CompositeVideoClip, VideoFileClip, afx, vfx

    paths = plan.ordered_clips or [e.filepath for e in timeline.selected]
    # Keep each clip's index in the *plan*: map frames and chapter markers are
    # numbered against that, so dropping an unresolvable path must not shift them.
    entries = [(i, p, timeline.by_path(p)) for i, p in enumerate(paths)]
    missing = [p for _, p, e in entries if e is None]
    if missing:
        log.warning("%d planned clip(s) are not in the timeline and will be skipped", len(missing))
    entries = [(i, p, e) for i, p, e in entries if e is not None]
    if not entries:
        raise RuntimeError("nothing to assemble - no selected clips")

    frame_size = cfg.resolution or target_resolution(timeline, [e for _, _, e in entries])
    log.info("assembling %d clips at %dx%d", len(entries), *frame_size)

    chapters = {c.at_clip: c.title for c in plan.chapter_markers} if cfg.show_chapters else {}
    place_headers = place_headers or {}
    use_map = cfg.map_mode != "none" and bool(map_tracks)
    shown_place: str | None = None

    visual_clips, clip_audio, duck_windows = [], [], []
    sources_to_close = []
    timeline_position = 0.0

    for position, (plan_index, path, entry) in enumerate(entries):
        duration = clip_duration(entry, cfg)
        overlays, overlay_clips = [], []
        if use_map and plan_index < len(map_tracks):
            track = map_tracks[plan_index]
            if getattr(track, "animated", False):
                overlay_clips.append(map_animation_clip(track, frame_size, cfg, duration))
            elif track.first is not None:
                overlays.append(map_overlay(Path(track.first), frame_size, cfg))

        caption = plan.captions.get(path, "")
        # The header names the place, and only when the trip reaches a new one.
        # Keyed on the resolved *name*, not the GPS cluster: a week in the
        # mountains can produce twenty clusters that are all "Kitzbuehel", and
        # re-titling every few clips would be noise.
        header = None
        if cfg.show_chapters:
            here = place_headers.get(entry.group) if entry.group is not None else None
            if here and here != shown_place:
                header = here
                shown_place = here
            elif not here and plan_index in chapters:
                header = chapters[plan_index]
        if caption or header:
            overlays.append(caption_overlay(caption, frame_size, header,
                                            caption_band(frame_size, cfg, use_map)))

        try:
            clip = build_clip(entry, duration, frame_size, overlays, cfg, sources_to_close,
                              overlay_clips)
        except Exception as exc:  # noqa: BLE001 - skip the file, keep the film
            log.error("skipping %s during assembly: %s", entry.name, exc)
            continue

        transition = plan.transition_for(path, "crossfade")
        overlap = cfg.crossfade if (transition == "crossfade" and visual_clips) else 0.0
        start = max(0.0, timeline_position - overlap)
        if overlap > 0:
            clip = clip.with_effects([vfx.CrossFadeIn(overlap)])
        elif transition == "fadein" or not visual_clips:
            clip = clip.with_effects([vfx.FadeIn(min(0.6, duration / 2))])
        clip = clip.with_start(start)
        visual_clips.append(clip)

        if entry.is_video:
            try:
                source = VideoFileClip(entry.filepath)
                sources_to_close.append(source)
                if source.audio is not None:
                    audio = source.audio.subclipped(0, min(duration, source.audio.duration))
                    speaking = bool(entry.transcript)
                    volume = 1.0 if speaking else cfg.ambient_volume
                    audio = audio.with_effects([
                        afx.MultiplyVolume(volume),
                        afx.AudioFadeIn(min(0.25, duration / 4)),
                        afx.AudioFadeOut(min(0.25, duration / 4)),
                    ]).with_start(start)
                    clip_audio.append(audio)
                    if speaking:
                        duck_windows.append((start, start + duration))
            except Exception as exc:  # noqa: BLE001
                log.warning("no usable audio from %s: %s", entry.name, exc)

        timeline_position = start + duration
        if progress_cb:
            progress_cb(position + 1, len(entries))

    if not visual_clips:
        raise RuntimeError("every clip failed to build - nothing to write")

    total = timeline_position
    video = CompositeVideoClip(visual_clips, size=frame_size).with_duration(total)
    video = video.with_effects([vfx.FadeOut(min(1.2, total / 8))])

    audio = build_audio(music_path, total, duck_windows, clip_audio, cfg)
    if audio is not None:
        video = video.with_audio(audio)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    log.info("writing %s (%.1fs, %d ducked window(s))", output_path, total, len(duck_windows))
    try:
        # MoviePy drops its intermediate audio file in the working directory
        # unless told otherwise; keep the project folder clean.
        with tempfile.TemporaryDirectory(prefix="tripvid_render_") as scratch:
            video.write_videofile(
                output_path,
                fps=cfg.fps,
                codec=cfg.codec,
                audio_codec=cfg.audio_codec,
                bitrate=cfg.bitrate,
                preset=cfg.preset,
                threads=cfg.threads or None,
                ffmpeg_params=cfg.ffmpeg_params,
                temp_audiofile_path=scratch,
                logger="bar",
            )
    finally:
        for clip in visual_clips:
            try:
                clip.close()
            except Exception:  # noqa: BLE001
                pass
        for source in sources_to_close:
            try:
                source.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            video.close()
        except Exception:  # noqa: BLE001
            pass
    return output_path
