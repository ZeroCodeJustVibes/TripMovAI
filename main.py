#!/usr/bin/env python3
"""Trip review video generator - one command, folder in, film out.

    python main.py --input ./immich_export/ --music ./song.mp3 --output trip.mp4

Phases run in order and all share `timeline.json`, so each is independently
re-runnable: what an earlier phase computed is reused rather than recomputed.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import re
import os
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
                           TextColumn, TimeElapsedColumn)
from rich.table import Table

import mediautil as _mediautil

_mediautil.use_system_ffmpeg()   # before MoviePy resolves its binary

import assemble as assemble_mod
import map_render
import mediautil
import music as music_mod
import narrative as narrative_mod
import selection
import understand as understand_mod
from ingest import exiftool_available, ingest, summarise
from models import NarrativePlan, Timeline

console = Console()
log = logging.getLogger("trip_video")


# ------------------------------------------------------------------- CLI

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Turn a folder of trip photos and videos into a polished recap video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="folder of exported photos/videos (searched recursively)")
    p.add_argument("--music", required=True, nargs="+", metavar="PATH",
                   help="background music: one file, several files played in order, "
                        "or a folder of them")
    p.add_argument("--output", required=True, help="output .mp4 path")

    p.add_argument("--map-mode", choices=["overlay", "side", "none"], default="overlay",
                   help="how the route map is composited")
    p.add_argument("--max-clips", type=int, default=50, help="target number of clips in the cut")
    p.add_argument("--photo-duration", type=float, default=3.5, help="seconds each photo is shown")
    p.add_argument("--duration", default=None, metavar="LENGTH",
                   help="target length of the finished video, e.g. 3m30s, 210 or 3.5m. "
                        "Without it the film runs exactly as long as the music does")
    p.add_argument("--max-video-duration", type=float, default=15.0,
                   help="seconds used from any one video; longer clips are cut down to "
                        "their most interesting moments")
    p.add_argument("--video-scenes", type=int, default=3,
                   help="how many separate moments a long video may contribute")
    p.add_argument("--no-highlights", action="store_true",
                   help="use the opening seconds of a long video instead of its best bits")
    p.add_argument("--blur-threshold", type=float, default=80.0, help="Laplacian variance below which a shot is dropped")
    p.add_argument("--prefer-people", action="store_true",
                   help="favour shots with people in them when choosing clips")
    p.add_argument("--people-weight", type=float, default=0.35,
                   help="how strongly --prefer-people outweighs picture quality, 0-1")
    p.add_argument("--all-videos", action="store_true",
                   help="put every video in the folder into the cut, skipping the quality "
                        "gate and deduplication for them")
    p.add_argument("--llm-model", default=narrative_mod.DEFAULT_MODEL, help="OpenRouter model for narrative planning")
    p.add_argument("--amount-labels", type=str, default="25%",
                   help="share of clips that get a funny caption, e.g. 25%% or 0.25")
    p.add_argument("--interactive", action="store_true", help="review and edit captions before rendering")
    p.add_argument("--skip-understand", action="store_true", help="reuse cached captions/transcripts, run no models")
    p.add_argument("--skip-narrative", action="store_true", help="skip the LLM, keep chronological order")
    p.add_argument("--refresh-narrative", action="store_true",
                   help="ask the LLM for new captions even if the cut is unchanged")
    p.add_argument("--no-map", action="store_true", help="disable the map overlay entirely")

    extra = p.add_argument_group("extras")
    extra.add_argument("--skip-select", action="store_true", help="reuse the cached selection from timeline.json")
    extra.add_argument("--work-dir", default="./tmp", help="scratch directory (timeline.json, map frames)")
    extra.add_argument("--timeline", default=None, help="path to timeline.json (default: <work-dir>/timeline.json)")
    extra.add_argument("--device", default=None, choices=["cuda", "cpu"], help="inference device (default: cuda if available)")
    extra.add_argument("--whisper-model", default=understand_mod.WHISPER_MODEL, help="Whisper model size")
    extra.add_argument("--blip-model", default=understand_mod.BLIP2_MODEL, help="BLIP-2 model id")
    extra.add_argument("--fps", type=int, default=30, help="output frame rate")
    extra.add_argument("--preset", default="medium", help="x264 preset (ultrafast..veryslow)")
    extra.add_argument("--threads", type=int, default=0, help="encoder threads (0 = ffmpeg default)")
    extra.add_argument("--workers", type=int, default=4,
                       help="how many parts of the film render at once")
    extra.add_argument("--chunk-size", type=int, default=8,
                       help="shots per independently rendered part (0 = one pass)")
    extra.add_argument("--encoder", choices=["auto", "x264", "nvenc"], default="auto",
                       help="auto uses the GPU encoder when few workers are running")
    extra.add_argument("--resolution", default=None, metavar="WxH", help="force output resolution, e.g. 1920x1080")
    extra.add_argument("--no-location", action="store_true",
                       help="do not draw the location line under the caption")
    extra.add_argument("--no-pair-portraits", action="store_true",
                       help="never show two tall photos side by side")
    extra.add_argument("--no-day-cards", action="store_true",
                       help="do not open each day with a title card")
    extra.add_argument("--no-closing-card", action="store_true",
                       help="do not end on the quote and the full route")
    extra.add_argument("--no-animation", action="store_true",
                       help="hold each picture still instead of easing it in and out")
    extra.add_argument("--no-beat-sync", action="store_true",
                       help="do not cut to the music")
    extra.add_argument("--beat-strength", type=float, default=0.7,
                       help="how far shot pacing follows the music's energy, 0-1")
    extra.add_argument("--map-zoom", type=int, default=12,
                       help="map zoom: higher is closer, 12-13 keeps place names readable")
    extra.add_argument("--map-tiles", choices=["topo", "osm"], default="topo",
                       help="topo = hillshading and peak names, osm = plain street map")
    extra.add_argument("--map-tilt", type=float, default=0.16,
                       help="rake the map back into perspective, 0 = flat top-down")
    extra.add_argument("--map-relief", type=float, default=1.0,
                       help="3d terrain shading and height on the map, 0 = plain tiles")
    extra.add_argument("--no-normalize-audio", action="store_true",
                       help="do not even out clip volumes")
    extra.add_argument("--audio-target", type=float, default=-18.0,
                       help="target level for clip audio, dBFS")
    extra.add_argument("--music-duck", type=float, default=0.40,
                       help="how far the music drops under speech, 0-1 (1 = no ducking)")
    extra.add_argument("--map-grow", type=float, default=1.45,
                       help="how far the map swells while flying to a new place")
    extra.add_argument("--no-map-animation", action="store_true",
                       help="do not fly the map between locations")
    extra.add_argument("--no-geocode", action="store_true",
                       help="skip reverse geocoding, so no location headers")
    extra.add_argument("--dry-run", action="store_true", help="run every phase except the final render")
    extra.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p.parse_args(argv)


def load_env_file(path: Path) -> list[str]:
    """Read `KEY=value` lines from an optional `.env` beside main.py.

    Real environment variables always win; this only fills in what is missing,
    so `OPENROUTER_API_KEY` can live in a file instead of being re-exported for
    every shell.
    """
    if not path.is_file():
        return []
    loaded = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value
                loaded.append(key)
    except OSError as exc:
        log.warning("could not read %s: %s", path, exc)
    return loaded


def parse_duration(value: str | None) -> float | None:
    """Accept "3m30s", "3.5m", "210s" or plain seconds."""
    if value in (None, ""):
        return None
    text = str(value).strip().lower().replace(" ", "")
    match = re.fullmatch(r"(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s?)?", text)
    if not match or not any(match.groups()):
        raise SystemExit(f"--duration must look like 3m30s, 3.5m or 210, got {value!r}")
    minutes = float(match.group(1) or 0)
    seconds = float(match.group(2) or 0)
    total = minutes * 60 + seconds
    if total < 5:
        raise SystemExit(f"--duration must be at least 5 seconds, got {value!r}")
    return total


def parse_fraction(value: str, name: str) -> float:
    """Accept "25%", "25" or "0.25" for a share."""
    text = str(value).strip().rstrip("%")
    try:
        number = float(text)
    except ValueError:
        raise SystemExit(f"{name} must be a percentage or fraction, got {value!r}")
    if str(value).strip().endswith("%") or number > 1.0:
        number /= 100.0
    if not 0.0 <= number <= 1.0:
        raise SystemExit(f"{name} must be between 0 and 100%, got {value!r}")
    return number


def detect_device(requested: str | None) -> str:
    if requested:
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def parse_resolution(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    try:
        w, h = (int(part) for part in value.lower().split("x"))
        return (w - w % 2, h - h % 2)
    except ValueError:
        raise SystemExit(f"--resolution must look like 1920x1080, got {value!r}")


# -------------------------------------------------------------- reporting

def phase_header(number: int, title: str) -> None:
    console.rule(f"[bold cyan]Phase {number} - {title}")


def make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )


def summary_table(title: str, data: dict) -> None:
    table = Table(title=title, title_justify="left", show_header=False, box=None, padding=(0, 2))
    for key, value in data.items():
        table.add_row(f"[dim]{key}[/dim]", str(value))
    console.print(table)


# ------------------------------------------------------------------ main

def run(args: argparse.Namespace) -> int:
    started = time.time()
    loaded = load_env_file(Path(__file__).resolve().parent / ".env")
    if loaded:
        console.print(f"[dim]loaded {', '.join(loaded)} from .env[/dim]")
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        console.print(f"[red]input folder not found:[/red] {input_dir}")
        return 2
    music = assemble_mod.music_files(args.music)
    if not music:
        console.print(f"[red]no music files found:[/red] {' '.join(args.music)}")
        return 2
    parse_duration(args.duration)        # fail fast on a bad --duration

    # only now that the arguments hold up: a bad invocation should not leave a
    # scratch directory behind
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    timeline_path = Path(args.timeline) if args.timeline else work_dir / "timeline.json"
    plan_path = work_dir / "narrative.json"
    device = detect_device(args.device)
    label_fraction = parse_fraction(args.amount_labels, "--amount-labels")

    music_label = (Path(music[0]).name if len(music) == 1
                   else f"{len(music)} tracks ({Path(music[0]).name} ...)")
    console.print(Panel.fit(
        f"[bold]Trip video generator[/bold]\n"
        f"input   {input_dir}\nmusic   {music_label}\noutput  {args.output}\n"
        f"device  {device}    map  {'none' if args.no_map else args.map_mode}"
        + (f"    length  {args.duration}" if args.duration else "")
        + ("    all videos" if args.all_videos else ""),
        border_style="cyan"))
    if not exiftool_available():
        console.print("[yellow]note:[/yellow] exiftool not found - falling back to ffprobe/Pillow metadata "
                      "(install with: sudo apt install libimage-exiftool-perl)")

    # ---------------------------------------------------------- Phase 1
    phase_header(1, "Ingest & sort")
    previous = Timeline.load_if_exists(timeline_path)
    with make_progress() as progress:
        task = progress.add_task("reading metadata", total=None)

        def tick(done: int, total: int) -> None:
            progress.update(task, completed=done, total=total)

        timeline = ingest(input_dir, previous, tick)
    if not len(timeline):
        console.print("[red]no photos or videos found[/red]")
        return 1
    timeline.save(timeline_path)
    summary_table("timeline", summarise(timeline))

    # ---------------------------------------------------------- Phase 2
    phase_header(2, "Auto selection")
    if args.skip_select and previous is not None and any(e.selected for e in timeline):
        console.print(f"[dim]reusing cached selection ({len(timeline.selected)} clips)[/dim]")
    else:
        cfg = selection.SelectionConfig(
            max_clips=args.max_clips, blur_threshold=args.blur_threshold,
            keep_all_videos=args.all_videos, prefer_people=args.prefer_people,
            people_weight=args.people_weight, device=device)
        with make_progress() as progress:
            quality_task = progress.add_task("quality check", total=len(timeline))
            clip_task = progress.add_task("CLIP dedup", total=len(timeline))
            selection.run(timeline, cfg, {
                "quality": lambda d, t: progress.update(quality_task, completed=d, total=t),
                "clip": lambda d, t: progress.update(clip_task, completed=d, total=t),
            })
        timeline.save(timeline_path)
    if not timeline.selected:
        console.print("[red]no clips survived selection - try lowering --blur-threshold[/red]")
        return 1
    summary_table("selection", selection.stats(timeline))
    mediautil.release_memory()

    # ---------------------------------------------------------- Phase 3
    phase_header(3, "Content understanding")
    if args.skip_understand:
        console.print("[dim]--skip-understand: reusing cached captions and transcripts[/dim]")
    else:
        cfg = understand_mod.UnderstandConfig(
            blip_model=args.blip_model, whisper_model=args.whisper_model, device=device,
            find_highlights=not args.no_highlights,
            highlight_budget=args.max_video_duration,
            highlight_segments=args.video_scenes)
        pending_captions = sum(1 for e in timeline.selected if not e.caption)
        pending_audio = sum(1 for e in timeline.selected if e.is_video and e.transcript is None)
        with make_progress() as progress:
            caption_task = progress.add_task("BLIP-2 captions", total=max(pending_captions, 1))
            whisper_task = progress.add_task("Whisper transcripts", total=max(pending_audio, 1))
            scan_task = progress.add_task("video highlights", total=max(
                sum(1 for e in timeline.selected if e.is_video), 1))
            understand_mod.run(timeline, cfg, {
                "caption": lambda d, t: progress.update(caption_task, completed=d, total=t),
                "whisper": lambda d, t: progress.update(whisper_task, completed=d, total=t),
                "highlights": lambda d, t: progress.update(scan_task, completed=d, total=t),
            }, checkpoint=lambda: timeline.save(timeline_path))
        timeline.save(timeline_path)
    summary_table("understanding", understand_mod.stats(timeline))
    # Phase 3's model loads leave tens of gigabytes in the allocator's arenas;
    # hand them back before the render rather than carrying them through it
    mediautil.release_memory()

    # ---------------------------------------------------------- Phase 4
    phase_header(4, "Narrative planning")
    cached_plan = NarrativePlan.load_if_exists(plan_path)
    selected_paths = {e.filepath for e in timeline.selected}
    plan_matches = bool(cached_plan and set(cached_plan.ordered_clips) == selected_paths)
    # A plan written before day cards existed has no titles or closing line, so
    # it is complete for the old film but not for this one.
    plan_complete = bool(cached_plan and (args.no_day_cards or cached_plan.day_titles)
                         and (args.no_closing_card or cached_plan.closing_quote))
    if plan_matches and not plan_complete and not args.skip_narrative:
        console.print("[dim]the cached plan has no day titles or closing line - asking for them[/dim]")
        plan_matches = False

    if args.skip_narrative:
        if plan_matches:
            plan = cached_plan
            console.print(f"[dim]--skip-narrative: reusing {plan_path} "
                          f"({len(plan.captions)} captions)[/dim]")
        else:
            plan = NarrativePlan.chronological(timeline)
            console.print("[dim]--skip-narrative: chronological order, no captions[/dim]")
    elif plan_matches and not args.refresh_narrative:
        # The cut has not changed, so neither should the captions: re-rolling
        # them on every render would rewrite the film's voice each time (and
        # pay for another API call). --refresh-narrative asks for new ones.
        plan = cached_plan
        console.print(f"[dim]reusing the cached plan for these {len(selected_paths)} clips "
                      f"({len(plan.captions)} captions) - pass --refresh-narrative for new ones[/dim]")
    elif not os.environ.get("OPENROUTER_API_KEY"):
        plan = NarrativePlan.chronological(timeline)
        console.print("[yellow]OPENROUTER_API_KEY not set[/yellow] - chronological order, no captions")
    else:
        if cached_plan and set(cached_plan.ordered_clips) != selected_paths:
            console.print("[dim]the selected clips changed - asking for a new plan[/dim]")
        with console.status(f"asking {args.llm_model} for an edit plan "
                            f"({round(label_fraction * len(timeline.selected))} captions)..."):
            plan = narrative_mod.plan_narrative(timeline, narrative_mod.NarrativeConfig(
                model=args.llm_model, label_fraction=label_fraction))
        summary_table("narrative", {
            "clips ordered": len(plan.ordered_clips),
            "captions": len(plan.captions),
            "chapters": ", ".join(c.title for c in plan.chapter_markers) or "-",
            "mood": plan.mood or "-",
        })
    if args.interactive:
        plan = narrative_mod.review_captions(plan, timeline)
    plan.save(plan_path)

    # ---------------------------------------------------------- Phase 5
    phase_header(5, "Map track & places")
    map_tracks: list = []

    if not args.no_geocode:
        with console.status("looking up place names..."):
            places = map_render.resolve_places(timeline, work_dir / "places.json")
        timeline.save(timeline_path)
        console.print(f"[dim]locations: {', '.join(places.values()) or 'none resolved'}[/dim]")

    if args.no_map or args.map_mode == "none":
        console.print("[dim]map disabled[/dim]")
    else:
        map_cfg = map_render.MapConfig(
            out_dir=work_dir / "map_frames",
            cache_dir=work_dir / "tile_cache",
            zoom=args.map_zoom,
            tiles=args.map_tiles,
            tilt=args.map_tilt,
            relief=args.map_relief,
            move_grow=args.map_grow,
            animate=not args.no_map_animation,
        )
        if not args.no_geocode:
            with console.status("reading ground elevation..."):
                map_render.resolve_elevations(timeline, map_cfg)
            timeline.save(timeline_path)
        with console.status(f"rendering map views (zoom {args.map_zoom}, {args.map_tiles} tiles)..."):
            map_tracks = map_render.render_frames(timeline, plan.ordered_clips, map_cfg)
        if map_tracks:
            moves = sum(1 for t in map_tracks if t.animated)
            console.print(f"[dim]{len(map_tracks)} map views, {moves} animated move(s)[/dim]")
        else:
            console.print("[yellow]not enough GPS data - continuing without a map[/yellow]")

    # ---------------------------------------------------------- Phase 6
    phase_header(6, "Video assembly")
    target = parse_duration(args.duration)
    cfg = assemble_mod.AssembleConfig(
        photo_duration=args.photo_duration,
        target_duration=target,
        max_video_duration=args.max_video_duration,
        use_highlights=not args.no_highlights,
        map_mode="none" if (args.no_map or not map_tracks) else args.map_mode,
        resolution=parse_resolution(args.resolution),
        fps=args.fps,
        normalize_audio=not args.no_normalize_audio,
        clip_target_db=args.audio_target,
        duck_volume=args.music_duck,
        preset=args.preset,
        threads=args.threads,
        workers=args.workers,
        chunk_size=args.chunk_size,
        encoder=args.encoder,
        show_location=not args.no_location,
        pair_portraits=not args.no_pair_portraits,
        day_cards=not args.no_day_cards,
        closing_card=not args.no_closing_card,
        animate_cards=not args.no_animation,
        beat_sync=not args.no_beat_sync,
        beat_sync_strength=args.beat_strength,
    )
    music_map = None
    if not args.no_beat_sync:
        with console.status("listening to the music..."):
            music_map = music_mod.analyse(music, work_dir / "music.json")
        if music_map:
            console.print(f"[dim]{music_map.tempo:.0f} BPM, {len(music_map.beats)} beats, "
                          f"{len(music_map.drops)} drop(s)[/dim]")

    if target is None:
        # no length asked for: run exactly as long as the music does, so the
        # film ends with the last song instead of looping or cutting it
        bed = assemble_mod.music_bed_length(music, cfg)
        if bed > 0:
            cfg = dataclasses.replace(cfg, target_duration=bed)
            console.print(f"[dim]no --duration given: matching the music, "
                          f"{bed / 60:.1f} min across {len(music)} track(s)[/dim]")
            # videos and title cards cannot be squeezed, so they set a floor on
            # how short the film can be; say so now rather than after the render
            floor = sum(assemble_mod.clip_duration(e, cfg)
                        for e in timeline.selected if e.is_video)
            floor += sum(cfg.min_photo_duration for e in timeline.selected if not e.is_video)
            if floor > bed * 1.05:
                console.print(f"[yellow]the clips alone run about {floor / 60:.1f} min, "
                              f"longer than the music[/yellow] - the film will be longer "
                              f"and the music will loop. Use --max-video-duration or "
                              f"--max-clips to tighten it, or --duration to force a length.")

    estimated = sum(assemble_mod.clip_duration(e, cfg) for e in timeline.selected)
    console.print(f"[dim]~{estimated / 60:.1f} min of footage at "
                  f"{cfg.resolution or assemble_mod.target_resolution(timeline)}[/dim]")
    if args.dry_run:
        console.print("[yellow]--dry-run: stopping before the render[/yellow]")
        return 0

    with make_progress() as progress:
        build_task = progress.add_task("building clips", total=len(plan.ordered_clips) or 1)
        try:
            output = assemble_mod.assemble(
                timeline, plan, music, args.output, map_tracks, cfg,
                lambda d, t: progress.update(build_task, completed=d, total=t),
                work_dir=work_dir, music=music_map)
        except Exception as exc:  # noqa: BLE001
            log.exception("assembly failed")
            console.print(f"[red]assembly failed:[/red] {exc}")
            return 1

    size_mb = Path(output).stat().st_size / 1e6 if Path(output).exists() else 0
    console.print(Panel.fit(
        f"[bold green]Done[/bold green]  {output}\n"
        f"{size_mb:.1f} MB   {len(plan.ordered_clips)} clips   "
        f"{time.time() - started:.0f}s total",
        border_style="green"))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, show_path=args.verbose, rich_tracebacks=True,
                              markup=False, show_time=False)],
    )
    for noisy in ("httpx", "httpcore", "PIL", "urllib3", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        return run(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted[/yellow]")
        return 130


if __name__ == "__main__":
    sys.exit(main())
