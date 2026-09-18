"""Fast tests for the pipeline's decision logic - no GPU, no network.

Run with:  pytest tests/ -q      (from the trip_video/ directory)
"""

from __future__ import annotations

import json
import sys
import threading
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import assemble
import map_render
import mediautil
import narrative
import selection
from ingest import _parse_iso6709, _parse_timestamp, _valid_gps, _sort_key
from models import GPS, NarrativePlan, Timeline, TimelineEntry


def entry(name: str, minutes: int = 0, lat: float | None = None, lon: float | None = None,
          kind: str = "photo", **kwargs) -> TimelineEntry:
    stamp = (datetime(2024, 6, 12, 8, 0) + timedelta(minutes=minutes)).isoformat()
    return TimelineEntry(
        filepath=f"/trip/{name}", type=kind, timestamp=stamp,
        gps=GPS(lat, lon) if lat is not None else None,
        duration=kwargs.pop("duration", 6.0 if kind == "video" else 0.0),
        quality_score=kwargs.pop("quality_score", 0.5), **kwargs)


# ------------------------------------------------------------ Phase 1

class TestIngest:
    @pytest.mark.parametrize("raw,expected", [
        ("2024:06:12 14:30:05", "2024-06-12T14:30:05"),
        ("2024-06-12T14:30:05", "2024-06-12T14:30:05"),
        (1718202605, "2024-06-12T14:30:05"),
        ("1718202605", "2024-06-12T14:30:05"),
    ])
    def test_timestamp_formats(self, raw, expected):
        assert _parse_timestamp(raw) == expected

    @pytest.mark.parametrize("raw", ["", None, "0000:00:00 00:00:00", "not a date"])
    def test_rejects_junk_timestamps(self, raw):
        assert _parse_timestamp(raw) is None

    def test_iso6709_quicktime_location(self):
        gps = _parse_iso6709("+37.7749-122.4194+010.000/")
        assert gps and round(gps.lat, 4) == 37.7749 and round(gps.lon, 4) == -122.4194

    def test_null_island_is_not_a_fix(self):
        assert _valid_gps(0.0, 0.0) is None
        assert _valid_gps(91.0, 10.0) is None
        assert _valid_gps("46.02", "7.75") is not None

    def test_undated_files_sort_last_by_name(self):
        dated, undated = entry("b.jpg", 10), entry("a.jpg")
        undated.timestamp = None
        assert sorted([undated, dated], key=_sort_key) == [dated, undated]


# ------------------------------------------------------------ Phase 2

class TestSelection:
    def test_group_cap_limits_any_single_place(self):
        cfg = selection.SelectionConfig(max_clips=10, group_share_cap=0.3)
        # 30 shots in one place, 4 in another
        entries = [entry(f"a{i}.jpg", i, 46.0, 7.7) for i in range(30)]
        entries += [entry(f"b{i}.jpg", 100 + i, 47.5, 8.5) for i in range(4)]
        chosen = selection.balanced_select(entries, cfg)
        per_group: dict[int, int] = {}
        for e in chosen:
            per_group[e.group] = per_group.get(e.group, 0) + 1
        assert max(per_group.values()) <= 3, per_group
        assert len(chosen) <= cfg.max_clips

    def test_videos_are_always_kept(self):
        cfg = selection.SelectionConfig(max_clips=3)
        entries = [entry(f"p{i}.jpg", i, 46.0, 7.7) for i in range(20)]
        entries.append(entry("v.mp4", 5, 46.0, 7.7, kind="video"))
        chosen = selection.balanced_select(entries, cfg)
        assert any(e.is_video for e in chosen)

    def test_spread_pick_covers_the_time_range(self):
        entries = [entry(f"p{i}.jpg", i * 10) for i in range(12)]
        picked = selection._spread_pick(entries, 3)
        assert len(picked) == 3
        indexes = sorted(entries.index(p) for p in picked)
        assert indexes[0] < 4 and indexes[-1] > 7  # start and end both represented

    def test_time_gap_grouping_without_gps(self):
        entries = [entry("a.jpg", 0), entry("b.jpg", 30), entry("c.jpg", 400), entry("d.jpg", 430)]
        assert selection.assign_groups(entries) == 2
        assert entries[0].group == entries[1].group != entries[2].group == entries[3].group

    def test_exposure_beats_blur_as_a_drop_reason(self, tmp_path):
        from PIL import Image
        dark = tmp_path / "dark.jpg"
        Image.new("RGB", (400, 300), (4, 4, 4)).save(dark)
        e = TimelineEntry(filepath=str(dark), type="photo")
        selection.score_entry(e, selection.SelectionConfig())
        assert "too dark" in e.drop_reason


# ------------------------------------------------------------ Phase 4

class _MockHandler(BaseHTTPRequestHandler):
    replies: list[str] = []
    received: list[dict] = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        _MockHandler.received.append(json.loads(body))
        reply = _MockHandler.replies.pop(0) if _MockHandler.replies else "{}"
        payload = json.dumps({"choices": [{"message": {"content": reply}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="session")
def tiny_music(tmp_path_factory) -> str:
    """A short real audio file: MoviePy's reader raises during garbage
    collection if handed a path that does not exist."""
    import subprocess

    path = tmp_path_factory.mktemp("audio") / "music.mp3"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    "sine=frequency=330:duration=6", "-c:a", "libmp3lame", str(path)], check=True)
    return str(path)


@pytest.fixture
def mock_openrouter():
    server = HTTPServer(("127.0.0.1", 0), _MockHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _MockHandler.replies, _MockHandler.received = [], []
    yield f"http://127.0.0.1:{server.server_port}", _MockHandler
    server.shutdown()


@pytest.fixture
def three_clip_timeline():
    entries = [entry("a.jpg", 0, 46.0, 7.7), entry("b.jpg", 20, 46.6, 7.8),
               entry("c.mp4", 40, 47.0, 8.3, kind="video")]
    for e in entries:
        e.selected = True
        e.caption = "a machine caption"
    return Timeline(entries)


class TestNarrative:
    def test_parses_a_good_reply(self, mock_openrouter, three_clip_timeline):
        url, handler = mock_openrouter
        handler.replies.append(json.dumps({
            "ordered_clips": ["c2", "c0", "c1"],
            "captions": {"c0": "Golden hour on the ridge line today", "c2": "Arrival"},
            "transition_hints": {"c0": "cut", "c1": "nonsense"},
            "chapter_markers": [{"at_clip": 0, "title": "Arrival"}, {"at_clip": 99, "title": "Nope"}],
            "mood": "calm alpine wander",
        }))
        plan = narrative.plan_narrative(three_clip_timeline, narrative.NarrativeConfig(base_url=url, api_key="k"))
        assert [Path(p).name for p in plan.ordered_clips] == ["c.mp4", "a.jpg", "b.jpg"]
        assert plan.captions["/trip/a.jpg"] == "Golden hour on the ridge line"  # capped at 6 words
        assert plan.transition_for("/trip/a.jpg") == "cut"
        assert plan.transition_for("/trip/b.jpg") == "crossfade"  # invalid hint -> default
        assert [c.title for c in plan.chapter_markers] == ["Arrival"]  # out-of-range dropped
        assert plan.mood == "calm alpine wander"

    def test_unwraps_markdown_and_prose(self, mock_openrouter, three_clip_timeline):
        url, handler = mock_openrouter
        handler.replies.append(
            'Sure! Here is the edit:\n```json\n{"ordered_clips": ["c1", "c0", "c2"], "mood": "x"}\n```\nEnjoy!')
        plan = narrative.plan_narrative(three_clip_timeline, narrative.NarrativeConfig(base_url=url, api_key="k"))
        assert [Path(p).name for p in plan.ordered_clips] == ["b.jpg", "a.jpg", "c.mp4"]

    def test_missing_clips_are_appended(self, mock_openrouter, three_clip_timeline):
        url, handler = mock_openrouter
        handler.replies.append(json.dumps({"ordered_clips": ["c2"]}))
        plan = narrative.plan_narrative(three_clip_timeline, narrative.NarrativeConfig(base_url=url, api_key="k"))
        assert len(plan.ordered_clips) == 3
        assert Path(plan.ordered_clips[0]).name == "c.mp4"

    def test_retries_once_then_falls_back(self, mock_openrouter, three_clip_timeline):
        url, handler = mock_openrouter
        handler.replies.extend(["not json at all", "still not json"])
        plan = narrative.plan_narrative(three_clip_timeline, narrative.NarrativeConfig(base_url=url, api_key="k"))
        assert len(handler.received) == 2, "should retry exactly once"
        assert "stricter" not in plan.mood
        assert [Path(p).name for p in plan.ordered_clips] == ["a.jpg", "b.jpg", "c.mp4"]  # chronological
        assert plan.captions == {}

    def test_no_api_key_raises_not_crashes(self, three_clip_timeline, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        plan = narrative.plan_narrative(three_clip_timeline, narrative.NarrativeConfig(api_key=None))
        assert len(plan.ordered_clips) == 3 and plan.captions == {}

    def test_prompt_carries_captions_not_paths(self, mock_openrouter, three_clip_timeline):
        url, handler = mock_openrouter
        handler.replies.append(json.dumps({"ordered_clips": ["c0", "c1", "c2"]}))
        narrative.plan_narrative(three_clip_timeline, narrative.NarrativeConfig(base_url=url, api_key="k"))
        sent = handler.received[0]["messages"][-1]["content"]
        assert "a machine caption" in sent and "/trip/a.jpg" not in sent


# ------------------------------------------------------------ Phase 5

class TestMap:
    def test_interpolates_between_known_fixes(self):
        points = [(46.0, 7.0), None, (48.0, 9.0)]
        assert map_render.interpolate(points)[1] == (47.0, 8.0)

    def test_extends_at_both_ends(self):
        points = [None, (46.0, 7.0), None]
        filled = map_render.interpolate(points)
        assert filled[0] == filled[1] == filled[2] == (46.0, 7.0)

    def test_skips_when_too_few_fixes(self, tmp_path):
        entries = [entry("a.jpg", 0, 46.0, 7.7), entry("b.jpg", 1), entry("c.jpg", 2)]
        for e in entries:
            e.selected = True
        tl = Timeline(entries)
        tracks = map_render.render_frames(tl, [e.filepath for e in entries],
                                          map_render.MapConfig(out_dir=tmp_path))
        assert tracks == []

    def test_tilt_makes_a_receding_trapezoid(self):
        quad = map_render.tilt_quad(600, 0.30)
        top_width = quad[1][0] - quad[0][0]
        bottom_width = quad[2][0] - quad[3][0]
        assert top_width < bottom_width          # the far edge is narrower
        assert quad[0][1] > 0 and quad[2][1] == 600

    def test_flat_tilt_is_identity(self):
        from PIL import Image
        img = Image.new("RGBA", (64, 64), (10, 20, 30, 255))
        out, matrix = map_render.apply_tilt(img, map_render.MapConfig(tilt=0.0))
        assert out is img and (matrix == __import__("numpy").eye(3)).all()

    def test_homography_round_trips_a_corner(self):
        square = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        quad = map_render.tilt_quad(100, 0.3)
        forward = map_render._homography(square, quad)
        assert map_render._project(forward, (0.0, 0.0)) == pytest.approx(quad[0], abs=1e-6)
        assert map_render._project(forward, (100.0, 100.0)) == pytest.approx(quad[2], abs=1e-6)

    def test_place_label_prefers_the_settlement(self):
        assert map_render._place_label({"village": "Wilderswil", "state": "Bern"}) == "Wilderswil"
        assert map_render._place_label({"county": "Interlaken-Oberhasli"}) == "Interlaken-Oberhasli"
        assert map_render._place_label({"road": "Hauptstrasse"}) is None


# ------------------------------------------------------------ Phase 6

class TestAssemble:
    def test_photo_and_video_durations(self):
        cfg = assemble.AssembleConfig(photo_duration=3.5, max_video_duration=12)
        assert assemble.clip_duration(entry("a.jpg"), cfg) == 3.5
        assert assemble.clip_duration(entry("v.mp4", kind="video", duration=40), cfg) == 12
        assert assemble.clip_duration(entry("v.mp4", kind="video", duration=5), cfg) == 5

    def test_majority_resolution_wins_and_is_even(self):
        entries = []
        for i in range(5):
            e = entry(f"a{i}.jpg", i); e.width, e.height = 1600, 1067; e.selected = True
            entries.append(e)
        e = entry("b.jpg", 9); e.width, e.height = 1280, 720; e.selected = True
        entries.append(e)
        assert assemble.target_resolution(Timeline(entries)) == (1600, 1066)

    def test_duck_envelope_shape(self):
        import numpy as np
        gain = assemble._gain_envelope([(10.0, 20.0)], 0.15, 0.4, 30.0)
        assert gain(np.array([5.0]))[0] == pytest.approx(1.0)
        assert gain(np.array([15.0]))[0] == pytest.approx(0.15)
        assert 0.15 < gain(np.array([9.8]))[0] < 1.0  # fading down

    def test_caption_overlay_is_transparent_outside_the_bar(self):
        img = assemble.caption_overlay("A short caption", (800, 600))
        assert img.size == (800, 600)
        assert img.getpixel((400, 30))[3] == 0          # nothing at the top
        assert any(img.getpixel((x, y))[3] > 0
                   for x in range(40, 400) for y in range(430, 590))

    def test_no_caption_no_overlay(self):
        assert assemble.caption_overlay("", (800, 600)) is None


# ------------------------------------------------------------ models

class TestTimeline:
    def test_roundtrip_preserves_unknown_keys(self, tmp_path):
        e = entry("a.jpg", 0, 46.0, 7.7)
        e.extra = {"future_field": 42}
        path = tmp_path / "timeline.json"
        Timeline([e]).save(path)
        loaded = Timeline.load(path)
        assert loaded[0].extra["future_field"] == 42
        assert loaded[0].gps.lat == 46.0

    def test_by_path_falls_back_to_basename(self):
        tl = Timeline([entry("a.jpg")])
        assert tl.by_path("/somewhere/else/a.jpg") is not None

    def test_chronological_plan_has_no_captions(self):
        entries = [entry("a.jpg", 0), entry("b.jpg", 5)]
        for e in entries:
            e.selected = True
        plan = NarrativePlan.chronological(Timeline(entries))
        assert plan.captions == {} and len(plan.ordered_clips) == 2


class TestCaptionBand:
    def test_overlay_band_is_the_full_width(self):
        # the map moved to the top-left corner, so the caption has the run of
        # the bottom of the frame again
        cfg = assemble.AssembleConfig(map_mode="overlay")
        assert assemble.caption_band((1000, 600), cfg, has_map=True) == (0, 1000)

    def test_map_sits_top_left_in_overlay_mode(self):
        cfg = assemble.AssembleConfig(map_mode="overlay")
        size, (x, y) = assemble.map_geometry((1920, 1080), cfg)
        assert x < 1920 * 0.05 and y < 1080 * 0.05           # tucked into the corner
        assert size == pytest.approx(1920 * 0.14, abs=2)     # half the previous size
        assert size + y < 1080 * 0.35                        # nowhere near the caption

    def test_side_band_stops_left_of_the_panel(self):
        cfg = assemble.AssembleConfig(map_mode="side", side_panel_fraction=0.30)
        assert assemble.caption_band((1000, 600), cfg, has_map=True) == (0, 700)

    def test_no_map_uses_the_whole_width(self):
        cfg = assemble.AssembleConfig(map_mode="overlay")
        assert assemble.caption_band((1000, 600), cfg, has_map=False) == (0, 1000)

    def test_caption_sits_bottom_left(self):
        img = assemble.caption_overlay("Nobody admits being lost", (1000, 600))
        pixels = img.load()
        bottom_left = any(pixels[x, y][3] > 0 for x in range(20, 400) for y in range(480, 580))
        bottom_right = any(pixels[x, y][3] > 0 for x in range(700, 990) for y in range(480, 580))
        top = any(pixels[x, y][3] > 0 for x in range(0, 1000) for y in range(0, 100))
        assert bottom_left and not bottom_right and not top


class _RejectJsonModeHandler(_MockHandler):
    """Rejects the request while `response_format` is present, like some models."""

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        _MockHandler.received.append(body)
        if "response_format" in body:
            self.send_response(400)
            payload = b'{"error": "response_format is not supported"}'
        else:
            self.send_response(200)
            payload = json.dumps({"choices": [{"message": {"content": json.dumps(
                {"ordered_clips": ["c0", "c1", "c2"], "mood": "plain mode"})}}]}).encode()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def test_retries_without_json_mode(three_clip_timeline):
    server = HTTPServer(("127.0.0.1", 0), _RejectJsonModeHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _MockHandler.replies, _MockHandler.received = [], []
    try:
        plan = narrative.plan_narrative(three_clip_timeline, narrative.NarrativeConfig(
            base_url=f"http://127.0.0.1:{server.server_port}", api_key="k"))
    finally:
        server.shutdown()
    sent = [b.get("response_format") is not None for b in _MockHandler.received]
    assert sent[:2] == [True, False]      # json mode rejected, retried without it
    assert plan.mood == "plain mode"


class TestRenderIntegration:
    """A real (tiny) render: exercises MoviePy, the audio mix and the layout."""

    @staticmethod
    def _photo(path: Path, color: tuple[int, int, int]) -> None:
        from PIL import Image
        Image.new("RGB", (320, 240), color).save(path, quality=90)

    def test_renders_and_skips_unknown_planned_clips(self, tmp_path, tiny_music):
        import subprocess
        a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
        self._photo(a, (30, 90, 160))
        self._photo(b, (160, 90, 30))
        music = tiny_music

        entries = []
        for path in (a, b):
            e = TimelineEntry(filepath=str(path), type="photo", timestamp="2024-06-12T08:00:00")
            e.width, e.height, e.selected = 320, 240, True
            entries.append(e)
        tl = Timeline(entries)

        plan = NarrativePlan(
            ordered_clips=[str(a), "/gone/missing.jpg", str(b)],   # middle clip cannot resolve
            captions={str(b): "Second frame"},
            transition_hints={str(b): "cut"},
        )
        out = tmp_path / "out.mp4"
        cfg = assemble.AssembleConfig(photo_duration=1.0, fps=6, preset="ultrafast",
                                      map_mode="none", resolution=(320, 240))
        assemble.assemble(tl, plan, music, str(out), None, cfg)  # no map

        assert out.exists() and out.stat().st_size > 0
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(out)], capture_output=True, text=True, check=True)
        assert float(probe.stdout.strip()) == pytest.approx(2.0, abs=0.35)  # two 1s photos
        strays = list(Path.cwd().glob(f"{out.stem}TEMP_MPY*")) + list(tmp_path.glob("*TEMP_MPY*"))
        assert not strays, f"left MoviePy temp files behind: {strays}"


class TestUnderstandCaching:
    def test_already_captioned_clips_are_skipped(self, monkeypatch):
        import understand
        calls = []

        class _Boom:
            def __init__(self, cfg):
                calls.append("loaded")
                raise AssertionError("BLIP-2 must not load when nothing is pending")

        monkeypatch.setattr(understand, "Captioner", _Boom)
        e = entry("a.jpg", caption="already there")
        assert understand.caption_entries([e], understand.UnderstandConfig()) == 0
        assert calls == []

    def test_checkpoint_fires_as_results_land(self, monkeypatch):
        import understand
        saves = []

        class _Fake:
            def __init__(self, cfg):
                pass

            def caption_batch(self, images):
                return [("A caption", 0.9) for _ in images]

            def close(self):
                pass

        monkeypatch.setattr(understand, "Captioner", _Fake)
        monkeypatch.setattr(understand, "load_pil", lambda p, max_side=None: object())
        entries = [entry(f"a{i}.jpg", i) for i in range(4)]
        for e in entries:
            e.caption = None
        understand.caption_entries(entries, understand.UnderstandConfig(batch_size=2),
                                   checkpoint=lambda: saves.append(1))
        assert len(saves) == 2                       # one per batch
        assert all(e.caption == "A caption" for e in entries)

    def test_empty_transcript_means_checked_not_pending(self):
        import understand
        silent = entry("v.mp4", kind="video", transcript="")
        assert understand.transcribe_entries([silent], understand.UnderstandConfig()) == 0


class TestVideoDedup:
    """Immich exports ship a `name+1.ext` twin of many files."""

    def test_duplicate_videos_collapse_to_one(self, monkeypatch):
        import numpy as np
        import selection as sel

        vids = [entry("clip.mp4", 0, 46.0, 7.7, kind="video", duration=8.0),
                entry("clip+1.mp4", 0, 46.0, 7.7, kind="video", duration=8.0),
                entry("other.mp4", 60, 46.0, 7.7, kind="video", duration=6.0)]
        for i, e in enumerate(vids):
            e.blur_score = 100.0 + i
        # identical embeddings for the pair, a different one for the third clip
        embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        monkeypatch.setattr(sel, "clip_embeddings", lambda *a, **kw: embeddings)

        kept = sel.deduplicate(vids, sel.SelectionConfig())
        names = sorted(e.name for e in kept)
        assert names == ["clip+1.mp4", "other.mp4"], names   # the sharper twin survives
        assert vids[0].drop_reason == "near-duplicate"

    def test_a_video_is_never_dropped_for_a_photo(self, monkeypatch):
        import numpy as np
        import selection as sel

        items = [entry("shot.jpg", 0, 46.0, 7.7), entry("shot.mp4", 0, 46.0, 7.7, kind="video")]
        items[0].blur_score, items[1].blur_score = 900.0, 90.0   # photo is much sharper
        monkeypatch.setattr(sel, "clip_embeddings",
                            lambda *a, **kw: np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32))
        kept = sel.deduplicate(items, sel.SelectionConfig())
        assert sorted(e.name for e in kept) == ["shot.jpg", "shot.mp4"]


class TestNoTopHeader:
    """The title at the top was removed: the location line names the place."""

    def test_overlay_draws_nothing_in_the_top_strip(self):
        img = assemble.caption_overlay("Nobody admits being lost", (1000, 600),
                                       location="Interlaken · 568 m")
        pixels = img.load()
        assert not any(pixels[x, y][3] > 0
                       for x in range(0, 1000) for y in range(0, 220))

    def test_caption_overlay_takes_no_header(self):
        import inspect
        assert "header" not in inspect.signature(assemble.caption_overlay).parameters

    def test_assemble_takes_no_place_headers(self):
        import inspect
        assert "place_headers" not in inspect.signature(assemble.assemble).parameters


class TestAudioNormalisation:
    """Phone clips arrive between -25 and -64 dBFS; the mix has to even that out."""

    cfg = assemble.AssembleConfig()

    def db(self, volume: float) -> float:
        import math
        return 20 * math.log10(max(volume, 1e-9))

    def test_quiet_clip_is_lifted(self):
        quiet = {"mean_db": -38.0, "max_db": -18.0}
        volume = assemble.normalized_volume(quiet, 1.0, -20.0, self.cfg)
        assert self.db(volume) == pytest.approx(17.0, abs=0.5)

    def test_peaks_are_never_clipped(self):
        # quiet on average but with peaks near full scale: lift only to headroom
        spiky = {"mean_db": -40.0, "max_db": -0.5}
        volume = assemble.normalized_volume(spiky, 1.0, -20.0, self.cfg)
        assert self.db(volume) + spiky["max_db"] <= -self.cfg.headroom_db + 1e-6

    def test_silence_is_not_amplified(self):
        silent = {"mean_db": -63.6, "max_db": -39.8}
        assert assemble.normalized_volume(silent, 0.4, -20.0, self.cfg) == 0.4

    def test_gain_is_capped(self):
        very_quiet = {"mean_db": -70.0, "max_db": -60.0}
        cfg = assemble.AssembleConfig(silence_floor_db=-99.0, max_gain_db=18.0)
        assert self.db(assemble.normalized_volume(very_quiet, 1.0, -20.0, cfg)) <= 18.0 + 1e-6

    def test_unmeasured_and_disabled_fall_back_to_the_role_volume(self):
        assert assemble.normalized_volume(None, 0.4, -20.0, self.cfg) == 0.4
        off = assemble.AssembleConfig(normalize_audio=False)
        assert assemble.normalized_volume({"mean_db": -50.0, "max_db": -30.0}, 0.4, -20.0, off) == 0.4

    def test_measure_audio_reads_a_real_file(self, tiny_music):
        levels = mediautil.measure_audio(tiny_music)
        assert levels and -60.0 < levels["mean_db"] < 0.0


class TestLocationLabel:
    def test_name_with_elevation(self):
        e = entry("a.jpg", 0, 47.27, 11.51)
        e.place, e.altitude = "Kitzbuehel", 1878.2
        assert assemble.location_label(e) == "Kitzbuehel · 1,878 m"

    def test_coordinates_when_unnamed(self):
        e = entry("a.jpg", 0, 47.2712, 11.5134)
        assert assemble.location_label(e) == "47.2712, 11.5134"

    def test_nothing_without_gps(self):
        assert assemble.location_label(entry("a.jpg")) is None

    def test_label_is_drawn_below_the_caption(self):
        img = assemble.caption_overlay("Nobody admits being lost", (1000, 600),
                                       location="Kitzbuehel · 1,878 m")
        pixels = img.load()
        rows = [y for y in range(300, 600)
                if any(pixels[x, y][3] > 0 for x in range(20, 500))]
        assert rows, "nothing drawn"
        # the location line is the lowest ink on the frame
        assert max(rows) > 540


class TestTerrainRelief:
    def _dem(self, peak: float = 3000.0):
        import numpy as np
        ys, xs = np.mgrid[0:64, 0:64]
        return (peak * np.exp(-((xs - 32) ** 2 + (ys - 32) ** 2) / 300.0)).astype("float32")

    def test_hillshade_lights_the_sun_facing_slope(self):
        cfg = map_render.MapConfig(sun_azimuth=315.0, sun_altitude=45.0)
        shade = map_render.hillshade(self._dem(), 30.0, cfg)
        assert 0.0 <= shade.min() and shade.max() <= 1.0
        # north-west light: the NW flank is brighter than the SE flank
        assert shade[20, 20] > shade[44, 44]

    def test_relief_changes_the_image_and_flat_terrain_does_not(self):
        import numpy as np
        from PIL import Image
        base = Image.fromarray(np.full((64, 64, 4), 200, dtype="uint8"), "RGBA")
        cfg = map_render.MapConfig(size=64, relief=1.0)
        shaded = map_render.apply_relief(base, self._dem(), 12, (47.0, 11.0), cfg)
        assert np.asarray(shaded).std() > 1.0            # terrain now has shape

        flat = np.zeros((64, 64), dtype="float32")
        unshaded = map_render.apply_relief(base, flat, 12, (47.0, 11.0), cfg)
        assert np.asarray(unshaded)[..., :3].std() < 1.0  # nothing to shade

    def test_relief_disabled_returns_the_original(self):
        from PIL import Image
        base = Image.new("RGBA", (64, 64), (200, 200, 200, 255))
        cfg = map_render.MapConfig(size=64, relief=0.0)
        assert map_render.apply_relief(base, self._dem(), 12, (47.0, 11.0), cfg) is base
        assert map_render.apply_relief(base, None, 12, (47.0, 11.0),
                                       map_render.MapConfig(size=64)) is base

    def test_terrarium_decoding(self):
        import numpy as np
        from PIL import Image
        # terrarium: height = R*256 + G + B/256 - 32768
        tile = Image.fromarray(np.array([[[128, 100, 0]]], dtype="uint8"), "RGB")
        assert map_render._decode_terrarium(tile)[0, 0] == pytest.approx(128 * 256 + 100 - 32768)


class TestLabelAmount:
    def test_fraction_parsing(self):
        import main
        assert main.parse_fraction("25%", "--amount-labels") == pytest.approx(0.25)
        assert main.parse_fraction("0.25", "--amount-labels") == pytest.approx(0.25)
        assert main.parse_fraction("40", "--amount-labels") == pytest.approx(0.40)

    def test_rejects_nonsense(self):
        import main
        with pytest.raises(SystemExit):
            main.parse_fraction("loads", "--amount-labels")
        with pytest.raises(SystemExit):
            main.parse_fraction("300%", "--amount-labels")

    def test_target_reaches_the_prompt(self, mock_openrouter, three_clip_timeline):
        url, handler = mock_openrouter
        handler.replies.append(json.dumps({"ordered_clips": ["c0", "c1", "c2"],
                                           "captions": {"c0": "One", "c1": "Two", "c2": "Three"}}))
        narrative.plan_narrative(three_clip_timeline, narrative.NarrativeConfig(
            base_url=url, api_key="k", label_fraction=1.0))
        system = handler.received[0]["messages"][0]["content"]
        assert "EXACTLY 3 of the 3 clips" in system

    def test_shortfall_triggers_one_top_up(self, mock_openrouter, three_clip_timeline):
        url, handler = mock_openrouter
        handler.replies.append(json.dumps({"ordered_clips": ["c0", "c1", "c2"],
                                           "captions": {"c0": "Only one"}}))
        handler.replies.append(json.dumps({"captions": {"c1": "Second thoughts",
                                                        "c2": "Third, reluctantly"}}))
        plan = narrative.plan_narrative(three_clip_timeline, narrative.NarrativeConfig(
            base_url=url, api_key="k", label_fraction=1.0))
        assert len(handler.received) == 2
        assert len(plan.captions) == 3
        assert "uncaptioned" in handler.received[1]["messages"][1]["content"]

    def test_no_top_up_when_target_is_met(self, mock_openrouter, three_clip_timeline):
        url, handler = mock_openrouter
        handler.replies.append(json.dumps({"ordered_clips": ["c0", "c1", "c2"],
                                           "captions": {"c0": "Enough", "c1": "Plenty"}}))
        plan = narrative.plan_narrative(three_clip_timeline, narrative.NarrativeConfig(
            base_url=url, api_key="k", label_fraction=0.34))
        assert len(handler.received) == 1 and len(plan.captions) == 2


class TestMapSwell:
    """The map is small while parked and swells briefly during a move."""

    def _track(self, tmp_path, scales):
        from PIL import Image
        import map_render
        frames = []
        for i, _ in enumerate(scales):
            path = tmp_path / f"f{i}.png"
            Image.new("RGBA", (600, 600), (10, 120, 60, 255)).save(path)
            frames.append(path)
        return map_render.MapTrack(frames, 12.0, True, list(scales))

    def test_scale_schedule_rises_and_returns(self):
        import map_render
        track = map_render.MapTrack([], 12.0, True, [1.0, 1.6, 2.2, 2.2, 1.4, 1.0])
        assert track.max_scale == 2.2
        assert track.scale_at(0) == 1.0
        assert track.scale_at(2) == 2.2
        assert track.scale_at(99) == 1.0           # holds the parked size at the end

    def test_still_track_never_scales(self, tmp_path):
        import map_render
        still = map_render.MapTrack([tmp_path / "x.png"], 12.0, False, [1.0])
        assert still.max_scale == 1.0 and still.scale_at(5) == 1.0

    def test_canvas_fits_the_largest_frame_and_holds_position(self, tmp_path):
        cfg = assemble.AssembleConfig(map_mode="overlay")
        frame_size = (1920, 1080)
        size, position = assemble.map_geometry(frame_size, cfg)
        track = self._track(tmp_path, [1.0, 1.6, 2.2, 1.0])
        clip = assemble.map_animation_clip(track, frame_size, cfg, 3.0)
        assert clip is not None
        canvas = clip.get_frame(0).shape[0]
        assert canvas == pytest.approx(size * 2.2, abs=2)   # room for the swell
        assert tuple(clip.pos(0)) == position               # anchored in the corner

    def test_map_grows_then_shrinks_over_time(self, tmp_path):
        import numpy as np
        cfg = assemble.AssembleConfig(map_mode="overlay")
        track = self._track(tmp_path, [1.0, 1.6, 2.2, 1.0])
        clip = assemble.map_animation_clip(track, (1920, 1080), cfg, 4.0)
        mask = clip.mask

        def covered(t):
            return float(np.asarray(mask.get_frame(t)).sum())

        parked, mid, grown = covered(0.0), covered(1.0 / 12), covered(2.0 / 12)
        assert parked < mid < grown                 # swells while travelling
        assert covered(3.0 / 12) == pytest.approx(parked, rel=0.02)   # and returns


class TestAltitude:
    def test_real_values_pass_through(self):
        from ingest import _valid_altitude
        assert _valid_altitude("1878.2") == 1878.2
        assert _valid_altitude(586) == 586.0

    def test_placeholder_zero_is_not_sea_level(self):
        # ffmpeg writes "+0.000" into a QuickTime location when it has none
        from ingest import _iso6709_altitude, _valid_altitude
        assert _valid_altitude(0) is None
        assert _iso6709_altitude("+46.0207+7.7491+0.000/") is None
        assert _iso6709_altitude("+47.2712+011.5134+586.000/") == 586.0

    def test_below_sea_level_reference(self):
        from ingest import _valid_altitude
        assert _valid_altitude(12.0, below_sea="1") == -12.0

    def test_out_of_range_is_rejected(self):
        from ingest import _valid_altitude
        assert _valid_altitude(99999) is None
        assert _valid_altitude("not a number") is None
