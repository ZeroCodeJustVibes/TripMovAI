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
        assert any(img.getpixel((x, 500))[3] > 0 for x in range(40, 400))

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
    def test_overlay_band_stops_left_of_the_map(self):
        cfg = assemble.AssembleConfig(map_mode="overlay", map_width_fraction=0.28)
        x0, x1 = assemble.caption_band((1000, 600), cfg, has_map=True)
        size, position = assemble.map_geometry((1000, 600), cfg)
        assert x0 == 0 and x1 <= position[0]      # caption never runs under the map

    def test_map_sits_bottom_right_in_overlay_mode(self):
        cfg = assemble.AssembleConfig(map_mode="overlay", map_width_fraction=0.28)
        size, (x, y) = assemble.map_geometry((1920, 1080), cfg)
        assert x + size < 1920 and x > 1920 * 0.6     # right-hand side
        assert y + size < 1080 and y > 1080 * 0.4     # lower half

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

    def test_header_sits_top_left(self):
        img = assemble.caption_overlay("", (1000, 600), header="Interlaken")
        pixels = img.load()
        assert any(pixels[x, y][3] > 0 for x in range(20, 400) for y in range(20, 140))
        assert not any(pixels[x, y][3] > 0 for x in range(0, 1000) for y in range(400, 600))


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
    assert [b.get("response_format") is not None for b in _MockHandler.received] == [True, False]
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


class TestPlaceHeaders:
    """A week in one valley yields many GPS clusters but one place name."""

    def _run(self, groups, names, tmp_path, music):
        """Return the header drawn for each clip, by stubbing the overlay call."""
        import assemble as A

        from PIL import Image

        drawn = []
        entries = []
        for i, group in enumerate(groups):
            path = tmp_path / f"p{i}.jpg"
            Image.new("RGB", (64, 48), (40 + i * 20, 80, 120)).save(path)
            e = entry(f"p{i}.jpg", i)
            e.filepath = str(path)
            e.group, e.selected = group, True
            e.width, e.height = 64, 48
            entries.append(e)
        tl = Timeline(entries)
        plan = NarrativePlan(ordered_clips=[e.filepath for e in entries])

        original = A.caption_overlay

        def spy(text, frame_size, header=None, text_band=None):
            drawn.append(header)
            return None

        A.caption_overlay = spy
        try:
            A.assemble(tl, plan, music, str(tmp_path / "x.mp4"), None,
                       A.AssembleConfig(photo_duration=0.5, fps=4, preset="ultrafast",
                                        map_mode="none", resolution=(64, 48)),
                       place_headers=names)
        finally:
            A.caption_overlay = original
        return drawn

    def test_header_only_on_a_new_name(self, tmp_path, tiny_music):
        # five clusters, but only two actual places
        drawn = self._run([0, 1, 2, 3, 4], {0: "Kitzbuehel", 1: "Kitzbuehel",
                                            2: "Kitzbuehel", 3: "Innsbruck", 4: "Innsbruck"},
                          tmp_path, tiny_music)
        assert drawn == ["Kitzbuehel", "Innsbruck"]

    def test_unnamed_clusters_do_not_clear_the_header(self, tmp_path, tiny_music):
        drawn = self._run([0, None, 0, 1], {0: "Zermatt", 1: "Interlaken"}, tmp_path, tiny_music)
        assert drawn == ["Zermatt", "Interlaken"]
