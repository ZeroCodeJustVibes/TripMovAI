"""Fast tests for the pipeline's decision logic - no GPU, no network.

Run with:  pytest tests/ -q      (from the trip_video/ directory)
"""

from __future__ import annotations

import json
import sys

import numpy as np
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

    def test_caption_overlay_is_transparent_outside_the_card(self):
        layout = assemble.layout_for((800, 600), (800, 600), assemble.AssembleConfig())
        img = assemble.caption_overlay("A short caption", layout, assemble.AssembleConfig())
        assert img.size == (800, 600)
        assert img.getpixel((400, 30))[3] == 0          # nothing at the top
        assert any(img.getpixel((x, y))[3] > 0
                   for x in range(40, 400) for y in range(430, 590))

    def test_no_caption_no_overlay(self):
        layout = assemble.layout_for((800, 600), (800, 600), assemble.AssembleConfig())
        assert assemble.caption_overlay("", layout, assemble.AssembleConfig()) is None


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


class TestLayout:
    """One grid: a fixed UI column, the picture taking everything left over."""

    cfg = assemble.AssembleConfig()

    def test_landscape_sits_flush_beside_the_column(self):
        layout = assemble.layout_for((4000, 3000), (1920, 1080), self.cfg)
        x, y, w, h = layout.card
        size, (map_x, map_y) = assemble.map_geometry(layout, self.cfg)
        assert map_x + size <= x                      # map entirely left of the picture
        assert x - (map_x + size) < layout.gap * 2    # and flush against it
        assert h == pytest.approx(1080 - 2 * round(0.024 * 1920), abs=2)   # full height

    def test_map_fills_the_column(self):
        layout = assemble.layout_for((4000, 3000), (1920, 1080), self.cfg)
        size, _ = assemble.map_geometry(layout, self.cfg)
        assert size == pytest.approx(round(0.24 * 1920), abs=2)
        assert size > 400          # far bigger than the old badge

    def test_picture_keeps_its_aspect_and_stays_in_frame(self):
        for media in [(4000, 3000), (1600, 1067), (1920, 1080), (3024, 4032)]:
            layout = assemble.layout_for(media, (1920, 1080), self.cfg)
            x, y, w, h = layout.card
            assert w / h == pytest.approx(media[0] / media[1], rel=0.02)
            assert x >= 0 and y >= 0 and x + w <= 1920 and y + h <= 1080

    def test_without_a_map_the_picture_takes_the_whole_frame(self):
        cfg = assemble.AssembleConfig(map_mode="none")
        wide = assemble.layout_for((1600, 1067), (1920, 1080), cfg, with_map=False)
        columned = assemble.layout_for((1600, 1067), (1920, 1080), self.cfg)
        assert wide.card[2] > columned.card[2]        # bigger without the column
        x, y, w, h = wide.card
        assert wide.ui_left >= x and wide.ui_bottom <= y + h   # caption sits on it

    def test_ui_column_does_not_move_between_clips(self):
        wide = assemble.layout_for((4000, 3000), (1920, 1080), self.cfg)
        tall = assemble.layout_for((3024, 4032), (1920, 1080), self.cfg)
        assert (wide.ui_left, wide.ui_top, wide.ui_right) == (tall.ui_left, tall.ui_top, tall.ui_right)

    def test_backdrop_fills_the_frame_and_is_dimmed(self):
        import numpy as np
        from PIL import Image
        source = Image.new("RGB", (400, 300), (220, 200, 180))
        layout = assemble.layout_for((400, 300), (640, 360), self.cfg)
        plate = assemble.backdrop(source, layout, self.cfg)
        assert plate.size == (640, 360)
        assert np.asarray(plate.convert("RGB")).mean() < 200

    def test_card_border_is_drawn_per_panel(self):
        layout = assemble.layout_for([(3024, 4032), (3024, 4032)], (1920, 1080), self.cfg)
        overlay = assemble.card_border_overlay(layout, self.cfg)
        pixels = overlay.load()
        assert len(layout.panels) == 2
        for x, y, w, h in layout.panels:
            assert pixels[x + w // 2, y][3] > 0        # each panel outlined
        mid = (layout.panels[0][0] + layout.panels[0][2] + layout.panels[1][0]) // 2
        assert pixels[mid, layout.panels[0][1] + 10][3] == 0   # gap between them is clear


class TestPortraitPairing:
    """Two tall photos side by side beat one tall photo and a lot of blur."""

    cfg = assemble.AssembleConfig()

    def _entries(self, shapes):
        out = []
        for i, (w, h) in enumerate(shapes):
            e = entry(f"p{i}.jpg", i, kind="video" if w == 0 else "photo")
            e.width, e.height = (abs(w) or 1920), h
            out.append((i, e.filepath, e))
        return out

    def test_two_portraits_pair_up(self):
        entries = self._entries([(3024, 4032), (3024, 4032), (4000, 3000)])
        paired = assemble.pair_portraits(entries, self.cfg, animated=set())
        assert len(paired) == 2                      # three clips became two
        assert paired[0][3] is not None              # first has a companion
        assert paired[1][3] is None                  # the landscape stands alone

    def test_landscape_never_pairs(self):
        entries = self._entries([(4000, 3000), (4000, 3000)])
        paired = assemble.pair_portraits(entries, self.cfg, animated=set())
        assert [item[3] for item in paired] == [None, None]

    def test_videos_never_pair(self):
        entries = self._entries([(1080, 1920), (1080, 1920)])
        for _, _, e in entries:
            e.type = "video"
        paired = assemble.pair_portraits(entries, self.cfg, animated=set())
        assert [item[3] for item in paired] == [None, None]

    def test_a_flying_map_keeps_its_own_clip(self):
        entries = self._entries([(3024, 4032), (3024, 4032)])
        paired = assemble.pair_portraits(entries, self.cfg, animated={1})
        assert [item[3] for item in paired] == [None, None]

    def test_pairing_can_be_switched_off(self):
        entries = self._entries([(3024, 4032), (3024, 4032)])
        off = assemble.AssembleConfig(pair_portraits=False)
        assert [item[3] for item in assemble.pair_portraits(entries, off, set())] == [None, None]

    def test_a_pair_fills_more_of_the_picture_area(self):
        single = assemble.layout_for((3024, 4032), (1920, 1080), self.cfg)
        pair = assemble.layout_for([(3024, 4032), (3024, 4032)], (1920, 1080), self.cfg)
        assert len(pair.panels) == 2
        single_area = single.card[2] * single.card[3]
        pair_area = sum(w * h for _, _, w, h in pair.panels)
        assert pair_area > single_area * 1.4
        assert pair.card[0] + pair.card[2] <= 1920


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
        layout = assemble.layout_for((1000, 600), (1000, 600), assemble.AssembleConfig())
        img = assemble.caption_overlay("Nobody admits being lost", layout,
                                       assemble.AssembleConfig(), location="Interlaken · 568 m")
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
        layout = assemble.layout_for((1000, 600), (1000, 600), assemble.AssembleConfig())
        img = assemble.caption_overlay("Nobody admits being lost", layout,
                                       assemble.AssembleConfig(), location="Kitzbuehel · 1,878 m")
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
        layout = assemble.layout_for((1600, 1067), (1920, 1080), cfg)
        size, position = assemble.map_geometry(layout, cfg)
        track = self._track(tmp_path, [1.0, 1.6, 2.2, 1.0])
        clip = assemble.map_animation_clip(track, layout, cfg, 3.0)
        assert clip is not None
        canvas = clip.get_frame(0).shape[0]
        assert canvas == pytest.approx(size * 2.2, abs=2)   # room for the swell
        assert tuple(clip.pos(0)) == position               # anchored in the corner

    def test_map_grows_then_shrinks_over_time(self, tmp_path):
        import numpy as np
        cfg = assemble.AssembleConfig(map_mode="overlay")
        layout = assemble.layout_for((1600, 1067), (1920, 1080), cfg)
        track = self._track(tmp_path, [1.0, 1.6, 2.2, 1.0])
        clip = assemble.map_animation_clip(track, layout, cfg, 4.0)
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




class TestCaptionFitsTheColumn:
    cfg = assemble.AssembleConfig()

    @pytest.mark.parametrize("location", [
        "Hall in Tirol · 558 m",
        "Matrei in Osttirol · 2,177 m",
        "Neukirchen am Großvenediger · 1,487 m",     # the longest real name
        "47.2712, 11.5134 · 1,878 m",
    ])
    def test_location_never_overflows(self, location):
        layout = assemble.layout_for((4000, 3000), (1920, 1080), self.cfg)
        img = assemble.caption_overlay("Nobody admits being lost", layout, self.cfg,
                                       location=location)
        pixels = img.load()
        inked = [x for x in range(1920)
                 if any(pixels[x, y][3] > 0 for y in range(700, 1080))]
        assert max(inked) <= layout.ui_right    # stays inside the column
        assert min(inked) >= layout.ui_left - layout.gap

    def test_long_caption_wraps_rather_than_clipping(self):
        layout = assemble.layout_for((4000, 3000), (1920, 1080), self.cfg)
        img = assemble.caption_overlay("A caption long enough to need three whole lines",
                                       layout, self.cfg, location="Innsbruck · 574 m")
        pixels = img.load()
        rows = [y for y in range(1080) if any(pixels[x, y][3] > 0 for x in range(0, 700))]
        assert max(rows) - min(rows) > 100      # it grew downward instead of clipping


class TestDayCards:
    cfg = assemble.AssembleConfig()

    def _timeline(self):
        entries = []
        for day, (date, count) in enumerate([("2026-02-25", 3), ("2026-02-26", 2)]):
            for i in range(count):
                e = entry(f"d{day}_{i}.jpg", day * 1440 + i * 30, 47.2 + day * 0.1, 11.5)
                e.timestamp = f"{date}T{8 + i:02d}:00:00"
                e.selected, e.place = True, ["Innsbruck", "Kitzbuehel"][day]
                e.width, e.height = 4000, 3000
                e.quality_score = 0.5 + i * 0.1
                entries.append(e)
        return Timeline(entries)

    def test_one_card_opens_each_day(self):
        tl = self._timeline()
        plan = NarrativePlan(ordered_clips=[e.filepath for e in tl.selected],
                             day_titles={"2026-02-25": "Everything uphill, obviously"})
        shots = assemble.plan_shots(tl, plan, self.cfg, map_tracks=None)
        days = [s for s in shots if s.kind == "day"]
        assert len(days) == 2
        assert days[0].eyebrow == "DAY 1" and days[1].eyebrow == "DAY 2"
        assert days[0].caption == "Everything uphill, obviously"
        assert shots.index(days[0]) == 0                      # opens the film
        assert shots.index(days[1]) < len(shots) - 1

    def test_day_without_a_title_falls_back_to_the_date(self):
        tl = self._timeline()
        shots = assemble.plan_shots(tl, NarrativePlan(
            ordered_clips=[e.filepath for e in tl.selected]), self.cfg, map_tracks=None)
        assert [s for s in shots if s.kind == "day"][0].caption == "Wed 25 Feb"

    def test_hero_prefers_a_good_landscape_photo(self):
        tl = self._timeline()
        day_one = [e for e in tl.selected if e.timestamp.startswith("2026-02-25")]
        day_one[0].width, day_one[0].height = 3024, 4032     # portrait, best score
        day_one[0].quality_score = 0.99
        hero = assemble.hero_for(day_one)
        assert not assemble.is_portrait(hero)
        assert hero is max([e for e in day_one if not assemble.is_portrait(e)],
                           key=lambda e: e.quality_score)

    def test_closing_card_carries_the_quote_and_a_summary(self):
        tl = self._timeline()
        plan = NarrativePlan(ordered_clips=[e.filepath for e in tl.selected],
                             closing_quote="We should have stayed a week")
        shots = assemble.plan_shots(tl, plan, self.cfg, map_tracks=None)
        last = shots[-1]
        assert last.kind == "closing"
        assert last.caption == "We should have stayed a week"
        assert "2 days" in last.location and "2 places" in last.location

    def test_cards_can_be_switched_off(self):
        tl = self._timeline()
        cfg = assemble.AssembleConfig(day_cards=False, closing_card=False)
        shots = assemble.plan_shots(tl, NarrativePlan(
            ordered_clips=[e.filepath for e in tl.selected]), cfg, map_tracks=None)
        assert all(s.kind == "clip" for s in shots)
        assert len(shots) == len(tl.selected)

    def test_trip_summary_adds_distance_when_there_is_a_route(self):
        tl = self._timeline()
        coords = [(47.2, 11.5), (47.3, 11.5)]      # ~11 km apart
        assert "km" in assemble.trip_summary(tl, coords)
        assert "km" not in assemble.trip_summary(tl, None)


class TestEntranceAnimation:
    def test_picture_rises_then_drifts(self, tmp_path):
        import numpy as np
        from PIL import Image

        path = tmp_path / "shot.jpg"
        Image.new("RGB", (1200, 900), (40, 120, 200)).save(path)
        e = TimelineEntry(filepath=str(path), type="photo")
        e.width, e.height = 1200, 900
        cfg = assemble.AssembleConfig(map_mode="none", resolution=(640, 360))
        clip = assemble.build_clip(e, 3.0, (640, 360), cfg, lambda layout: [], animate=True)

        def card_width(t):
            frame = np.asarray(clip.get_frame(t), dtype=float).mean(axis=2)
            edge = np.abs(np.diff(frame, axis=1)).sum(axis=0)
            cols = np.where(edge > edge.max() * 0.25)[0]
            return cols.max() - cols.min()

        entering, settled, leaving = card_width(0.0), card_width(1.5), card_width(2.95)
        assert entering < settled          # rises from behind
        assert leaving > settled           # drifts past on the way out
        assert clip.duration == pytest.approx(3.0)

    def test_animation_can_be_switched_off(self, tmp_path):
        from PIL import Image
        path = tmp_path / "shot.jpg"
        Image.new("RGB", (1200, 900), (40, 120, 200)).save(path)
        e = TimelineEntry(filepath=str(path), type="photo")
        e.width, e.height = 1200, 900
        cfg = assemble.AssembleConfig(map_mode="none", resolution=(320, 180))
        still = assemble.build_clip(e, 2.0, (320, 180), cfg, lambda layout: [], animate=False)
        import numpy as np
        assert np.array_equal(still.get_frame(0.1), still.get_frame(1.5))   # nothing moves


class TestFlyingMapSurvivesFlattening:
    """A still photo is flattened to one frame - the moving map must not be
    flattened away with it."""

    def test_animated_map_is_composited_over_a_photo(self, tmp_path):
        import numpy as np
        from PIL import Image
        import map_render

        photo = tmp_path / "shot.jpg"
        Image.new("RGB", (1200, 900), (20, 20, 20)).save(photo)
        frames = []
        for i in range(4):
            path = tmp_path / f"m{i}.png"
            Image.new("RGBA", (600, 600), (255, 80, 0, 255)).save(path)
            frames.append(path)
        track = map_render.MapTrack(frames, 12.0, True, [1.0, 1.2, 1.4, 1.0])

        e = TimelineEntry(filepath=str(photo), type="photo")
        e.width, e.height = 1200, 900
        cfg = assemble.AssembleConfig(resolution=(640, 360), map_mode="overlay")
        clip = assemble.build_clip(
            e, 2.0, (640, 360), cfg, lambda layout: [],
            overlay_clips_for=lambda layout: [
                assemble.map_animation_clip(track, layout, cfg, 2.0)],
            animate=False)
        frame = np.asarray(clip.get_frame(0.5), dtype=int)
        layout = assemble.layout_for((1200, 900), (640, 360), cfg)
        patch = frame[layout.ui_top + 5:layout.ui_top + 30, layout.ui_left + 5:layout.ui_left + 30]
        assert patch[..., 0].mean() > 150 and patch[..., 2].mean() < 90   # the orange map is there


class TestTextNeverOverflows:
    cfg = assemble.AssembleConfig()

    def _draw(self):
        from PIL import Image, ImageDraw
        return ImageDraw.Draw(Image.new("RGBA", (8, 8)))

    def test_long_text_shrinks_instead_of_being_cut(self):
        draw = self._draw()
        text = "Everything uphill obviously and then quite a lot more uphill after that"
        big, big_lines = assemble.fit_text(text, 600, 3, draw, 60, 20)
        assert len(big_lines) <= 3
        assert " ".join(big_lines) == text            # nothing lost
        assert all(draw.textlength(line, font=big) <= 600 for line in big_lines)

    def test_impossible_text_is_ellipsized_not_overflowed(self):
        draw = self._draw()
        text = "Supercalifragilistic" * 12
        font, lines = assemble.fit_text(text, 200, 2, draw, 40, 18)
        assert lines and all(draw.textlength(line, font=font) <= 200 for line in lines)
        assert lines[-1].endswith("…")

    def test_short_text_keeps_its_size(self):
        draw = self._draw()
        font, lines = assemble.fit_text("Short one", 900, 3, draw, 60, 20)
        assert lines == ["Short one"] and font.size == 60

    @pytest.mark.parametrize("title", [
        "Short one",
        "Everything uphill, obviously",
        "Everything uphill obviously and then quite a lot more uphill after that",
    ])
    def test_day_card_stack_stays_inside_the_frame(self, title):
        shot = assemble.Shot(entry=entry("a.jpg"), duration=4.0, caption=title,
                             location="Fri 27 Feb · Hopfgarten im Brixental",
                             eyebrow="DAY 3", kind="day")
        frame = (1920, 1080)
        margin = round(self.cfg.frame_margin * 1920)
        sizes = [(4000, 3000), (3024, 4032), (4000, 3000)]
        probe = assemble.day_card_layout(sizes, frame, self.cfg)
        height = assemble.day_card_text_height(shot, probe["text_width"], frame, self.cfg)
        geometry = assemble.day_card_layout(sizes, frame, self.cfg, height)

        row_x, row_y, row_w, row_h = geometry["row"]
        map_x, map_y, map_w, _ = geometry["map"]
        assert row_y >= margin and row_x >= margin
        assert row_x + row_w <= 1920 - margin
        assert map_y + map_w <= 1080 - margin          # the map still fits under the text
        assert map_y >= geometry["text_top"] + height  # and never lands on it

    def test_day_card_text_is_centred(self):
        shot = assemble.Shot(entry=entry("a.jpg"), duration=4.0, eyebrow="DAY 2",
                             caption="Everything uphill, obviously",
                             location="Thu 26 Feb", kind="day")
        frame = (1280, 720)
        geometry = assemble.day_card_layout([(4000, 3000)], frame, self.cfg)
        overlay = assemble.day_card_text(shot, geometry, frame, self.cfg)
        pixels = overlay.load()
        inked = [x for x in range(1280)
                 if any(pixels[x, y][3] > 0 for y in range(geometry["text_top"], 720))]
        left, right = min(inked), max(inked)
        assert abs((left + right) / 2 - 640) < 12       # balanced about the centre line


class TestClosingCard:
    cfg = assemble.AssembleConfig()

    def test_map_is_the_hero_and_corners_hold_photos(self, tmp_path):
        import numpy as np
        from PIL import Image

        photos = []
        for i, colour in enumerate([(220, 40, 40), (40, 220, 40), (40, 40, 220), (220, 220, 40)]):
            path = tmp_path / f"c{i}.jpg"
            Image.new("RGB", (1200, 900), colour).save(path)
            e = TimelineEntry(filepath=str(path), type="photo")
            e.width, e.height = 1200, 900
            photos.append(e)
        map_path = tmp_path / "map.png"
        Image.new("RGBA", (700, 700), (250, 250, 250, 255)).save(map_path)

        shot = assemble.Shot(entry=photos[0], duration=6.0, kind="closing",
                             caption="Austria doesn't ask for much, just your knees",
                             location="6 days · 6 places · 177 km",
                             map_still=map_path, gallery=photos)
        clip = assemble.build_closing_card(shot, (1280, 720), self.cfg, animate=False)
        frame = np.asarray(clip.get_frame(1.0), dtype=int)

        centre = frame[150:420, 500:780]
        assert centre.mean() > 180                     # the big pale map dominates the middle
        margin = round(self.cfg.frame_margin * 1280)
        for x, y in [(margin + 20, margin + 20), (1280 - margin - 20, margin + 20),
                     (margin + 20, 720 - margin - 20), (1280 - margin - 20, 720 - margin - 20)]:
            patch = frame[y - 8:y + 8, x - 8:x + 8]
            assert patch.std() > 0 or patch.mean() > 20   # a photo sits in every corner

    def test_quote_and_stats_are_drawn_and_fit(self, tmp_path):
        import numpy as np
        from PIL import Image
        path = tmp_path / "a.jpg"
        Image.new("RGB", (1200, 900), (10, 10, 10)).save(path)
        e = TimelineEntry(filepath=str(path), type="photo")
        e.width, e.height = 1200, 900
        shot = assemble.Shot(entry=e, duration=6.0, kind="closing", gallery=[e],
                             caption="A closing line that runs on considerably longer than usual",
                             location="6 days · 6 places · 177 km")
        clip = assemble.build_closing_card(shot, (1280, 720), self.cfg, animate=False)
        frame = np.asarray(clip.get_frame(1.0), dtype=float).mean(axis=2)
        margin = round(self.cfg.frame_margin * 1280)
        bright = frame > 200
        cols = np.where(bright.any(axis=0))[0]
        rows = np.where(bright.any(axis=1))[0]
        assert cols.min() >= margin - 2 and cols.max() <= 1280 - margin + 2
        assert rows.max() <= 720 - margin + 2          # nothing runs off the bottom


class TestKeepAllVideos:
    """--all-videos: everything shot as video makes the cut."""

    def _clips(self):
        short = entry("brief.mp4", 0, 46.0, 7.7, kind="video", duration=0.9)
        twin_a = entry("clip.mp4", 5, 46.0, 7.7, kind="video", duration=8.0)
        twin_b = entry("clip+1.mp4", 5, 46.0, 7.7, kind="video", duration=8.0)
        photo = entry("still.jpg", 9, 46.0, 7.7)
        for e in (short, twin_a, twin_b, photo):
            e.blur_score, e.brightness = 400.0, 128.0
        return [short, twin_a, twin_b, photo]

    def test_short_videos_survive_the_quality_gate(self, tmp_path):
        path = tmp_path / "v.mp4"
        e = entry("v.mp4", kind="video", duration=0.8)
        e.filepath = str(path)
        selection.score_entry(e, selection.SelectionConfig(keep_all_videos=True))
        assert e.drop_reason != "video shorter than 2s"

        strict = entry("v.mp4", kind="video", duration=0.8)
        selection.score_entry(strict, selection.SelectionConfig())
        assert "shorter than" in strict.drop_reason

    def test_duplicate_videos_are_kept(self, monkeypatch):
        import numpy as np
        clips = self._clips()[1:3]
        monkeypatch.setattr(selection, "clip_embeddings",
                            lambda *a, **kw: np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32))
        kept = selection.deduplicate(list(clips), selection.SelectionConfig(keep_all_videos=True))
        assert len(kept) == 2
        collapsed = selection.deduplicate(list(clips), selection.SelectionConfig())
        assert len(collapsed) == 1

    def test_all_videos_can_exceed_max_clips(self):
        videos = [entry(f"v{i}.mp4", i, 46.0, 7.7, kind="video", duration=5.0) for i in range(8)]
        cfg = selection.SelectionConfig(max_clips=3, keep_all_videos=True)
        chosen = selection.balanced_select(videos, cfg)
        assert len(chosen) == 8
        capped = selection.balanced_select(videos, selection.SelectionConfig(max_clips=3))
        assert len(capped) == 3


class TestMultipleSongs:
    def test_accepts_a_file_a_list_and_a_folder(self, tmp_path):
        first, second = tmp_path / "a.mp3", tmp_path / "b.mp3"
        for path in (first, second):
            path.write_bytes(b"x")
        (tmp_path / "notes.txt").write_text("not music")
        assert assemble.music_files(str(first)) == [str(first)]
        assert assemble.music_files([str(first), str(second)]) == [str(first), str(second)]
        assert assemble.music_files([str(tmp_path)]) == [str(first), str(second)]   # name order
        assert assemble.music_files(["/nope/missing.mp3"]) == []

    def test_songs_are_levelled_and_joined(self, tmp_path):
        import subprocess
        import numpy as np

        loud, quiet = tmp_path / "loud.mp3", tmp_path / "quiet.mp3"
        for path, gain in ((loud, "0.9"), (quiet, "0.12")):
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                            "sine=frequency=330:duration=12", "-af", f"volume={gain}",
                            "-c:a", "libmp3lame", str(path)], check=True)

        levels = [mediautil.measure_audio(str(p))["mean_db"] for p in (loud, quiet)]
        assert abs(levels[0] - levels[1]) > 10          # the files really are far apart

        bed, _ = assemble.build_music_bed([str(loud), str(quiet)], 20.0, assemble.AssembleConfig())
        assert bed.duration == pytest.approx(20.0, abs=0.1)
        arr = bed.to_soundarray(fps=22050)

        def level(start, end):
            window = arr[int(start * 22050):int(end * 22050)]
            return 20 * np.log10(max(float(np.sqrt((window ** 2).mean())), 1e-9))

        assert level(2, 8) == pytest.approx(level(14, 19), abs=2.0)   # matched in the mix

    def test_a_missing_track_is_skipped_not_fatal(self, tiny_music):
        bed, _ = assemble.build_music_bed([tiny_music], 8.0, assemble.AssembleConfig())
        assert bed is not None and bed.duration == pytest.approx(8.0, abs=0.1)


class TestTargetDuration:
    cfg_base = dict(photo_duration=3.5, crossfade=0.5, day_cards=False, closing_card=False)

    def _shots(self, photos=8, videos=2):
        shots = []
        for i in range(photos):
            shots.append(assemble.Shot(entry=entry(f"p{i}.jpg", i), duration=3.5))
        for i in range(videos):
            e = entry(f"v{i}.mp4", 100 + i, kind="video", duration=6.0)
            shots.append(assemble.Shot(entry=e, duration=6.0))
        return shots

    def _length(self, shots, cfg):
        overlap = cfg.crossfade * max(0, sum(1 for s in shots[1:] if s.transition == "crossfade"))
        return sum(s.duration for s in shots) - overlap

    @pytest.mark.parametrize("target", [45.0, 60.0, 75.0])
    def test_lands_on_the_requested_length(self, target):
        cfg = assemble.AssembleConfig(target_duration=target, **self.cfg_base)
        shots = self._shots()
        assemble.fit_to_duration(shots, cfg)
        assert self._length(shots, cfg) == pytest.approx(target, abs=0.05)

    def test_videos_keep_their_own_length(self):
        cfg = assemble.AssembleConfig(target_duration=45.0, **self.cfg_base)
        shots = self._shots()
        assemble.fit_to_duration(shots, cfg)
        assert all(s.duration == 6.0 for s in shots if s.entry.is_video)

    def test_too_short_a_target_clamps_rather_than_flashing_frames(self):
        cfg = assemble.AssembleConfig(target_duration=8.0, **self.cfg_base)
        shots = self._shots()
        used = assemble.fit_to_duration(shots, cfg)
        assert used == cfg.min_photo_duration            # as short as it will go
        assert all(s.duration >= cfg.min_photo_duration for s in shots if not s.entry.is_video)

    def test_too_long_a_target_stops_at_a_watchable_still(self, caplog):
        # 8 photos cannot fill 90s without ~10s stills, so it stops at the cap
        cfg = assemble.AssembleConfig(target_duration=90.0, **self.cfg_base)
        shots = self._shots()
        with caplog.at_level("WARNING"):
            used = assemble.fit_to_duration(shots, cfg)
        assert used == cfg.max_photo_duration
        assert "cannot reach" in caplog.text
        assert self._length(shots, cfg) < 90.0

    def test_without_a_target_nothing_changes(self):
        cfg = assemble.AssembleConfig(**self.cfg_base)
        shots = self._shots()
        assert assemble.fit_to_duration(shots, cfg) == 3.5
        assert all(s.duration in (3.5, 6.0) for s in shots)

    def test_duration_parsing(self):
        import main
        assert main.parse_duration("3m30s") == 210.0
        assert main.parse_duration("3.5m") == 210.0
        assert main.parse_duration("210") == 210.0
        with pytest.raises(SystemExit):
            main.parse_duration("ages")
        with pytest.raises(SystemExit):
            main.parse_duration("2")            # under the 5 second floor


class TestMusicAnalysis:
    """Beat tracking on a click track with a known tempo."""

    @pytest.fixture(scope="class")
    @classmethod
    def click_track(cls, tmp_path_factory):
        import subprocess
        path = tmp_path_factory.mktemp("beats") / "click.wav"
        # 120 BPM: a click every 0.5 s, with a loud second half
        subprocess.run([
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "sine=frequency=1000:duration=24",
            "-f", "lavfi", "-i", "sine=frequency=2:duration=24",
            "-filter_complex",
            "[1]aeval=0.5*(1-cos(2*PI*t*2)):c=mono[env];[0][env]amultiply,"
            "volume='if(gt(t,12),1.0,0.25)':eval=frame",
            "-ar", "22050", str(path)], check=True)
        return str(path)

    def test_finds_the_tempo_and_beats(self, click_track):
        import music
        analysed = music.analyse([click_track])
        assert analysed is not None
        assert analysed.tempo == pytest.approx(120, rel=0.12)
        assert len(analysed.beats) > 20
        gaps = np.diff(analysed.beats)
        assert float(np.median(gaps)) == pytest.approx(0.5, abs=0.06)

    def test_energy_follows_the_track(self, click_track):
        import music
        analysed = music.analyse([click_track])
        assert analysed.energy_at(4.0) < analysed.energy_at(18.0)   # quiet half, loud half
        assert 0.0 <= analysed.busy_threshold() <= 1.0

    def test_analysis_is_cached_per_track(self, click_track, tmp_path):
        import music
        cache = tmp_path / "music.json"
        first = music.analyse([click_track], cache)
        assert cache.exists()
        again = music.analyse([click_track], cache)
        assert again.beats == [round(b, 3) for b in first.beats]   # cached to the millisecond
        assert music.MusicMap.load_if_matching(cache, ["/other/song.mp3"]) is None

    def test_missing_or_broken_audio_is_not_fatal(self, tmp_path):
        import music
        assert music.analyse([]) is None
        broken = tmp_path / "broken.mp3"
        broken.write_bytes(b"not audio at all")
        assert music.analyse([str(broken)]) is None

    def test_nearest_beat_respects_the_tolerance(self):
        import music
        grid = music.MusicMap(tempo=120.0, beats=[0.0, 0.5, 1.0, 1.5, 2.0])
        assert grid.nearest_beat(1.04, 0.1) == 1.0
        assert grid.nearest_beat(1.25, 0.1) is None          # too far from either
        assert grid.beat_interval == pytest.approx(0.5)

    def test_pacing_is_shorter_when_the_music_is_busy(self):
        import music
        m = music.MusicMap(tempo=120.0, beats=[0.0], energy_times=[0.0, 10.0], energy=[0.1, 0.95])
        assert music.pacing_factor(m, 0.0, 1.0) > 1.0        # quiet: linger
        assert music.pacing_factor(m, 10.0, 1.0) < 1.0       # busy: move on
        assert music.pacing_factor(m, 10.0, 0.0) == pytest.approx(1.0)   # strength 0 = off


class TestBeatSync:
    def _music(self, bpm=120.0, length=120.0):
        import music
        step = 60.0 / bpm
        beats = [round(i * step, 3) for i in range(int(length / step))]
        times = [i * 0.5 for i in range(int(length * 2))]
        return music.MusicMap(duration=length, tempo=bpm, beats=beats,
                              downbeats=beats[::4], energy_times=times,
                              energy=[0.5] * len(times))

    def _shots(self, count=10):
        return [assemble.Shot(entry=entry(f"p{i}.jpg", i), duration=3.3) for i in range(count)]

    def _cuts(self, shots, cfg):
        t, cuts = 0.0, []
        for i, shot in enumerate(shots):
            nxt = shots[i + 1] if i + 1 < len(shots) else None
            overlap = cfg.crossfade if (nxt and nxt.transition == "crossfade") else 0.0
            t += shot.duration - overlap
            cuts.append(t)
        return cuts

    def test_cuts_move_onto_the_beat(self):
        cfg = assemble.AssembleConfig(beat_sync=True, beat_sync_strength=0.0)
        shots, music_map = self._shots(), self._music()
        assemble.sync_to_music(shots, music_map, cfg)
        offsets = [min(abs(c - b) for b in music_map.beats) for c in self._cuts(shots, cfg)[:-1]]
        assert max(offsets) < 0.05

    def test_it_does_not_run_away_with_the_length(self):
        # worst case: identical shot lengths, so every snap rounds the same way
        cfg = assemble.AssembleConfig(beat_sync=True, beat_sync_strength=0.6)
        shots = self._shots(12)
        before = self._cuts(shots, cfg)[-1]
        assemble.sync_to_music(shots, self._music(), cfg)
        assert self._cuts(shots, cfg)[-1] == pytest.approx(before, rel=0.08)

    def test_transitions_are_decided_before_the_length_is_fitted(self):
        cfg = assemble.AssembleConfig(beat_sync=True, beat_sync_strength=0.0,
                                      target_duration=40.0, day_cards=False,
                                      closing_card=False)
        shots = self._shots(10)
        loud = self._music()
        loud.energy = [0.95] * len(loud.energy_times)      # every join is a busy one
        assemble.choose_transitions(shots, loud, cfg)
        assert all(s.transition == "cut" for s in shots[1:])
        assemble.fit_to_duration(shots, cfg)
        # with no crossfades there are no overlaps to subtract, and the fit knows it
        assert sum(s.duration for s in shots) == pytest.approx(40.0, abs=0.05)

    def test_videos_are_trimmed_to_a_beat_never_stretched(self):
        cfg = assemble.AssembleConfig(beat_sync=True, beat_sync_strength=0.0)
        video = entry("v.mp4", 0, kind="video", duration=7.0)
        shots = [assemble.Shot(entry=video, duration=7.0),
                 assemble.Shot(entry=entry("p.jpg", 1), duration=3.3)]
        assemble.sync_to_music(shots, self._music(), cfg)
        assert shots[0].duration <= 7.0

    def test_a_drop_becomes_a_hard_cut_with_a_punch(self):
        cfg = assemble.AssembleConfig(beat_sync=True, beat_sync_strength=0.0)
        shots = self._shots(4)
        m = self._music()
        m.drops = [self._cuts(shots, cfg)[0]]       # a drop right on the first cut
        assemble.choose_transitions(shots, m, cfg)
        assert shots[1].transition == "cut" and shots[1].punch is True

    def test_switching_it_off_changes_nothing(self):
        cfg = assemble.AssembleConfig(beat_sync=False)
        shots = self._shots()
        assert assemble.sync_to_music(shots, self._music(), cfg) == 0
        assert all(s.duration == 3.3 for s in shots)

    def test_no_music_map_is_harmless(self):
        cfg = assemble.AssembleConfig(beat_sync=True)
        shots = self._shots()
        assert assemble.sync_to_music(shots, None, cfg) == 0


class TestVideoHighlights:
    """Long videos should give up their best moments, not their first ones."""

    @pytest.fixture(scope="class")
    @classmethod
    def marked_video(cls, tmp_path_factory):
        """20 s where each second is a distinct, identifiable brightness, and
        the interesting action (movement + sound) sits at 12-16 s."""
        import subprocess
        path = tmp_path_factory.mktemp("video") / "marked.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "color=c=gray:s=320x180:d=20:r=10",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=20",
            "-filter_complex",
            # brightness encodes the second; a burst of noise at 12-16s
            "[0]geq=lum='if(between(T,12,16),128+100*sin(40*T+X/9),8*floor(T))':cb=128:cr=128[v];"
            "[1]volume='if(between(t,12,16),1.0,0.02)':eval=frame[a]",
            "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", str(path)], check=True)
        return str(path)

    def test_the_interesting_part_is_chosen(self, marked_video):
        import highlights
        cfg = highlights.HighlightConfig(budget=6.0, min_segment=3.0, max_segments=1)
        picked = highlights.highlights_for(marked_video, 20.0, cfg)
        assert len(picked) == 1
        start, end = picked[0]
        assert end - start == pytest.approx(6.0, abs=0.6)
        assert 9.0 <= start <= 14.0        # lands on the burst at 12-16s, not at 0

    def test_a_short_video_is_used_whole(self, marked_video):
        import highlights
        cfg = highlights.HighlightConfig(budget=30.0)
        assert highlights.highlights_for(marked_video, 20.0, cfg) == [(0.0, 20.0)]

    def test_speech_lifts_the_score_where_someone_talks(self, marked_video):
        import highlights
        times, plain = highlights.score_video(marked_video)
        _, spoken = highlights.score_video(marked_video, [{"start": 2.0, "end": 5.0}])
        window = (times >= 2.0) & (times <= 5.0)
        # the bonus is smoothed over ~1.5 s, so the mean rises by less than it
        assert spoken[window].mean() > plain[window].mean() + 0.1
        assert spoken[window].max() > plain[window].max() + 0.3
        outside = times > 17.0
        assert spoken[outside].mean() == pytest.approx(plain[outside].mean(), abs=0.01)

    def test_a_quiet_word_does_not_beat_a_loud_scene(self, marked_video):
        # speech helps, but it should not drag the pick off an obvious burst
        import highlights
        cfg = highlights.HighlightConfig(budget=6.0, min_segment=3.0, max_segments=1)
        picked = highlights.highlights_for(marked_video, 20.0, cfg, [{"start": 2.0, "end": 5.0}])
        assert 9.0 <= picked[0][0] <= 14.0

    def test_budget_is_respected_across_segments(self):
        import highlights
        times = np.arange(0, 60, 0.5)
        score = np.zeros_like(times)
        for peak in (5.0, 25.0, 45.0):     # three separate moments
            score[(times > peak - 2) & (times < peak + 2)] = 1.0
        cfg = highlights.HighlightConfig(budget=9.0, min_segment=3.0, max_segments=3)
        picked = highlights.pick_segments(times, score, 60.0, cfg)
        assert 1 <= len(picked) <= 3
        assert sum(e - s for s, e in picked) <= 9.0 + 1e-6
        assert picked == sorted(picked)                       # chronological
        for (_, end), (start, _) in zip(picked, picked[1:]):
            assert start >= end                               # never overlapping

    def test_assembly_plays_the_chosen_stretches(self, marked_video):
        src = __import__("moviepy").VideoFileClip(marked_video, audio=False)
        try:
            joined = assemble._joined_segments(src, [(2.0, 5.0), (14.0, 17.0)], 6.0)
            assert joined.duration == pytest.approx(6.0, abs=0.2)

            def brightness(clip, t):
                return float(np.asarray(clip.get_frame(t), dtype=float).mean())

            # first stretch plays seconds 2-5, second stretch seconds 14-17
            assert brightness(joined, 0.5) == pytest.approx(brightness(src, 2.5), abs=2.0)
            assert brightness(joined, 4.5) == pytest.approx(brightness(src, 15.5), abs=6.0)
            assert brightness(joined, 4.5) != pytest.approx(brightness(src, 4.5), abs=6.0)
        finally:
            src.close()

    def test_segments_drive_the_clip_length(self):
        e = entry("v.mp4", kind="video", duration=90.0)
        cfg = assemble.AssembleConfig(max_video_duration=15.0)
        assert assemble.clip_duration(e, cfg) == 15.0        # no scan: opening 15s
        assert assemble.video_segments(e, cfg) == [(0.0, 15.0)]

        e.extra["highlights"] = [[10.0, 15.0], [60.0, 66.0]]
        assert assemble.video_segments(e, cfg) == [(10.0, 15.0), (60.0, 66.0)]
        assert assemble.clip_duration(e, cfg) == pytest.approx(11.0)

    def test_a_cached_scan_still_respects_a_smaller_budget(self):
        e = entry("v.mp4", kind="video", duration=90.0)
        e.extra["highlights"] = [[10.0, 20.0], [60.0, 70.0]]
        cfg = assemble.AssembleConfig(max_video_duration=12.0)
        assert assemble.clip_duration(e, cfg) == pytest.approx(12.0)

    def test_highlights_can_be_switched_off(self):
        e = entry("v.mp4", kind="video", duration=90.0)
        e.extra["highlights"] = [[60.0, 70.0]]
        cfg = assemble.AssembleConfig(max_video_duration=15.0, use_highlights=False)
        assert assemble.video_segments(e, cfg) == [(0.0, 15.0)]


class TestAwkwardContainers:
    """iPhone MOVs carry data streams MoviePy's parser refuses."""

    @pytest.fixture(scope="class")
    @classmethod
    def mov_with_data_streams(cls, tmp_path_factory):
        import subprocess
        base = tmp_path_factory.mktemp("mov")
        plain, tricky = base / "plain.mov", base / "tricky.mov"
        subprocess.run(["ffmpeg", "-y", "-v", "error",
                        "-f", "lavfi", "-i", "testsrc=size=320x180:rate=15:duration=3",
                        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                        "-shortest", str(plain)], check=True)
        # add a timecode data stream, like a phone does
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(plain),
                        "-map", "0", "-c", "copy", "-timecode", "01:02:03:04",
                        str(tricky)], check=True)
        return str(plain), str(tricky)

    def test_a_normal_file_is_passed_straight_through(self, mov_with_data_streams, tmp_path):
        plain, _ = mov_with_data_streams
        assert mediautil.playable_video(plain, tmp_path) == plain    # no copy made
        assert not list(Path(tmp_path).glob("*.mp4"))

    def test_an_unopenable_file_is_remuxed_and_then_opens(self, tmp_path):
        from moviepy import VideoFileClip

        # a file MoviePy cannot parse, standing in for the phone containers
        broken = tmp_path / "broken.mov"
        broken.write_bytes(b"\x00" * 2048)
        assert mediautil.playable_video(broken, tmp_path / "remux") is None   # nothing to rescue

        import subprocess
        real = tmp_path / "real.mov"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                        "-i", "testsrc=size=160x120:rate=10:duration=2",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(real)], check=True)
        mediautil._PLAYABLE.clear()
        path = mediautil.playable_video(real, tmp_path / "remux")
        clip = VideoFileClip(path, audio=False)
        try:
            assert clip.duration == pytest.approx(2.0, abs=0.2)
        finally:
            clip.close()

    def test_the_decision_is_memoised(self, mov_with_data_streams, tmp_path):
        plain, _ = mov_with_data_streams
        mediautil._PLAYABLE.clear()
        first = mediautil.playable_video(plain, tmp_path)
        assert str(Path(plain).resolve()) in mediautil._PLAYABLE
        assert mediautil.playable_video(plain, tmp_path) == first

    def test_remuxing_keeps_orientation_and_length(self, tmp_path):
        import subprocess
        from moviepy import VideoFileClip

        portrait = tmp_path / "portrait.mov"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                        "-i", "testsrc=size=180x320:rate=10:duration=2.5",
                        "-f", "lavfi", "-i", "sine=frequency=300:duration=2.5",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                        "-shortest", str(portrait)], check=True)
        mediautil._PLAYABLE.clear()
        path = mediautil.playable_video(portrait, tmp_path / "remux")
        clip = VideoFileClip(path)
        try:
            assert clip.size == [180, 320]                  # still portrait
            assert clip.duration == pytest.approx(2.5, abs=0.2)
            assert clip.audio is not None                   # audio survived
        finally:
            clip.close()


class TestBuildClipIntegration:
    """Through `build_clip` itself, not its helpers.

    These exist because two patches to the video branch were once silently
    skipped: the helpers were tested directly and passed, while the picture
    still played the opening seconds and unreadable containers were dropped.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def marked_clip(cls, tmp_path_factory):
        """8 s where brightness encodes the second, plus data streams."""
        import subprocess
        base = tmp_path_factory.mktemp("integration")
        plain, tricky = base / "plain.mp4", base / "tricky.mov"
        subprocess.run(["ffmpeg", "-y", "-v", "error",
                        "-f", "lavfi", "-i", "color=c=gray:s=320x180:d=8:r=10",
                        "-f", "lavfi", "-i", "sine=frequency=440:duration=8",
                        "-filter_complex", "[0]geq=lum='24*floor(T)':cb=128:cr=128[v]",
                        "-map", "[v]", "-map", "1:a", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", str(plain)], check=True)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(plain), "-map", "0",
                        "-c", "copy", "-timecode", "01:00:00:00", str(tricky)], check=True)
        return str(plain), str(tricky)

    def _entry(self, path, duration=8.0):
        e = TimelineEntry(filepath=str(path), type="video")
        e.width, e.height, e.duration = 320, 180, duration
        return e

    def test_the_picture_plays_the_highlight_stretch(self, marked_clip, tmp_path):
        plain, _ = marked_clip
        e = self._entry(plain)
        e.extra["highlights"] = [[5.0, 8.0]]          # brightness ~120-168 there
        cfg = assemble.AssembleConfig(resolution=(320, 180), map_mode="none",
                                      remux_dir=tmp_path / "remux", max_video_duration=15.0)
        assert assemble.clip_duration(e, cfg) == pytest.approx(3.0)

        clip = assemble.build_clip(e, 3.0, (320, 180), cfg, lambda layout: [], animate=False)
        card = np.asarray(clip.get_frame(0.5), dtype=float)
        centre = card[70:110, 130:190].mean()

        opening = self._entry(plain)
        opening.extra["highlights"] = [[0.0, 3.0]]
        other = assemble.build_clip(opening, 3.0, (320, 180), cfg, lambda layout: [], animate=False)
        opening_centre = np.asarray(other.get_frame(0.5), dtype=float)[70:110, 130:190].mean()

        # second 5 is much brighter than second 0 in this clip
        assert centre > opening_centre + 30

    def test_a_container_moviepy_refuses_is_still_rendered(self, marked_clip, tmp_path):
        _, tricky = marked_clip
        mediautil._PLAYABLE.clear()
        e = self._entry(tricky)
        cfg = assemble.AssembleConfig(resolution=(320, 180), map_mode="none",
                                      remux_dir=tmp_path / "remux")
        clip = assemble.build_clip(e, 2.0, (320, 180), cfg, lambda layout: [], animate=False)
        assert clip.duration == pytest.approx(2.0)
        assert np.asarray(clip.get_frame(1.0)).shape == (180, 320, 3)

    def test_heic_photos_render(self, tmp_path):
        from PIL import Image
        import pillow_heif

        pillow_heif.register_heif_opener()
        path = tmp_path / "shot.heic"
        try:
            Image.new("RGB", (1200, 900), (30, 140, 90)).save(path, format="HEIF")
        except Exception:                      # no HEIF encoder available here
            pytest.skip("pillow-heif cannot write HEIC in this build")
        e = TimelineEntry(filepath=str(path), type="photo")
        e.width, e.height = 1200, 900
        cfg = assemble.AssembleConfig(resolution=(320, 180), map_mode="none")
        clip = assemble.build_clip(e, 2.0, (320, 180), cfg, lambda layout: [], animate=False)
        frame = np.asarray(clip.get_frame(1.0))
        assert frame.shape == (180, 320, 3)
        assert frame[:, :, 1].mean() > frame[:, :, 0].mean()      # the green card is there

    def test_picture_and_sound_use_the_same_stretches(self, marked_clip, tmp_path):
        plain, _ = marked_clip
        e = self._entry(plain)
        e.extra["highlights"] = [[2.0, 4.0], [6.0, 8.0]]
        cfg = assemble.AssembleConfig(resolution=(320, 180), map_mode="none",
                                      remux_dir=tmp_path / "remux")
        assert assemble.video_segments(e, cfg) == [(2.0, 4.0), (6.0, 8.0)]
        assert assemble.clip_duration(e, cfg) == pytest.approx(4.0)

        from moviepy import VideoFileClip
        source = VideoFileClip(mediautil.playable_video(plain, tmp_path / "remux"))
        try:
            audio = assemble._joined_audio(source.audio, assemble.video_segments(e, cfg), 4.0)
            clip = assemble.build_clip(e, 4.0, (320, 180), cfg, lambda layout: [], animate=False)
            assert audio.duration == pytest.approx(clip.duration, abs=0.15)
        finally:
            source.close()
