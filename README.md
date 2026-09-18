# Trip Review Video Generator

One command turns a folder of exported trip photos and videos into a finished
recap video — selection, captioning, narration, route map and edit, all
automatic and all local except one small LLM call.

```bash
python main.py --input ./immich_export/ --music ./song.mp3 --output trip.mp4
```

No prompts, no decisions. `--interactive` re-enables caption editing if you
want it.

```bash
python main.py --input ./export --music ./song.mp3 --output trip.mp4 \
  --resolution 1920x1080 --amount-labels 25%
```

## How it works

| Phase | Module | What it does |
|---|---|---|
| 1 | `ingest.py` | Scans recursively, reads EXIF/GPS via exiftool, Immich/Takeout `.json` sidecars and ffprobe, sorts chronologically → `timeline.json` |
| 2 | `selection.py` | Drops blurry/under/over-exposed shots and sub-2s videos, removes near-duplicates with CLIP, then picks a place- and time-balanced set |
| 3 | `understand.py` | BLIP-2 captions every chosen clip; Whisper large-v2 transcribes video audio and each clip's loudness is measured |
| 4 | `narrative.py` | Sends the annotated timeline to an LLM via OpenRouter, gets back clip order, dry/funny captions, transitions and chapters |
| 5 | `map_render.py` | Reverse-geocodes place names and ground elevation, and renders a zoomed-in map per clip with 3D terrain that flies to each new location |
| 6 | `assemble.py` | Cuts everything together with transitions, captions, location labels, map overlay and a loudness-matched, ducked soundtrack |

`models.py` holds the shared `TimelineEntry` / `NarrativePlan` types, and
`mediautil.py` the pixel-level helpers (frame grabs, keyframes, sharpness) that
phases 2, 3 and 6 all need.

Every phase reads and writes the same `timeline.json`, so each one is
independently re-runnable and expensive work is never repeated: a clip that
already has a caption is skipped on the next run.

## On screen

```
 ┌────────────────────────────────────────────────┐
 │ ╭────╮                                         │  route map, small in the
 │ │ ▲▲ │  <- parked: 14% of frame width          │  top-left corner, 3d terrain
 │ ╰────╯                                         │
 │ ╭──────────────╮                               │  and briefly 2.2x that,
 │ │   ▲▲  ▲      │  <- travelling: swells and    │  near-solid, while the view
 │ │      ▲▲▲     │     turns near-solid, then    │  flies to a new place
 │ ╰──────────────╯     shrinks back              │
 │ ┌──────────────────────┐                       │
 │ │ Nobody admits lost   │                       │  caption, bottom left
 │ └──────────────────────┘                       │
 │ ◉ Interlaken · 568 m                           │  location label under it
 └────────────────────────────────────────────────┘
```

There is no title at the top: the location line already names the place, and a
second label competed with the footage.

**The map** is one view per clip, centred on where that clip was taken and
zoomed in far enough to read town and mountain names (OpenTopoMap by default,
so peaks come with names and heights). It stays flat and north-up; the 3D comes
from the terrain itself. Elevation tiles are fetched alongside the map tiles,
turned into a hillshade and multiplied over the map, and high ground is also
displaced slightly upward with a painter's ordering, so summits stand proud of
their valleys and occlude the ground behind them.

It parks small in the top-left corner, out of the way of the footage. When the
trip moves somewhere new, the view flies there over ~1.6 s while swelling to
`--map-grow` times its size and turning nearly opaque, then shrinks and fades
back on arrival. It grows from its own corner, so the space it occupies never
jumps. Tiles are cached on disk, so a second run makes no requests.

`--map-tilt` can still rake the whole plane back into perspective, but it is
off by default: raking the map tilts the labels too, while shading the terrain
keeps them flat and readable.

**The caption** sits bottom left, and the narrative model is asked for dry,
funny lines rather than descriptions - partly for tone, partly because BLIP-2's
captions are often wrong. The prompt tells the model to treat them as an
unreliable hint and never to assert a specific object or landmark unless the
place name or transcript backs it up, so a joke that lands does not depend on
the image model having been right. `--amount-labels` (default `25%`) sets how
many clips get one; it is a floor, and if the model under-delivers, the
shortfall is requested once more in a follow-up call.

**The location label** sits under the caption with a drawn pin (the bundled
fonts have no emoji glyphs, so a marker character would render as a blank box).
It shows the place name, or raw coordinates when nothing is named, plus the
elevation - taken from EXIF where the camera recorded it, and otherwise read
from the same terrain tiles the map already uses. The name is the one resolved
for the whole GPS cluster, so neighbouring clips do not flicker between
"Matten bei Interlaken" and "Interlaken".

## Audio

Clips arrive at wildly different levels - a real export here ranged from
-25 dBFS to -64 dBFS - so every clip's loudness is measured with one ffmpeg
pass (cached on the timeline) and normalised to `--audio-target`, default
-18 dBFS, before its role volume is applied. Two guards keep that honest: the
peak-limited gain is computed on the *combined* figure, so a clip that is quiet
on average but has loud transients is lifted only as far as its peaks allow and
never clips; and anything below -45 dBFS is treated as silence rather than
quiet, because boosting it 40 dB only amplifies hiss. The music bed is
normalised the same way to sit a few dB under, then ducked to 15% under any
clip where Whisper heard speech. `--no-normalize-audio` turns it all off.

## Setup

```bash
# system dependencies
sudo apt install ffmpeg libimage-exiftool-perl        # Ubuntu/Debian
brew install ffmpeg exiftool                          # macOS

# python environment (uv)
uv venv --python 3.11 .venv
source .venv/bin/activate

# PyTorch first, matched to your CUDA version
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

uv pip install -r requirements.txt

export OPENROUTER_API_KEY=sk-or-...      # only needed for Phase 4
```

Instead of exporting it every time, you can drop the key in a `.env` file next
to `main.py` (git-ignored); real environment variables still take precedence:

```
OPENROUTER_API_KEY=sk-or-...
```

`torch >= 2.6` is required: current `transformers` refuses to load
`torch.load`-format checkpoints on anything older.

exiftool is optional — without it the pipeline falls back to ffprobe and
Pillow, which covers JPEG/PNG/MP4/MOV but is weaker on HEIC and on video GPS.

## CLI

```
--input PATH              input folder, searched recursively        (required)
--music PATH              background music file                     (required)
--output PATH             output .mp4                               (required)

--map-mode overlay|side|none   how the map is composited            (overlay)
--max-clips N             target clip count                         (50)
--photo-duration SEC      seconds per photo                         (3.5)
--max-video-duration SEC  longest any single video clip             (12)
--blur-threshold N        Laplacian variance floor                  (80)
--llm-model NAME          OpenRouter model                          (anthropic/claude-haiku-4-5)
--interactive             review/edit each caption before rendering (off)
--skip-understand         reuse cached captions and transcripts     (off)
--skip-narrative          reuse a cached plan, else chronological   (off)
--no-map                  disable the map entirely                  (off)
```

Extras beyond the brief: `--skip-select`, `--work-dir`, `--timeline`,
`--device`, `--whisper-model`, `--blip-model`, `--fps`, `--preset`,
`--threads`, `--resolution`, `--dry-run`, `-v`, plus the map
controls:

```
--map-zoom 12           higher is closer; 12-13 keeps place names readable
--map-tiles topo|osm    topo = hillshading and peak names, osm = plain streets
--map-relief 1.0        3d terrain shading and height; 0 for plain flat tiles
--no-location           hide the location line under the caption
--map-tilt 0.0          rake the whole plane into perspective; off by default
--map-grow 2.2          how far the map swells during a location change
--no-map-animation      hold each map still instead of flying between places
--no-geocode            skip Nominatim and elevation lookups
--audio-target -18      dBFS each clip's own audio is normalised to
--no-normalize-audio    leave clip volumes exactly as recorded
```

Re-render quickly after a first full run:

```bash
python main.py --input ./export --music song.mp3 --output trip.mp4 \
  --skip-select --skip-understand --skip-narrative --photo-duration 3.0
```

## Notes on the implementation

**`selection.py`, not `select.py`.** CPython ships a builtin module named
`select`, and builtins outrank `sys.path`, so a file called `select.py` next to
`main.py` can never be imported — `import select` silently returns the stdlib
module instead.

**VRAM.** The target is a 10 GB RTX 3080, and the two big models are loaded one
at a time. BLIP-2 `opt-2.7b` is ~7.4 GB in fp16, which does not fit next to a
desktop session, so the loader checks free VRAM first and drops to 8-bit
(~4.5 GB) or CPU offload as needed. Whisper `large-v2` is loaded on CPU and cast
to fp16 before moving to the GPU (6.2 GB → 3.1 GB); its LayerNorms are left in
fp32 because Whisper's own `LayerNorm` casts activations to float and would
otherwise raise a dtype error.

**Deduplication** clusters against cluster representatives rather than by
transitive similarity: a slow pan is a chain of near-identical neighbours, and
transitive merging would collapse the whole pan into one frame.

**The map basemap is fetched once**, not per clip. Every frame shares the same
extent, so the tiles are identical — re-rendering per clip would hammer the OSM
tile servers for no benefit.

**Hillshade aspect has a sign trap.** A north-up raster needs the ESRI aspect
convention (`atan2(dy, -dx)`); get it backwards and the sun effectively lights
from the south-east, at which point every peak reads as a crater. There is a
test pinning the north-west flank brighter than the south-east one.

**Failure is never fatal.** A corrupt file, a missing API key, an unparseable
LLM reply, sparse GPS or a dead tile server each degrade that one feature and
let the render proceed.

## Tests

```bash
cd trip_video && pytest tests/ -q      # 39 tests, no GPU or network needed
```

One of them does a real (tiny) render; the rest are pure logic. They cover
timestamp/GPS parsing, the selection quotas and group cap, GPS
interpolation, the audio ducking envelope, caption layout, and the narrative
JSON contract (including markdown-wrapped replies, omitted clips and the
retry-then-fallback path) against a mock OpenRouter server.

## Output

H.264 + AAC `.mp4` at the majority resolution of the selected files; clips that
do not match are letterboxed rather than cropped.

Phone libraries are usually a mix of landscape photos and portrait video, and
no single frame suits both. Auto-detection then lands on the photos' 4:3 at
full sensor size, which is both an odd shape for video and very slow to render,
so pass the frame you actually want:

```bash
--resolution 1920x1080   # landscape recap: photos fill, portrait video pillarboxed
--resolution 1080x1920   # phone/Reels cut: portrait video fills, photos banded
```

## Runtime

For a ~50 clip, 3-4 minute film from a few hundred input files, on an RTX 3080:

| Phase | Roughly |
|---|---|
| 1 ingest | seconds |
| 2 selection | ~1 min (CLIP over every surviving file) |
| 3 understanding | ~3 s per photo, plus Whisper at several times real time |
| 4 narrative | one API call, a few seconds, about $0.01 |
| 5 map | seconds |
| 6 assembly | the long pole - roughly 10-25 min |

Phase 6 dominates because MoviePy composites in Python. Photos are flattened to
a single still before rendering rather than being recomposited for each of
their ~100 identical frames — measured at 3.4x faster on a photo-only cut — so
the cost scales with how much *video* is in the edit, plus the handful of clips
where the map is flying and therefore cannot be flattened. (Reference point: 18
clips including 6 videos, 1600x1066, 91 s of output, 6.5 min to render.) To iterate faster, re-run with `--skip-select --skip-understand
--skip-narrative` and add `--resolution 1280x720 --fps 24 --preset veryfast`;
drop those flags for the final pass.
