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
| 3 | `understand.py` | BLIP-2 captions every chosen clip; Whisper large-v2 transcribes video audio, each clip's loudness is measured, and long videos are scanned for their best moments |
| - | `highlights.py` | Scores a video over its length and picks the stretches worth keeping |
| 4 | `narrative.py` | Sends the annotated timeline to an LLM via OpenRouter, gets back clip order, dry/funny captions, a title for each day and the closing line |
| 5 | `map_render.py` | Reverse-geocodes place names and ground elevation, and renders a zoomed-in map per clip with 3D terrain that flies to each new location |
| - | `music.py` | Beats, tempo, energy and drops of the soundtrack, cached alongside the timeline |
| 6 | `assemble.py` | Cuts everything together: day cards, transitions and motion, captions, location labels, map overlay, a closing card and a loudness-matched, ducked soundtrack |

`models.py` holds the shared `TimelineEntry` / `NarrativePlan` types, and
`mediautil.py` the pixel-level helpers (frame grabs, keyframes, sharpness) that
phases 2, 3 and 6 all need.

Every phase reads and writes the same `timeline.json`, so each one is
independently re-runnable and expensive work is never repeated: a clip that
already has a caption is skipped on the next run. The narrative plan is cached
the same way - as long as the selected clips are unchanged, re-rendering reuses
the existing captions rather than paying for a new call and rewriting the
film's voice. `--refresh-narrative` asks for a new set.

## On screen

Every element hangs off one grid: a fixed UI column on the left, and the
picture taking everything left over.

```
 ┌──────────────────────────────────────────────┐
 │ ╭────────────╮   ╭────────────────────────╮  │  the column is sized so a
 │ │            │   │                        │  │  full-height landscape shot
 │ │  3d  map   │   │       the picture      │  │  sits flush beside the map
 │ │            │   │                        │  │
 │ ╰────────────╯   │                        │  │  two tall photos share the
 │                  │                        │  │  picture area instead of
 │ ╭────────────╮   │                        │  │  one narrow card
 │ │ Nobody     │   │                        │  │
 │ │ admits lost│   ╰────────────────────────╯  │
 │ │ ◉ Matrei · │                               │
 │ │   2,177 m  │        blurred backdrop       │
 │ ╰────────────╯                               │
 └──────────────────────────────────────────────┘
```

**The picture card.** The shot is fitted (never cropped) onto a rounded card
with a hairline border and a soft drop shadow, and the space around it is
filled with a blurred, dimmed copy of the same frame rather than flat black
bars. Videos get the same treatment: the backdrop and shadow are one static
plate, and the footage is masked to the card's rounded corners.

**Two tall photos share a clip.** A single portrait photo in a wide frame
leaves most of the picture area empty, so portraits are paired side by side.
Partners do not have to be adjacent - portraits are usually scattered between
landscape shots - so each looks a few clips ahead for the next portrait *in the
same place*, matching on the resolved place name rather than the GPS cluster
(DBSCAN splits one valley into twenty clusters, which would block almost every
pair). Only photos pair; videos keep their own clip, as do clips whose map is
flying. `--no-pair-portraits` turns it off.

**The map fills the column.** One view per clip, centred on where that clip was
taken and zoomed in far enough to read town and mountain names (OpenTopoMap by
default, so peaks come with names and heights). Elevation tiles are fetched
alongside the map tiles, turned into a hillshade and multiplied over the map,
and high ground is displaced slightly upward with a painter's ordering, so
summits stand proud of their valleys. A slight rake (`--map-tilt`, default
0.16) gives the shaded ridges enough perspective to read as height without
skewing the labels out of legibility.

When the trip moves somewhere new, the view flies there over ~1.6 s while
swelling to `--map-grow` times its size and turning nearly opaque, then shrinks
and fades back on arrival. Tiles are cached on disk, so a second run makes no
requests.

**Caption and location share one card** in the column, same radius and hairline
as the picture card, separated by a hairline rule. The caption is dry and funny
rather than descriptive (see `--amount-labels`), and the location line beneath
it shows the place name - or raw coordinates when nothing is named - plus the
elevation, from EXIF where the camera recorded it and otherwise from the same
terrain tiles the map uses. Long names step down a point size rather than
clipping, so "Neukirchen am Großvenediger · 1,487 m" still fits.

There is no title at the top: the location line already names the place.

## Chapters and the finale

Day cards and the closing card are their own compositions, not the film's
column layout - they are a different beat, and centring them says so.

```
      day card                          closing card
 ┌───────────────────────┐        ┌───────────────────────┐
 │  ┌────┐┌────┐┌────┐   │        │ ┌──┐             ┌──┐  │
 │  │    ││    ││    │   │        │ └──┘  ┌───────┐  └──┘  │
 │  └────┘└────┘└────┘   │        │       │  the  │        │
 │        DAY 3          │        │       │ whole │        │
 │  Everything uphill,   │        │       │ route │        │
 │      obviously        │        │       └───────┘        │
 │  Fri 27 Feb · Matrei  │        │  "just your knees"     │
 │        ┌────┐         │        │  6 days · 5 places ·   │
 │        └────┘         │        │ ┌──┐   177 km    ┌──┐  │
 └───────────────────────┘        │ └──┘             └──┘  │
```

**Day cards** open each calendar day: one to three photos from that day in a
band across the top, spread across the day rather than three frames of the same
hour; the day number as an accent eyebrow; the title centred and set large,
because on this card the words are the subject; the date and place beneath it;
and that day's leg of the route on a small map under the stack. Everything
stacks down one centre line, the map included - a map parked in a corner
collides with the title the moment that title needs a third line.

**The closing card** makes the whole route the hero: the complete track large
in the middle, the model's parting line beneath it, the trip's numbers
(`6 days · 5 places · 177 km`, the distance measured along the route) under an
accent rule, and up to four photos tucked into the corners to frame it.

**Nothing is ever clipped.** Text is laid out by stepping the type size down
until it genuinely fits its box, and only ellipsised if even the smallest size
cannot hold it. Wrapping alone silently drops whatever will not fit on the last
line, which is how a long day title lost its ending and `Matrei in Osttirol ·
2,177 m` lost its unit. The day card stack measures its own title first and
gives back gallery height if a long one needs the room.

**Pictures ease in and out.** Each one rises from slightly behind, settles, and
drifts past on the way out, which reads as a carousel rather than a slideshow.
Only the entrance and exit windows are actually animated - the still middle of
a photo shot stays a single flattened frame - so the motion costs about a
second of compositing per shot instead of the whole clip. `--no-animation`
holds everything still, `--no-day-cards` and `--no-closing-card` drop the
chapters and the finale.

## Audio

`--music` takes a single file, several files (played in the order you list
them), or a folder. A folder plays in **natural** filename order, so `2 - song`
comes before `10 - song` rather than after it. A playlist mixes masters from different
years, so each song is normalised on its own before the joins - otherwise one
track arrives twice as loud as the one before it - and consecutive songs
crossfade rather than butting together. The bed is then looped or trimmed to
the length of the film.

Under a clip where Whisper heard speech the bed ducks to `--music-duck`
(default 0.40, about -8 dB) rather than getting out of the way entirely: the
voice sits on top while the track stays clearly audible. `--music-duck 1`
disables ducking altogether.

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
--music PATH...           one song, several in order, or a folder   (required)
--output PATH             output .mp4                               (required)

--map-mode overlay|side|none   how the map is composited            (overlay)
--max-clips N             target clip count                         (50)
--photo-duration SEC      seconds per photo                         (3.5)
--duration LENGTH         target length, e.g. 3m30s - photo timing is fitted to it
--all-videos              keep every video in the folder, filters and dedup skipped
--prefer-people           favour shots with people in them
--people-weight 0.35      how strongly, against picture quality
--music-duck 0.40         how far the music drops under speech (1 = no ducking)
--video-scenes N          separate moments a long video may contribute   (3)
--no-highlights           use the opening seconds instead of the best bits
--max-video-duration SEC  seconds used from any one video           (15)
--blur-threshold N        Laplacian variance floor                  (80)
--llm-model NAME          OpenRouter model                          (anthropic/claude-haiku-4-5)
--interactive             review/edit each caption before rendering (off)
--skip-understand         reuse cached captions and transcripts     (off)
--skip-narrative          reuse a cached plan, else chronological   (off)
--refresh-narrative       re-ask for captions even if the cut is unchanged
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
--no-pair-portraits     never put two tall photos side by side
--no-day-cards          skip the per-day title cards
--no-closing-card       skip the quote-and-full-route finale
--no-animation          hold each picture still instead of easing it in and out
--map-tilt 0.16         slight rake, so the shaded ridges read as height
--map-grow 1.45         how far the map swells during a location change
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

## Choosing what goes in

Selection ranks on picture quality by default. `--prefer-people` adds a bonus
for shots with someone in them, weighted by `--people-weight` (default 0.35),
so a decent photo of the group beats a marginally sharper one of an empty
street - while quality still decides between two shots that both have people.
It also breaks ties between near-duplicates in favour of the one with a face in
it.

The scoring is zero-shot CLIP, which Phase 2 already loads for deduplication:
each picture is compared against captions like "a photo of people together"
versus "an empty landscape with nobody in it", and the softmax of the two gives
a probability. On a real 696-file library it found people in 454 of 652 shots,
and the picks it changed were the right ones - group selfies scoring 1.00,
food and sunsets 0.00. Turning it on took photos-with-people from 53% to 100%
of the photo picks, for a 1.7% drop in mean picture quality.

## Formats

Photos: `.jpg`, `.jpeg`, `.png`, `.heic`. Videos: `.mp4`, `.mov`. Extensions are
matched case-insensitively, so `.HEIC` and `.MOV` are fine. Anything else in the
folder - RAW `.dng`, `.gif`, the music file itself - is ignored without comment.

**Timestamps come from the filename too.** Phones name files after the moment
they were taken (`20250910_144855.jpg`, `PXL_...`, `IMG-20250910-WA0007`), and
that name survives edits and exports that strip EXIF. Metadata still wins when
it has a real date - it is what the camera recorded - but a filename beats the
file's mtime, which is only when it was last copied, and it beats a video
container's `creation_time`, which is in **UTC**. That last one matters: on a
real library, Samsung videos were landing two hours before the photos taken
beside them, scrambling the order within a day. Where both sources exist they
agree: on 357 files, 98% matched within two seconds (median drift 0 s, worst
5 s), and 31 files gained a real timestamp they did not have before.

**HEIC** is read through `pillow-heif` everywhere pixels are needed, and its
EXIF is read the same way as a JPEG's: timestamp, GPS, altitude, camera. On a
real 336-file HEIC library that is 100% of timestamps and every GPS fix the
phone actually recorded. Cameras that write empty GPS tags (Samsung stores
`NaN` coordinates when location is off) are treated as having none rather than
landing the trip at the equator.

**MOV needs a repair step.** iPhone clips carry timecode and metadata streams
beside the picture, and often a second spatial-audio track. ffmpeg reads them
without complaint, but MoviePy's info parser refuses the container outright -
on a real library that was **14 of 73 videos**, each one silently dropped from
the finished film. Before assembly, any video MoviePy cannot open is remuxed
into a clean container (picture plus the first audio track, stream-copied, no
re-encoding, about a second per clip) and cached in `<work-dir>/remux`.
Rotation, duration and audio all survive; a portrait clip stays portrait.

Everything else in the pipeline talks to ffmpeg, OpenCV or Pillow directly and
was never affected - only the final assembly needed the repair.

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

## Cutting to the music

`music.py` listens to the soundtrack once (librosa) and writes a map of it next
to the timeline: tempo and beat times, an energy envelope, and where the track
drops. Phase 6 uses that in three passes, in this order, because each depends
on the one before:

1. **Transitions.** A cut landing on a drop becomes a hard cut with a harder
   entrance; a join in a busy passage cuts rather than dissolves; quiet
   passages keep their crossfades. Busy is measured against *this* track's own
   energy distribution, not an absolute level - a mellow acoustic song never
   reaches some fixed threshold and would otherwise never get a hard cut.
   This runs first because a hard cut has no crossfade overlap, so changing one
   changes the length of the film.
2. **Length.** `fit_to_duration` now counts the overlaps that will really exist.
3. **Pacing and snapping.** Shots in busy passages run shorter and quiet ones
   longer, rescaled to keep the length; then every cut point is nudged onto the
   nearest beat - a downbeat before a title card, which wants the bar. Videos
   are only ever *shortened* to reach a beat, never stretched past the footage
   that exists.

Measured on a real render: picture changes moved from 11 of 50 on the beat to
43 of 50, and in the encoded file every detectable cut landed within 100 ms of
a beat (median 24 ms) where chance at 112 BPM is about 37%.

```
--no-beat-sync          ignore the music, cut on even timing
--beat-strength 0.7     how far shot pacing follows the energy, 0-1
```

The analysis is cached per track list, so re-renders cost nothing, and a
missing librosa or an unreadable file just turns the feature off.

## Length

**Without `--duration` the film runs exactly as long as the music**: the
playlist's length (songs minus their crossfade overlaps) becomes the target, so
the film ends with the last song instead of the music being looped or cut
mid-phrase. Measured on a real run: a 234.1 s track produced a 233.3 s film.
If the clips alone are longer than the music - videos and title cards cannot be
squeezed - it says so up front and the music loops.

`--duration 3m30s` sets the length explicitly instead. Either way it - the length is the input, and the photo timing is stretched
or squeezed to land on it. Videos keep their own length (trimming them further
would cut mid-action) and so do the title cards, so the whole adjustment lands
on the stills, clamped to between 1.6 s and 9 s. If the target cannot be
reached inside those bounds the film comes out as close as it can and says so,
rather than flashing frames past.

## Long videos

`--max-video-duration` (default 15 s) is a budget, not a trim point. A clip
longer than that is scanned over its whole length and the best stretches are
kept instead of the opening seconds - which, on a phone, is usually a pocket, a
lift queue or somebody asking whether it is recording.

The score combines movement (something is happening), sharpness (the camera has
settled), sensible exposure and loudness, and adds a large bonus for anything
Whisper heard as speech: if someone is talking, that is almost always the bit
worth keeping. Dark or blown samples are discounted whatever else they score.
The scan runs in Phase 3, right after Whisper so the transcript timings are
available, and is cached on the timeline like everything else - about half a
second per video.

A long clip can contribute up to `--video-scenes` (default 3) separate moments,
joined in the order they happen, with the clip's own audio following the same
stretches. `--no-highlights` goes back to using the opening seconds.

```
   28 s clip, 15 s budget  ->  3-18 s     (skips the dull opening)
   28 s clip,  8 s budget  ->  10-18 s    (tightens onto the best part)
   90 s clip, 15 s budget  ->  up to three 5 s moments
```

## What gets in

`--all-videos` puts every video in the folder into the cut: no minimum length,
no quality gate, no deduplication for them. Useful when the clips *are* the
trip and the stills are context - on a real ski week here, 17 of 25 videos were
otherwise dropped, 13 of them as near-duplicates of each other. Videos then
count first and photos fill whatever `--max-clips` is left, so the cut can end
up longer than that number.

## Speed and memory

Phase 6 used to render in one pass, single-threaded, holding every clip open
for the whole film. That meant ~17 output frames per second with the CPU and
GPU both mostly idle, and a process tree of 80 ffmpeg readers.

**The film renders in chunks, in parallel.** Runs of `--chunk-size` shots (8 by
default) render independently to temporary files and are joined with ffmpeg's
concat demuxer, without re-encoding. `--workers` (4) sets how many render at
once. A boundary can fall mid-crossfade: the shot that straddles it appears in
both chunks, sliced, so the dissolve completes in the first file and the second
picks up the remainder. Both chunks build the shot identically and then slice
it, so nothing is lost - splitting only at hard cuts would mean no parallelism
at all on a film that dissolves throughout.

**ffmpeg scales while it decodes.** Asking the reader for the card's size
instead of decoding a full 1080x1920 frame and shrinking it in Python measured
34 fps against 83 fps on a phone clip.

**The soundtrack is built once**, against the same schedule the chunks are laid
out from, and muxed onto the joined picture with both streams copied - so music,
crossfades and ducking are unaffected by how the picture was split.

Measured on a 110 s film, 960x540, 40 shots:

| | wall time | output rate |
|---|---|---|
| one pass (as before) | 157.5 s | 16.8 fps |
| decode-time scaling only | 125.1 s | 21.2 fps |
| + 4 workers | 59.2 s | 44.7 fps |
| + GPU encoding (NVENC) | 58.1 s | 45.6 fps |

**2.7x overall.** NVENC is worth almost nothing here, which is the useful
finding: the bottleneck is Python compositing, not encoding. It is still on by
default (`--encoder auto`) when few workers are running, because it costs
nothing and frees CPU on a busier machine.

On memory, three things changed:

- **Clip audio is read from a small extracted WAV**, not a second
  `VideoFileClip`. Reading sound through the video reader cost another
  persistent ffmpeg process per clip - with 25 videos that was 50 readers.
- **Memory is handed back between phases.** Phase 3 leaves ~5.4 GB of freed
  model memory in glibc's arenas, which Python never returns on its own;
  `malloc_trim` brings the process back to 1.0 GB. That is why an interrupted
  run that skips the models looked so much leaner than a full one.
- **Chunking bounds what is alive at once** - but `--workers 4` then holds four
  chunks, so parallelism trades memory for speed. On a 40-shot film that was
  7.5 GB at one worker against 9.1 GB at four. Lower `--workers` and
  `--chunk-size` if memory is tighter than time.

```
--workers 4             how many parts of the film render at once
--chunk-size 8          shots per part (0 renders in a single pass)
--encoder auto          auto | x264 | nvenc
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
| 6 assembly | the long pole - roughly 15-30 min |

Phase 6 dominates because MoviePy composites in Python. Photos are flattened to
a single still before rendering rather than being recomposited for each of
their ~100 identical frames — measured at 3.4x faster on a photo-only cut — so
the cost scales with how much *video* is in the edit, plus the handful of clips
where the map is flying and therefore cannot be flattened. (Reference point: 18
clips including 6 videos, 1600x1066, 91 s of output, 6.5 min to render.) To iterate faster, re-run with `--skip-select --skip-understand
--skip-narrative` and add `--resolution 1280x720 --fps 24 --preset veryfast`;
drop those flags for the final pass.
