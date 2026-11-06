# Implementation notes

## Shared code and caching

`models.py` holds the shared `TimelineEntry` / `NarrativePlan` types, and
`mediautil.py` the pixel-level helpers (frame grabs, keyframes, sharpness) that
phases 2, 3 and 6 all need.

Every phase reads and writes the same `timeline.json`, so each one is
independently re-runnable and expensive work is never repeated: a clip that
already has a caption is skipped on the next run. The narrative plan is cached
the same way - as long as the selected clips are unchanged, re-rendering reuses
the existing captions rather than paying for a new call and rewriting the
film's voice. `--refresh-narrative` asks for a new set.

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

[← Back to README](../README.md)
