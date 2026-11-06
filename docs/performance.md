# Performance

Runtime per phase, and how rendering was made faster and leaner.

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
clips including 6 videos, 1600x1066, 91 s of output, 6.5 min to render.)

To iterate faster, re-run with `--skip-select --skip-understand
--skip-narrative` and add `--resolution 1280x720 --fps 24 --preset veryfast`;
drop those flags for the final pass.

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

[← Back to README](../README.md)
