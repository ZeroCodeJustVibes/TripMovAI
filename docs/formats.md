# Input and output formats

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

[← Back to README](../README.md)
