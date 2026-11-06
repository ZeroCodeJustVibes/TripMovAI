# CLI reference

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

Options described in the other pages:

```
--vertical              1080x1920 phone layout                  (layout.md)
--blur-nudity           blur explicit regions (needs nudenet)    (editing.md)
--nudity-threshold 0.40 how sure the detector has to be          (editing.md)
--no-beat-sync          ignore the music, cut on even timing     (audio.md)
--beat-strength 0.7     how far shot pacing follows the energy   (audio.md)
--workers 4             how many parts of the film render at once (performance.md)
--chunk-size 8          shots per part (0 = single pass)         (performance.md)
--encoder auto          auto | x264 | nvenc                      (performance.md)
```

Run `python main.py --help` for the authoritative list.

Re-render quickly after a first full run:

```bash
python main.py --input ./export --music song.mp3 --output trip.mp4 \
  --skip-select --skip-understand --skip-narrative --photo-duration 3.0
```

[← Back to README](../README.md)
