# Audio and music

How the soundtrack is built, how clip audio is levelled and ducked, and how cuts follow the music.

## Audio

`--music` takes a single file, several files (played in the order you list
them), or a folder. A folder plays in **natural** filename order, so `2 - song`
comes before `10 - song` rather than after it. A playlist mixes masters from different
years, so each song is normalised on its own before the joins - otherwise one
track arrives twice as loud as the one before it - and consecutive songs
crossfade rather than butting together. The bed is then looped or trimmed to
the length of the film.

A clip's own sound follows the talking rather than the clip: full level while
someone is speaking, down to `ambient_volume` when nobody is. Whisper's segment
times are mapped through the highlight stretches the clip actually plays, so
the lift lands on the right moment. Holding a whole clip at full level for the
sake of three spoken seconds just raises wind and lift noise over the music.

Under a clip where Whisper heard speech the bed ducks to `--music-duck`
(default 0.40, about -8 dB) rather than getting out of the way entirely: the
voice sits on top while the track stays clearly audible. The duck now covers
only the seconds where talking actually happens, not the whole clip. `--music-duck 1`
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

[← Back to README](../README.md)
