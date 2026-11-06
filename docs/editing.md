# Selection and editing

What goes into the film, how long it runs, and how long videos are trimmed.

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

## What gets in

`--all-videos` puts every video in the folder into the cut: no minimum length,
no quality gate, no deduplication for them. Useful when the clips *are* the
trip and the stills are context - on a real ski week here, 17 of 25 videos were
otherwise dropped, 13 of them as near-duplicates of each other. Videos then
count first and photos fill whatever `--max-clips` is left, so the cut can end
up longer than that number.

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

## Length

**Without `--duration` the film runs exactly as long as the music**: the
playlist's length (songs minus their crossfade overlaps) becomes the target, so
the film ends with the last song instead of the music being looped or cut
mid-phrase. Measured on a real run: a 234.1 s track produced a 233.3 s film.
If the clips alone are longer than the music - videos and title cards cannot be
squeezed - it says so up front and the music loops.

`--duration 3m30s` sets the length explicitly instead. Either way the length is the input, and the photo timing is stretched
or squeezed to land on it. Videos keep their own length (trimming them further
would cut mid-action) and so do the title cards, so the whole adjustment lands
on the stills, clamped to between 1.6 s and 9 s. If the target cannot be
reached inside those bounds the film comes out as close as it can and says so,
rather than flashing frames past.

## Blurring

`--blur-nudity` detects explicit regions with NudeNet during Phase 3 and blurs
them when the film is rendered. Detection is cached on the timeline as
*normalised* boxes, so it runs once per clip and stays correct at any output
resolution, and nothing is ever written back to your originals.

Only genuinely exposed anatomy is acted on. NudeNet also reports faces, feet,
bellies and everything "covered"; blurring those would ruin half a beach
holiday, so they are ignored. A still gets just its flagged regions blurred; a
video gets the whole picture blurred for the clip, because a box from one
keyframe will not cover a moving subject. `--nudity-threshold` (0.40) sets how
sure the detector has to be. The flag is off by default - a false positive
costs you a photo - and without `nudenet` installed it simply does nothing.

[← Back to README](../README.md)
