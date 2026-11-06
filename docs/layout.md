# Layout and design

How the frame is laid out, in landscape and vertical cuts, and how the day cards and closing card are composed.

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

## Vertical cuts

`--vertical` renders 1080x1920 for a phone or Instagram. It is a different
layout, not the same one squeezed: a 1080-wide frame has no room for the side
column, so

- the **map becomes a small badge in the bottom-right corner** (27% of the
  width) instead of filling a column,
- the **caption takes the band beneath the picture**, on the same baseline as
  the map and stopping short of it,
- the **picture spans the full width**, with the blurred backdrop filling above
  and below a wide shot,
- **pairing flips**: in a tall frame it is *landscape* photos that leave the
  picture area half empty, so those pair and stack one above the other. Two
  portraits side by side would be slivers.

Type is sized from a reference dimension rather than raw frame height, so
captions stay the same apparent size on screen instead of shrinking to match
the narrower frame. Day cards drop to two gallery pictures and a wider text
column; the closing card gets a much bigger map and squarer corner pictures.

Landscape output is untouched by all of this - same margins, same column, same
type sizes - and there is a test pinning that.

```bash
python main.py --input ./export --music ./song.mp3 --output reel.mp4 --vertical
```

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

[← Back to README](../README.md)
