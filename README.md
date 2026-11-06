# Trip Review Video Generator

One command turns a folder of exported trip photos and videos into a finished
recap video — selection, captioning, narration, route map and edit, all
automatic and all local except one small LLM call.

```bash
python main.py --input ./immich_export/ --music ./song.mp3 --output trip.mp4
```

## Features

- **Automatic selection** – drops blurry, badly exposed and duplicate shots,
  then picks a set balanced across places and days.
- **Captions with a voice** – BLIP-2 describes each clip, Whisper transcribes
  speech, and an LLM writes dry, funny captions and a title for each day.
- **3D route map** – hillshaded terrain that flies to each new location, with
  place names and elevation.
- **Cut to the music** – transitions and cut points follow the beats and drops
  of your soundtrack; the film ends with the last song.
- **Smart audio** – every clip is loudness-matched, and the music ducks only
  while someone is talking.
- **Best moments of long videos** – long clips are scanned for the stretches
  worth keeping instead of using the opening seconds.
- **Landscape or vertical** – `--vertical` renders a dedicated 1080x1920 phone
  layout.
- **Re-runnable** – every phase caches its results, so a re-render repeats no
  expensive work.

## Setup

```bash
# system dependencies
sudo apt install ffmpeg libimage-exiftool-perl        # Ubuntu/Debian
brew install ffmpeg exiftool                          # macOS

# python environment (uv)
uv venv --python 3.11 .venv
source .venv/bin/activate

# PyTorch first, matched to your CUDA version (torch >= 2.6 required)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

uv pip install -r requirements.txt
```

The narrative step (Phase 4) needs an [OpenRouter](https://openrouter.ai) API
key. Either export it, or copy `.env.example` to `.env` and fill it in (`.env`
is git-ignored; real environment variables take precedence):

```bash
cp .env.example .env
```

exiftool is optional — without it the pipeline falls back to ffprobe and
Pillow, which covers JPEG/PNG/MP4/MOV but is weaker on HEIC and on video GPS.

## Usage

```bash
# landscape recap, captions on a quarter of the clips
python main.py --input ./export --music ./song.mp3 --output trip.mp4 \
  --resolution 1920x1080 --amount-labels 25%

# phone / Instagram cut
python main.py --input ./export --music ./song.mp3 --output reel.mp4 --vertical

# fast re-render after a first full run
python main.py --input ./export --music song.mp3 --output trip.mp4 \
  --skip-select --skip-understand --skip-narrative
```

No prompts, no decisions. `--interactive` re-enables caption editing if you
want it. `--music` takes one file, several files, or a folder. See the
[CLI reference](docs/cli.md) for all options.

## How it works

| Phase | Module | What it does |
|---|---|---|
| 1 | `ingest.py` | Scans recursively, reads EXIF/GPS via exiftool, Immich/Takeout `.json` sidecars and ffprobe, sorts chronologically → `timeline.json` |
| 2 | `selection.py` | Drops blurry/under/over-exposed shots and sub-2s videos, removes near-duplicates with CLIP, then picks a place- and time-balanced set |
| 3 | `understand.py` | BLIP-2 captions every chosen clip; Whisper large-v2 transcribes video audio, each clip's loudness is measured, and long videos are scanned for their best moments |
| - | `highlights.py` | Scores a video over its length and picks the stretches worth keeping |
| - | `nudity.py` | Optional: detects and blurs explicit regions (`--blur-nudity`) |
| 4 | `narrative.py` | Sends the annotated timeline to an LLM via OpenRouter, gets back clip order, dry/funny captions, a title for each day and the closing line |
| 5 | `map_render.py` | Reverse-geocodes place names and ground elevation, and renders a zoomed-in map per clip with 3D terrain that flies to each new location |
| - | `music.py` | Beats, tempo, energy and drops of the soundtrack, cached alongside the timeline |
| 6 | `assemble.py` | Cuts everything together: day cards, transitions and motion, captions, location labels, map overlay, a closing card and a loudness-matched, ducked soundtrack |

A ~50 clip, 3-4 minute film takes roughly 20-35 minutes on an RTX 3080, most
of it in the final render. The LLM call costs about $0.01.

## Documentation

- [CLI reference](docs/cli.md) – every option
- [Layout and design](docs/layout.md) – the on-screen grid, vertical cuts, day cards and closing card
- [Selection and editing](docs/editing.md) – what goes in, film length, long videos, blurring
- [Audio and music](docs/audio.md) – normalisation, ducking, cutting to the beat
- [Input and output formats](docs/formats.md) – HEIC, MOV, timestamps, output resolution
- [Performance](docs/performance.md) – runtime per phase, parallel rendering, memory
- [Implementation notes](docs/implementation.md) – caching, VRAM, design decisions

## Tests

```bash
pytest tests/ -q      # no GPU or network needed
```

One test does a real (tiny) render; the rest are pure logic, including the
narrative JSON contract against a mock OpenRouter server.

## License

Copyright (C) 2026 Br0ver

This project is licensed under the [GNU General Public License v3.0](LICENSE).
You may use, copy, modify and redistribute it, but any distributed version -
modified or not - must stay open source under the same license and keep the
copyright notice above.

If you use or build on this project, please credit it, e.g. "Based on Trip
Review Video Generator by Br0ver", with a link to this repository. See
[`CITATION.cff`](CITATION.cff) for a citation entry.
