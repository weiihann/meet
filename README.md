# meet

Record, transcribe, and summarise meetings with open-source models. Runs fully
locally on macOS — no meeting audio, transcript, or note leaves the machine.

Built for bilingual Mandarin/English meetings, which is the case most tools
handle badly, but works for either language alone.

- **Capture** — microphone and system audio as two separate tracks, via Core
  Audio process taps. No virtual audio driver to install.
- **Transcribe** — Qwen3-ASR through MLX, with your own vocabulary supplied as
  context so names and jargon come out right.
- **Summarise** — a local Qwen model writes the notes, into an Obsidian vault or
  any directory you point it at.

## Requirements

macOS 14.2 or later on Apple Silicon (Core Audio process taps and MLX),
plus [uv](https://docs.astral.sh/uv/), `ffmpeg`, `git`, and Swift (Xcode command
line tools) to build the capture binary.

```bash
brew install uv ffmpeg
```

## Install

```bash
git clone https://github.com/weiihann/meet.git
cd meet
uv sync
uv run meet setup
```

`meet setup` gets you to a working install: it creates `.env` and `glossary.txt`,
clones and builds the audio capture binary, downloads the speech detector and both
models, and installs a `meet` command at `~/.local/bin/meet` so you can run it
from any directory. It is safe to re-run, and it never overwrites a `meet` it did
not create.

**Budget about 9 GB and some patience** for the two default models
(3.8 GB recogniser, 5.5 GB summariser). They go into Hugging Face's global cache,
shared across projects, so a second checkout costs nothing.

- `--no-prefetch` skips the models; they download on first use instead.
- `--no-launcher` skips the `meet` command; use `uv run meet`.
- `MEET_BIN` installs the command somewhere other than `~/.local/bin`. If that
  directory is not on your `PATH`, setup tells you what to add.

Then grant your terminal two macOS permissions and **restart it**:

- **Microphone**
- **Screen & System Audio Recording**

Finally, check everything landed:

```bash
meet doctor
```

Notes are written to `<project>/notes` by default, so nothing is assumed about
how you keep notes. Point `MEET_NOTES` at an Obsidian vault or anywhere else.

## Usage

The usual flow is two commands:

```bash
meet start standup             # record mic + system audio as two tracks
meet stop                      # stop, transcribe, and write the note
```

`meet stop` runs the whole pipeline by default. Pass `--no-process` to just stop
and deal with it later, then pick it up with any of:

```bash
meet process <session-dir>     # transcribe + summarise in one step
meet transcribe <session-dir>  # two-track: labelled "Me" and "Them"
meet transcribe <audio-file>   # single mixed file, no speaker labels
meet summarize <transcript.md> --title "Design review" --org "Acme"

meet devices                   # list audio inputs, show which one is used
meet doctor                    # check config, models, and the microphone
meet setup                     # (re-)install models and the capture binary
```

`transcribe` accepts anything ffmpeg can read: `wav m4a mp3 mp4 mov mkv pcm`.
Video containers decode to their audio track, so screen recordings work.

Flags:

- `--glossary path/to/file` — vocabulary for this run.
- `--language en` (default) / `zh` / `auto` — the recogniser is pinned to one
  language, since auto-detection drifts mid-sentence into unrelated ones. Pick
  `zh` for Mandarin **and for mixed Mandarin/English**: Chinese decoding
  transcribes embedded English correctly, where English decoding does not. Set
  `MEET_LANGUAGE` in `.env` to change the default.
- `--engine sensevoice` — faster on paper, much worse English. Rarely worth it.
- `--summariser claude` — better notes via Claude Code, but the transcript leaves
  the device. `--summariser qwen` (default) is fully local.
- `--model <id>` — override the summarising model for one run.

## Vocabulary

Recognition of names and jargon improves markedly when the model is told what to
expect. `meet setup` copies `glossary.example.txt` to `glossary.txt`, which is
gitignored — edit it freely, one term per line.

It is worth adding colleagues' names, product names, and acronyms your team says
out loud. This matters most for names embedded in another language: a product
name spoken inside a Mandarin sentence is the case that fails without a glossary
and succeeds with one.

Point somewhere else per run with `--glossary`, or permanently with
`MEET_GLOSSARY` in `.env`.

## How it works

```
meet start ──> audiotee ──> system.pcm   (Core Audio process tap)
           └─> ffmpeg   ──> mic.pcm      (explicit avfoundation index)

transcribe ──> silero VAD ──> speech spans, silence dropped, capped at 30s
           ──> Qwen3-ASR via MLX, glossary as context
           ──> merge tracks onto one clock, join turns ──> transcript.md

summarize  ──> local Qwen via MLX (or claude -p)
           ──> <vault>/Notes/YYYYMMDD Meeting - X.md
```

Measured on an 18-minute meeting: 3m14s to transcribe, 33s to summarise locally.

## Configuration

Everything is an environment variable, resolved in this order:

1. the real environment, so `MEET_NOTES=/tmp meet stop` wins for one run
2. `.env` in the project directory
3. the built-in default

See `.env.example` for the full list with defaults. The ones most people change:

| variable | purpose |
|---|---|
| `MEET_NOTES` | where notes are written (default `<project>/notes`) |
| `MEET_LANGUAGE` | `en` (default), `zh`, or `auto` — use `zh` for mixed Mandarin/English |
| `MEET_ASR_MODEL` | the speech recogniser |
| `MEET_SUMMARY_MODEL` | the summarising model |
| `MEET_GLOSSARY` | your vocabulary file |
| `MEET_MODELS`, `MEET_RECORDINGS`, `MEET_VENDOR` | keep large files outside the checkout |

Paths default to somewhere inside the project directory, so a clone is
self-contained and two checkouts never share state.

### Switching models

Both models are Hugging Face ids, swapped with one line in `.env`:

```bash
# smaller and faster recogniser
MEET_ASR_MODEL=mlx-community/Qwen3-ASR-0.6B-bf16

# smaller and faster summariser
MEET_SUMMARY_MODEL=mlx-community/Qwen3.5-4B-MLX-4bit
```

Any Qwen3-ASR conversion works for the recogniser, and any mlx-lm compatible
chat model for the summariser. After changing either, run `meet setup` again to
download it.

While models download you may see `Warning: You are sending unauthenticated
requests to the HF Hub`. It is a warning, not an error — the models are public
and download fine without a token. Setting `HF_TOKEN` in `.env` silences it and
lifts the anonymous rate limit.

## Permissions and microphone gotchas

macOS signals a denied audio permission by delivering **exact zeros** rather than
an error, so `meet doctor` records a short probe and checks whether any audio
arrived, and `meet transcribe` warns rather than producing a transcript quietly
missing one side of the conversation.

The check is for literal zeros, not for a quiet track: a working Bluetooth mic in
a silent room peaks around -60 dB, which is inaudible but not zero, so an
amplitude threshold would report working hardware as broken.

**The built-in mic is unavailable with the lid closed.** In clamshell mode the
MacBook microphone returns digital silence — confirmed through both ffmpeg and
PortAudio, so it is OS behaviour rather than a bug here. Wear a headset or attach
an external mic.

**Do not change audio devices mid-recording.** ffmpeg holds one fixed
avfoundation index for the session and audiotee taps only the default output
device, so disconnecting a headset part-way through truncates the microphone
track while system audio keeps recording. `meet stop` compares the two track
lengths and warns when they diverge, but the lost audio is not recoverable.

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run ty check src
```

## Licence

MIT — see [LICENSE](LICENSE).

Third-party components have their own terms, and `meet setup` downloads some of
them:

- [audiotee](https://github.com/makeusabrew/audiotee) — MIT
- Qwen3-ASR weights — Apache 2.0
- SenseVoiceSmall weights (optional) — FunASR Model Open Source License, **not**
  MIT; check it before commercial use
- silero-vad — MIT
