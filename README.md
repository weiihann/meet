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

`meet setup` creates `.env` and `glossary.txt`, clones and builds the audio
capture binary, and downloads the speech detector (2 MB). It is safe to re-run.

Then grant your terminal two macOS permissions and **restart it**:

- **Microphone**
- **Screen & System Audio Recording**

Finally, check everything landed:

```bash
uv run meet doctor
```

To type `meet` instead of `uv run meet`, drop a launcher on your PATH:

```bash
printf '#!/usr/bin/env bash\nexec %s/.venv/bin/python -m meet.cli "$@"\n' "$PWD" \
  > ~/.local/bin/meet && chmod +x ~/.local/bin/meet
```

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
- `--language zh` (default) / `en` / `auto` — see below.
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

## Speaker labels

There are two levels of attribution, and only one of them is trustworthy.

**`meet start` recordings get real labels.** The microphone track is you, the
system-audio track is everyone else, so `Me` and `Them` follow from *where the
audio was captured* rather than from a model's guess. This is the reason to
record through `meet` rather than through a screen recorder.

**Imported mixed recordings get none.** Output is timestamped utterances with no
speaker prefix. Speaker diarisation was implemented and then removed: on a real
18-minute two-person call it reported 38 speakers at one clustering threshold and
13 at another. Tuning could not fix it, because the threshold that recovered
exactly 4 speakers on a clean reference recording was the same threshold that
invented 39 on real audio. Labels that confident and that wrong are worse than no
labels, since they invite you to trust them.

The cost shows up in summaries of mixed recordings: with nothing to distinguish
participants, the model writes "the speaker" throughout and may conflate two
people. If attribution matters, record with `meet start`.

## Language

Qwen3-ASR accepts exactly one language, and **Mandarin is the right choice for a
bilingual meeting**: decoding as Chinese transcribes embedded English correctly
(`不 sure if you can hear me because`) and, on an English-only track, produced
output identical to forcing English.

Left on auto-detection the model picks a language per chunk and drifts. On a
noisy English clip it emitted Devanagari mid-sentence. Forcing `zh` eliminated
that.

The trade-off: on English-only speech, Mandarin decoding occasionally renders a
function word in Chinese — `Kubernetes 和 Postgres` for a spoken "and". Harmless
for summarising. Use `--language en` for meetings you know are entirely English,
and `--language auto` only for audio that is neither Mandarin nor English.

Forcing `en` carries one hazard worth knowing about: on an unintelligible segment
the model has echoed the glossary back as if it were speech. Output that repeats
the vocabulary list is detected and discarded rather than written to a transcript.

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

1. the real environment, so `MEET_VAULT=/tmp meet stop` wins for one run
2. `.env` in the project directory
3. the built-in default

See `.env.example` for the full list with defaults. The ones most people change:

| variable | purpose |
|---|---|
| `MEET_VAULT` | where notes are written (default `~/Documents/Obsidian`) |
| `MEET_NOTES_SUBDIR` | subdirectory within the vault (default `Notes`) |
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
chat model for the summariser. Both download on first use;
`meet setup --prefetch-models` fetches them ahead of time instead.

## Design notes

All of these were measured, not assumed.

- **No virtual audio driver.** System audio uses Core Audio process taps
  (macOS 14.2+), so no BlackHole or Loopback, and no aggregate device.
- **Never `ffmpeg -i :default`.** It does not track the macOS default input and
  silently records digital silence. The device is resolved by name via
  `system_profiler` and passed as an explicit avfoundation index.
- **Raw PCM, not m4a.** No header to finalise, so a crash mid-meeting still
  leaves a readable recording.
- **Qwen3-ASR over SenseVoice.** On Mandarin/English code-switched audio
  SenseVoice mangles or drops embedded English — `debug` became `D`, `relayer`
  became `RE LAYER`. Qwen3 transcribes it correctly and punctuates naturally.
- **`use_itn=False` for SenseVoice.** Its inverse-text-normalisation path drops
  leading Chinese characters: `开饭时间...` became `饭时间...`.
- **Spans capped at 30s.** An autoregressive recogniser's cost grows
  superlinearly with input length. Un-split spans took **61 minutes** on an
  18-minute meeting; capping them took **3 minutes** for the same audio.
- **Glossary at the recognition layer, not the summariser.** Given as context to
  Qwen3-ASR it corrected a product name from acoustics alone. The small local
  summarising model, by contrast, tries to *use* the vocabulary it is shown and
  will occasionally invent a claim around a term. Treat local summaries as
  drafts.
- **The summariser runs isolated.** mlx-lm and qwen3-asr-mlx cannot share an
  environment: transformers 5.x needs `tokenizers<=0.23.0`, qwen3-asr-mlx needs
  `>=0.23.0`, and 0.23.0 was never released. The summariser therefore runs under
  `uv run --isolated --with mlx-lm`.

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
