# meet

Local meeting capture, bilingual (Mandarin/English) transcription, and
summarisation into an Obsidian vault. Nothing leaves the machine by default.

## Usage

The usual flow is two commands:

```bash
meet start standup             # record mic + system audio as two tracks
meet stop                      # stop, transcribe, and write the vault note
```

`meet stop` runs the whole pipeline by default. Pass `--no-process` to just stop
and deal with it later, then pick it up with any of:

```bash
meet process <session-dir>     # transcribe + summarise in one step
meet transcribe <session-dir>  # two-track: labelled "Me" and "Them"
meet transcribe <audio-file>   # single mixed file, no speaker labels
meet summarize <transcript.md> --title "Bridge sync" --org "BNB Chain"

meet devices                   # list audio inputs, show which one is used
meet doctor                    # check binaries, models, permissions
```

`transcribe` accepts anything ffmpeg can read: `wav m4a mp3 mp4 mov mkv pcm`.
Video containers decode to their audio track, so screen recordings work.

Flags:

- `--language zh` (default) / `en` / `auto` — see below.
- `--engine sensevoice` — faster on paper, much worse English. Rarely worth it.
- `--summariser claude` — better notes than the local model, but the transcript
  leaves the device. `--summariser qwen` (default) is fully local.
- `--model <id>` — override the summarising model.

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

The cost is visible in summaries of mixed recordings: with nothing to
distinguish participants, the model writes "the speaker" throughout and may
conflate two people. If attribution matters, record with `meet start`.

## Language

Qwen3-ASR accepts exactly one language, and **Mandarin is the right choice for a
bilingual meeting**: decoding as Chinese transcribes embedded English correctly
(`不 sure if you can hear me because`) and, on an English-only track, produced
output identical to forcing English.

Left on auto-detection the model picks a language per chunk and drifts. On a
noisy English clip it emitted Devanagari (`अच्छी`) mid-sentence. Forcing `zh`
eliminated that.

The trade-off: on English-only speech, Mandarin decoding occasionally renders a
function word in Chinese -- `Qwen3-ASR 和 Microsoft VibeVoice` for a spoken
"and". Harmless for summarising. Use `--language en` for meetings you know are
entirely English, and `--language auto` only for audio that is neither Mandarin
nor English.

Forcing `en` carries one hazard worth knowing about: on an unintelligible segment
the model has echoed the glossary back as if it were speech. Output that repeats
the vocabulary list is detected and discarded rather than written to a transcript.

## How it works

```
meet start ──> audiotee ──> system.pcm   (Core Audio process tap)
           └─> ffmpeg   ──> mic.pcm      (explicit avfoundation index)

transcribe ──> silero VAD ──> speech spans, silence dropped, capped at 30s
           ──> Qwen3-ASR-1.7B via MLX, glossary as context
           ──> merge tracks onto one clock, join turns ──> transcript.md

summarize  ──> Qwen3.5-9B via MLX (or claude -p)
           ──> Vault/Notes/YYYY-MM-DD Meeting - X.md
```

Measured on an 18-minute meeting: 3m14s to transcribe, 33s to summarise locally.

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
  SenseVoice mangles or drops embedded English -- `debug` became `D`, `relayer`
  became `RE LAYER`, `Parlia` became `PA`. Qwen3 transcribes it correctly and
  punctuates naturally.
- **`use_itn=False` for SenseVoice.** Its ITN path drops leading Chinese
  characters: `开饭时间...` became `饭时间...`.
- **Spans capped at 30s.** An autoregressive recogniser's cost grows
  superlinearly with input length. Un-split spans took **61 minutes** on an
  18-minute meeting; capping them took **3 minutes** for the same audio.
- **Glossary at the ASR layer, not the summariser.** Given as context to
  Qwen3-ASR it fixed `Party` to `Parlia` from acoustics alone. The small local
  summarising model, by contrast, tries to *use* the vocabulary it is shown and
  will occasionally invent a claim around a term. Treat local summaries as
  drafts.
- **Summariser runs isolated.** mlx-lm and qwen3-asr-mlx cannot share an
  environment: transformers 5.x needs `tokenizers<=0.23.0`, qwen3-asr-mlx needs
  `>=0.23.0`, and 0.23.0 was never released. The summariser therefore runs under
  `uv run --isolated --with mlx-lm`.

## Configuration

Edit `~/.local/share/meet/glossary.txt` — one term per line, `#` for comments.
Adding colleagues' names and project jargon directly improves recognition.

Overridable by environment variable: `MEET_VAULT`, `MEET_ROOT`, `MEET_MODELS`,
`MEET_RECORDINGS`, `MEET_QWEN3_MODEL`, `MEET_SUMMARY_MODEL`, `MEET_AUDIOTEE`.

## Permissions

Both must be granted to the terminal you run `meet` from, which then needs a
restart:

- **Microphone**
- **Screen & System Audio Recording**

macOS signals a denied audio permission by delivering **exact zeros** rather than
an error, so `meet doctor` records briefly and checks whether any audio arrived,
and `meet transcribe` warns rather than producing a transcript quietly missing one
side of the conversation.

The check is for literal zeros, not for a quiet track: a working Bluetooth mic in
a silent room peaks around -60 dB, which is inaudible but not zero, so an
amplitude threshold would report working hardware as broken.

## Microphone gotchas

**The built-in mic is unavailable with the lid closed.** In clamshell mode the
MacBook microphone returns digital silence -- confirmed through both ffmpeg and
PortAudio, so it is a hardware/OS behaviour rather than a tool bug. Wear a headset
or attach an external mic. `meet stop` warns when the microphone track is silent.

**Do not change audio devices mid-recording.** ffmpeg holds one fixed
avfoundation index for the whole session and audiotee taps only the default
output device, so disconnecting a headset part-way through truncates the
microphone track while system audio keeps recording. `meet stop` compares the two
track lengths and warns when they diverge, but the lost audio is not recoverable.
Start the recording with the devices you intend to finish with.

## Development

```bash
cd ~/.local/share/meet
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run ty check src
```
