"""Command line interface."""

import argparse
import sys
from datetime import date
from pathlib import Path

from meet import capture
from meet.asr import DEFAULT_LANGUAGE, ENGINE_NAMES, LANGUAGES, QWEN3, load_engine
from meet.audio import (
    decode,
    duration,
    is_digital_silence,
    list_input_devices,
    probe_input,
    resolve_input,
)
from meet.config import AUDIOTEE, RECORDINGS, VAD_MODEL, VAULT
from meet.glossary import GLOSSARY_FILE, context_prompt, ensure_glossary_file, load_terms
from meet.segments import Segment, format_transcript
from meet.summarize import QWEN, SUMMARISERS, note_path, render_note, summarise
from meet.transcribe import transcribe_file, transcribe_tracks

TRANSCRIPT_FILE = "transcript.md"


def _say(message: str) -> None:
    print(message, file=sys.stderr)


def _progress(stage: str, done: int, total: int) -> None:
    end = "\n" if done == total else "\r"
    print(f"  {stage}: {done}/{total} segments", file=sys.stderr, end=end, flush=True)


def _warn_if_silent(path: Path, label: str) -> None:
    if path.exists() and is_digital_silence(decode(path)):
        _say(f"  WARNING: {label} track is digital silence; it will contribute nothing")


def _transcribe_target(target: Path, args: argparse.Namespace) -> list[Segment]:
    """Transcribe either a recorded session directory or a single mixed file."""
    engine = load_engine(args.engine, context=context_prompt(), language=args.language)
    mic = target / capture.MIC_FILE
    system = target / capture.SYSTEM_FILE
    if target.is_dir() and mic.exists() and system.exists():
        _warn_if_silent(mic, "microphone")
        _warn_if_silent(system, "system")
        return transcribe_tracks(mic, system, engine, progress=_progress)
    if target.is_dir():
        raise SystemExit(f"{target} is not a recording session (no {capture.MIC_FILE})")
    _say(f"single mixed file: {duration(decode(target)):.0f}s (no speaker labels)")
    return transcribe_file(target, engine, progress=_progress)


def _write_transcript(target: Path, segments: list[Segment]) -> Path:
    """Save the transcript beside its recording and report where it went."""
    transcript = format_transcript(segments)
    out = target / TRANSCRIPT_FILE if target.is_dir() else target.with_suffix(".transcript.md")
    out.write_text(transcript + "\n", encoding="utf-8")
    speakers = sorted({s.speaker for s in segments if s.speaker})
    _say(f"{len(segments)} turns{', speakers: ' + ', '.join(speakers) if speakers else ''}")
    _say(f"transcript -> {out}")
    return out


def _summarise_to_vault(source: Path, args: argparse.Namespace) -> Path:
    """Summarise a transcript into a note in the vault."""
    transcript = source.read_text(encoding="utf-8")
    _say(f"summarising with {args.summariser}...")
    body = summarise(
        transcript, ", ".join(load_terms()), summariser=args.summariser, model=args.model
    )
    title = args.title or source.parent.name
    destination = Path(note_path(date.today(), title))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        render_note(date.today(), title, body, transcript, org=args.org), encoding="utf-8"
    )
    _say(f"note -> {destination}")
    return destination


def _process(target: Path, args: argparse.Namespace) -> None:
    """Transcribe then summarise, the whole way from audio to vault note."""
    transcript = _write_transcript(target, _transcribe_target(target, args))
    _summarise_to_vault(transcript, args)


def cmd_start(args: argparse.Namespace) -> int:
    session = capture.start(args.name)
    _say(f"recording -> {session['dir']}")
    _say(f"  mic:    {session['mic_device']} (avfoundation index {session['mic_index']})")
    _say("  system: Core Audio process tap (all processes)")
    _say("stop with: meet stop")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    session = capture.stop()
    _say(
        f"stopped after mic {session['mic_seconds']:.0f}s / system {session['system_seconds']:.0f}s"
    )
    if session["mic_seconds"] < 1:
        _say("  WARNING: microphone track is empty -- check Microphone permission")
    if session["system_seconds"] < 1:
        _say("  WARNING: system track is empty -- check Screen & System Audio Recording")
    if capture.tracks_diverged(session["mic_seconds"], session["system_seconds"]):
        _say(
            "  WARNING: the two tracks are different lengths, so one capture stopped "
            "early -- did an audio device disconnect mid-meeting?"
        )
    directory = Path(session["dir"])
    if not args.process:
        _say(f"next: meet process {directory}")
        return 0
    _process(directory, args)
    return 0


def cmd_transcribe(args: argparse.Namespace) -> int:
    target = Path(args.target).expanduser()
    if not target.exists():
        raise SystemExit(f"no such file or directory: {target}")
    out = _write_transcript(target, _transcribe_target(target, args))
    _say(f"next: meet summarize {out} --title '...'")
    print(out.read_text(encoding="utf-8"))
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    source = Path(args.transcript).expanduser()
    if not source.exists():
        raise SystemExit(f"no such transcript: {source}")
    _summarise_to_vault(source, args)
    return 0


def cmd_process(args: argparse.Namespace) -> int:
    target = Path(args.target).expanduser()
    if not target.exists():
        raise SystemExit(f"no such file or directory: {target}")
    _process(target, args)
    return 0


def cmd_devices(_args: argparse.Namespace) -> int:
    for name, index in sorted(list_input_devices().items(), key=lambda kv: kv[1]):
        print(f"  [{index}] {name}")
    print(f"would record from: {resolve_input()[0]}")
    return 0


def cmd_doctor(_args: argparse.Namespace) -> int:
    ensure_glossary_file()
    problems = 0
    checks = [
        ("audiotee binary", AUDIOTEE),
        ("speech detector", VAD_MODEL),
        ("vault", VAULT),
        ("glossary", GLOSSARY_FILE),
    ]
    for label, path in checks:
        ok = path.exists()
        problems += not ok
        print(f"  [{'ok' if ok else 'MISSING'}] {label}: {path}")
    print(f"  glossary terms: {len(load_terms())}")
    try:
        name, index = resolve_input()
        if probe_input(index):
            print(f"  [ok] microphone: {name} (index {index}) -- signal detected")
        else:
            problems += 1
            print(f"  [SILENT] microphone: {name} (index {index}) records only zeros")
            print("           common causes: the lid is closed, which disables the")
            print("           built-in mic; or Microphone permission is not granted")
            print("           to your terminal (grant it, then restart the terminal)")
    except RuntimeError as error:
        problems += 1
        print(f"  [MISSING] microphone: {error}")
    print(f"  recordings: {RECORDINGS}")
    if problems:
        print(f"\n{problems} problem(s) found")
    return 1 if problems else 0


def _add_engine_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine", choices=ENGINE_NAMES, default=QWEN3)
    parser.add_argument(
        "--language",
        choices=LANGUAGES,
        default=DEFAULT_LANGUAGE,
        help="zh handles Mandarin and English together (default); "
        "auto detects per chunk but drifts to other languages",
    )


def _add_note_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--title", default="", help="note title; defaults to the session name")
    parser.add_argument("--org", default="")
    parser.add_argument("--summariser", choices=SUMMARISERS, default=QWEN)
    parser.add_argument("--model", default="", help="override the summarising model")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meet", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="begin recording mic + system audio")
    start.add_argument("name", nargs="?", help="optional label for the session directory")
    start.set_defaults(func=cmd_start)

    stop = sub.add_parser("stop", help="end the current recording")
    stop.add_argument(
        "--no-process",
        dest="process",
        action="store_false",
        help="just stop; leave transcribing and summarising for later",
    )
    _add_engine_flag(stop)
    _add_note_flags(stop)
    stop.set_defaults(func=cmd_stop)

    process = sub.add_parser(
        "process", help="transcribe and summarise in one step (transcribe + summarize)"
    )
    process.add_argument("target")
    _add_engine_flag(process)
    _add_note_flags(process)
    process.set_defaults(func=cmd_process)

    transcribe = sub.add_parser(
        "transcribe", help="transcribe a session directory or any single audio file"
    )
    transcribe.add_argument("target")
    _add_engine_flag(transcribe)
    transcribe.set_defaults(func=cmd_transcribe)

    summarize = sub.add_parser("summarize", help="turn a transcript into a vault note")
    summarize.add_argument("transcript")
    _add_note_flags(summarize)
    summarize.set_defaults(func=cmd_summarize)

    devices = sub.add_parser("devices", help="list audio input devices")
    devices.set_defaults(func=cmd_devices)

    doctor = sub.add_parser("doctor", help="check binaries, models, and permissions")
    doctor.set_defaults(func=cmd_doctor)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    RECORDINGS.mkdir(parents=True, exist_ok=True)
    try:
        return int(args.func(args))
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        _say(f"error: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
