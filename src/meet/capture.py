"""Recording: two independent captures, never an aggregate device.

The microphone and the system audio are captured by two separate processes into
two separate files. That matters for two reasons:

* Raw PCM has no header to finalise, so if a process dies 40 minutes into a
  meeting everything already written is still a readable recording.
* The two captures cannot desynchronise the way members of a macOS aggregate
  device can, and a failure of one is visible rather than silent.

System audio uses Core Audio process taps via audiotee, so no virtual audio
driver is needed. The microphone goes through ffmpeg with an explicit
avfoundation device index -- never ``:default``, which yields silence.
"""

import contextlib
import json
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path

from meet.audio import resolve_input
from meet.config import AUDIOTEE, RECORDINGS, SAMPLE_RATE, require

ACTIVE = RECORDINGS / "active.json"
MIC_FILE = "mic.pcm"
SYSTEM_FILE = "system.pcm"
META_FILE = "session.json"

#: How long to wait for a capture process to exit after asking politely.
STOP_TIMEOUT = 5.0


def _spawn(args: list[str], stdout: Path, stderr: Path) -> int:
    """Start a detached capture process, returning its pid."""
    with stdout.open("wb") as out, stderr.open("wb") as err:
        process = subprocess.Popen(
            args, stdout=out, stderr=err, stdin=subprocess.DEVNULL, start_new_session=True
        )
    return process.pid


def active_session() -> dict | None:
    """The in-progress recording, or None."""
    if not ACTIVE.exists():
        return None
    try:
        return json.loads(ACTIVE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def start(name: str | None = None) -> dict:
    """Begin recording microphone and system audio into a new session directory.

    Raises:
        RuntimeError: If a recording is already in progress.
        FileNotFoundError: If the audiotee binary is missing.
    """
    if active_session() is not None:
        raise RuntimeError("a recording is already in progress; run `meet stop` first")
    require(AUDIOTEE, "build it: cd ~/.local/share/meet/vendor/audiotee && swift build -c release")

    started = datetime.now()
    slug = started.strftime("%Y-%m-%d-%H%M")
    if name:
        slug = f"{slug}-{name.replace('/', '-')}"
    directory = RECORDINGS / slug
    directory.mkdir(parents=True, exist_ok=True)

    device_name, device_index = resolve_input()

    system_pid = _spawn(
        [str(AUDIOTEE), "--sample-rate", str(SAMPLE_RATE)],
        directory / SYSTEM_FILE,
        directory / "system.log",
    )
    mic_pid = _spawn(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-nostdin",
            "-f",
            "avfoundation",
            "-i",
            f":{device_index}",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            "-f",
            "s16le",
            "-y",
            str(directory / MIC_FILE),
        ],
        directory / "mic.stdout",
        directory / "mic.log",
    )

    session = {
        "dir": str(directory),
        "started": started.isoformat(timespec="seconds"),
        "mic_device": device_name,
        "mic_index": device_index,
        "system_pid": system_pid,
        "mic_pid": mic_pid,
    }
    (directory / META_FILE).write_text(json.dumps(session, indent=2), encoding="utf-8")
    ACTIVE.write_text(json.dumps(session, indent=2), encoding="utf-8")
    return session


def _terminate(pid: int) -> None:
    """Ask a capture process to stop, then insist."""
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + STOP_TIMEOUT
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)


def stop() -> dict:
    """End the in-progress recording and report what was captured.

    Raises:
        RuntimeError: If no recording is in progress.
    """
    session = active_session()
    if session is None:
        raise RuntimeError("no recording in progress")
    for key in ("mic_pid", "system_pid"):
        _terminate(int(session[key]))
    ACTIVE.unlink(missing_ok=True)

    directory = Path(session["dir"])
    session["mic_seconds"] = pcm_seconds(directory / MIC_FILE)
    session["system_seconds"] = pcm_seconds(directory / SYSTEM_FILE)
    session["stopped"] = datetime.now().isoformat(timespec="seconds")
    (directory / META_FILE).write_text(json.dumps(session, indent=2), encoding="utf-8")
    return session


def pcm_seconds(path: Path) -> float:
    """Duration of a headerless 16-bit mono PCM file."""
    if not path.exists():
        return 0.0
    return path.stat().st_size / (2 * SAMPLE_RATE)


#: Fraction by which two tracks recorded together may differ in length.
#:
#: They start and stop together, so they should match closely. A larger gap means
#: one capture died early. The usual cause is the microphone's device vanishing
#: mid-meeting -- unplugging a headset, or a Bluetooth device disconnecting --
#: because ffmpeg holds one fixed avfoundation index for the whole recording and
#: cannot follow the change.
TRACK_DRIFT_TOLERANCE = 0.1

#: Seconds of difference to forgive regardless of the relative tolerance.
#:
#: The two capture processes do not start in lockstep -- ffmpeg takes about a
#: second longer to open its device than audiotee does -- and on a short
#: recording that skew alone exceeds the relative tolerance. Requiring both an
#: absolute and a relative gap keeps the warning meaningful.
MIN_ABSOLUTE_DRIFT = 3.0


def tracks_diverged(
    mic_seconds: float,
    system_seconds: float,
    tolerance: float = TRACK_DRIFT_TOLERANCE,
) -> bool:
    """Whether one track is materially shorter than the other."""
    longer = max(mic_seconds, system_seconds)
    if longer < 1:
        return False
    difference = abs(mic_seconds - system_seconds)
    return difference > MIN_ABSOLUTE_DRIFT and difference / longer > tolerance
