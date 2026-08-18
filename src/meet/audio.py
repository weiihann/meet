"""Audio device discovery and decoding.

All decoding funnels through ffmpeg, so any container the user points at
`meet transcribe` -- m4a, mp3, wav, headerless PCM -- arrives as float32 mono at
`SAMPLE_RATE` without pulling in soundfile or librosa.

Device selection deliberately avoids ffmpeg's ``:default`` input. On macOS that
does not track the system default input device and silently yields digital
silence, which is indistinguishable from a working recording until you look at
the levels.
"""

import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from meet.config import SAMPLE_RATE

RAW_PCM_SUFFIX = ".pcm"

_DEVICE_LINE = re.compile(r"\[(\d+)\]\s+(.+?)\s*$")
_AUDIO_HEADER = "AVFoundation audio devices:"
_VIDEO_HEADER = "AVFoundation video devices:"


def parse_avfoundation_devices(stderr: str) -> dict[str, int]:
    """Map device name to avfoundation index, reading only the audio section.

    Video devices are numbered from zero in the same listing, so the section
    boundary matters: mixing them up points ffmpeg at a camera.
    """
    devices: dict[str, int] = {}
    in_audio_section = False
    for line in stderr.splitlines():
        if _VIDEO_HEADER in line:
            in_audio_section = False
            continue
        if _AUDIO_HEADER in line:
            in_audio_section = True
            continue
        if not in_audio_section:
            continue
        match = _DEVICE_LINE.search(line)
        if match:
            devices[match.group(2)] = int(match.group(1))
    return devices


def pick_input_index(devices: dict[str, int], default_name: str) -> int:
    """Choose which avfoundation index to record from.

    Prefers the system default. If that device is not capturable by
    avfoundation -- true of some virtual and Bluetooth devices -- falls back to
    the built-in microphone rather than recording silence.

    Raises:
        RuntimeError: If no audio input devices are available at all.
    """
    if not devices:
        raise RuntimeError(
            "no audio input devices visible to ffmpeg avfoundation; "
            "grant Microphone permission to your terminal and restart it"
        )
    if default_name in devices:
        return devices[default_name]
    for name, index in devices.items():
        if "MacBook" in name and "Microphone" in name:
            return index
    return min(devices.values())


def list_input_devices() -> dict[str, int]:
    """Ask ffmpeg which avfoundation audio inputs exist."""
    # Listing devices always exits non-zero: ffmpeg reports the list, then
    # fails to open the empty input we passed.
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True,
        text=True,
        check=False,
    )
    return parse_avfoundation_devices(result.stderr)


def default_input_name() -> str:
    """Name of the current macOS default audio input device, or '' if unknown."""
    result = subprocess.run(
        ["system_profiler", "-json", "SPAudioDataType"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    try:
        items = json.loads(result.stdout)["SPAudioDataType"][0]["_items"]
    except (KeyError, IndexError, json.JSONDecodeError):
        return ""
    for item in items:
        if item.get("coreaudio_default_audio_input_device") == "spaudio_yes":
            return str(item.get("_name", ""))
    return ""


def resolve_input() -> tuple[str, int]:
    """Return the (name, avfoundation index) to record the microphone from."""
    devices = list_input_devices()
    default_name = default_input_name()
    index = pick_input_index(devices, default_name)
    name = next((n for n, i in devices.items() if i == index), f"index {index}")
    return name, index


def ffmpeg_decode_args(path: str | Path) -> list[str]:
    """Build ffmpeg arguments decoding `path` to float32 mono at `SAMPLE_RATE`.

    Any container ffmpeg understands works, including video: `-vn` drops the
    video stream so screen recordings and .mp4/.mov meeting exports decode to
    their audio track instead of relying on stream auto-selection.
    """
    args = ["-hide_banner", "-loglevel", "error", "-nostdin"]
    if Path(path).suffix.lower() == RAW_PCM_SUFFIX:
        # Headerless: ffmpeg cannot sniff this, so state what audiotee wrote.
        args += ["-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1"]
    args += ["-i", str(path), "-vn", "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-"]
    return args


def decode(path: str | Path) -> np.ndarray:
    """Decode an audio file to a float32 mono waveform.

    Raises:
        RuntimeError: If ffmpeg cannot decode the file.
    """
    result = subprocess.run(["ffmpeg", *ffmpeg_decode_args(path)], capture_output=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"ffmpeg could not decode {path}: {detail}")
    return np.frombuffer(result.stdout, dtype=np.float32)


def duration(samples: np.ndarray) -> float:
    """Length of a waveform in seconds."""
    return len(samples) / SAMPLE_RATE


#: How long to record when testing an input device.
#:
#: Must comfortably exceed Bluetooth HFP negotiation, which takes one to two
#: seconds: a headset probed for only a second returns its start-up silence and
#: looks broken when it is fine.
PROBE_SECONDS = 3.0


def probe_input(index: int, seconds: float = PROBE_SECONDS) -> bool:
    """Record briefly from an input device and report whether any signal arrived.

    Naming a device proves nothing: a MacBook microphone with the lid closed, or
    a revoked permission, both enumerate normally and then deliver pure zeros.
    Actually recording is the only way to tell.
    """
    with tempfile.TemporaryDirectory() as directory:
        probe = Path(directory) / "probe.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-f",
                "avfoundation",
                "-i",
                f":{index}",
                "-t",
                str(seconds),
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "1",
                "-y",
                str(probe),
            ],
            capture_output=True,
            check=False,
        )
        if not probe.exists():
            return False
        return not is_digital_silence(decode(probe))


def is_digital_silence(samples: np.ndarray) -> bool:
    """Whether a track is literally all zeros.

    This tests whether audio *arrived*, not whether anyone spoke. macOS signals a
    disabled or unpermitted input by delivering exact zeros, whereas a working
    microphone always carries some dither -- a Bluetooth headset in a quiet room
    measured a -60 dB peak, which is inaudible but not zero.

    Comparing against a small amplitude threshold instead would flag a working
    microphone in a silent room, so no threshold is used.
    """
    return len(samples) == 0 or not bool(np.any(samples))
