"""Filesystem layout and model locations.

Every path can be overridden by an environment variable so the tool stays usable
from a different vault or model directory without editing code.
"""

import os
from pathlib import Path

ROOT = Path(os.environ.get("MEET_ROOT", Path.home() / ".local/share/meet"))
MODELS = Path(os.environ.get("MEET_MODELS", ROOT / "models"))
RECORDINGS = Path(os.environ.get("MEET_RECORDINGS", ROOT / "recordings"))
VAULT = Path(os.environ.get("MEET_VAULT", Path.home() / "Documents/obsidian"))

AUDIOTEE = Path(os.environ.get("MEET_AUDIOTEE", ROOT / "vendor/audiotee/.build/release/audiotee"))

QWEN3_MODEL_ID = os.environ.get("MEET_QWEN3_MODEL", "mlx-community/Qwen3-ASR-1.7B-bf16")

#: Local model used for summarising, so a meeting need never leave the machine.
SUMMARY_MODEL_ID = os.environ.get("MEET_SUMMARY_MODEL", "mlx-community/Qwen3.5-9B-MLX-4bit")

SENSE_VOICE_DIR = MODELS / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2025-09-09"
VAD_MODEL = MODELS / "silero_vad.onnx"

SAMPLE_RATE = 16_000

# Where finished notes land inside the vault.
NOTES_SUBDIR = "Notes"


def require(path: Path, hint: str) -> Path:
    """Return `path`, or raise with an actionable message if it is missing.

    Args:
        path: The file or directory that must exist.
        hint: What the user should do about it.

    Raises:
        FileNotFoundError: If `path` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"missing {path}\n  {hint}")
    return path
