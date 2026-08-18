"""Filesystem layout, model selection, and `.env` loading.

Every parameter is an environment variable, so nothing about one machine is
baked into the code. Values are resolved in this order:

1. the real environment, so `MEET_NOTES=/tmp meet ...` always wins
2. `.env` in the project root, which `meet setup` creates
3. the built-in default

Paths hang off a single root. In a git checkout that root *is* the checkout, so a
clone anywhere keeps its own models, recordings, and configuration instead of
reaching back into another copy.
"""

import os
from collections.abc import MutableMapping
from pathlib import Path

#: Marks a directory as a checkout rather than an installed package.
_PROJECT_MARKER = "pyproject.toml"

#: Where paths live when the package is installed rather than checked out.
_INSTALLED_FALLBACK = Path.home() / ".local/share/meet"

DOTENV_NAME = ".env"
DOTENV_EXAMPLE = ".env.example"
GLOSSARY_EXAMPLE = "glossary.example.txt"


def parse_env(text: str) -> dict[str, str]:
    """Read `KEY=value` lines from `.env` contents.

    Blank lines and `#` comments are skipped, a leading `export` is allowed, and
    matched surrounding quotes are stripped. Only the first `=` splits, so values
    may contain more. A `#` inside a value is kept rather than treated as an
    inline comment, because paths and prompts legitimately contain one.
    """
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        line = line.removeprefix("export ").lstrip()
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def apply_env(values: dict[str, str], environ: MutableMapping[str, str]) -> int:
    """Copy `values` into `environ` without clobbering what is already set.

    Returns:
        How many variables were actually applied.
    """
    applied = 0
    for key, value in values.items():
        if key not in environ:
            environ[key] = value
            applied += 1
    return applied


def resolve_root(module_file: Path, fallback: Path = _INSTALLED_FALLBACK) -> Path:
    """Locate the project root that all other paths hang off.

    Args:
        module_file: Path of this module, i.e. `<root>/src/meet/config.py`.
        fallback: Used when `module_file` is not inside a checkout.
    """
    candidate = module_file.resolve().parents[2]
    return candidate if (candidate / _PROJECT_MARKER).exists() else fallback


def _bootstrap() -> Path:
    """Establish the root and load its `.env` before any setting is read."""
    root = Path(os.environ.get("MEET_ROOT", resolve_root(Path(__file__))))
    dotenv = root / DOTENV_NAME
    if dotenv.exists():
        apply_env(parse_env(dotenv.read_text(encoding="utf-8")), os.environ)
    return root


ROOT = _bootstrap()

MODELS = Path(os.environ.get("MEET_MODELS", ROOT / "models"))
RECORDINGS = Path(os.environ.get("MEET_RECORDINGS", ROOT / "recordings"))
VENDOR = Path(os.environ.get("MEET_VENDOR", ROOT / "vendor"))

#: Directory finished notes are written to, created on demand.
#:
#: Defaults inside the project, so the tool works with no configuration and
#: assumes nothing about how notes are kept. Point it into an Obsidian vault, or
#: anywhere else, with `MEET_NOTES=~/Documents/MyVault/Meetings`.
NOTES = Path(os.environ.get("MEET_NOTES", ROOT / "notes")).expanduser()

GLOSSARY_FILE = Path(os.environ.get("MEET_GLOSSARY", ROOT / "glossary.txt")).expanduser()

AUDIOTEE = Path(
    os.environ.get("MEET_AUDIOTEE", VENDOR / "audiotee/.build/release/audiotee")
).expanduser()

#: Speech recogniser. Any Qwen3-ASR conversion on Hugging Face works here.
ASR_MODEL_ID = os.environ.get("MEET_ASR_MODEL", "mlx-community/Qwen3-ASR-1.7B-bf16")

#: Language the recogniser is pinned to when `--language` is not given.
#:
#: English suits most meetings. Set `MEET_LANGUAGE=zh` for Mandarin or mixed
#: Mandarin/English audio: Chinese decoding transcribes English embedded in
#: Mandarin correctly, which English decoding does not.
FALLBACK_LANGUAGE = "en"
LANGUAGE = os.environ.get("MEET_LANGUAGE", FALLBACK_LANGUAGE)

#: Summarising model, kept local so a meeting need never leave the machine.
SUMMARY_MODEL_ID = os.environ.get("MEET_SUMMARY_MODEL", "mlx-community/Qwen3.5-9B-MLX-4bit")

SENSE_VOICE_DIR = Path(
    os.environ.get(
        "MEET_SENSEVOICE_DIR",
        MODELS / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2025-09-09",
    )
).expanduser()
VAD_MODEL = Path(os.environ.get("MEET_VAD_MODEL", MODELS / "silero_vad.onnx")).expanduser()

SAMPLE_RATE = 16_000


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
