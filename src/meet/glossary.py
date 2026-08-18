"""Domain vocabulary for recognition and correction.

Qwen3-ASR accepts a free-text context string, and supplying the words a meeting
will actually use measurably fixes recognition: on a code-switched test clip a
product name was transcribed as a similar-sounding common English word until the
glossary was given, and correctly afterwards.

SenseVoice has no equivalent hook -- CTC models accept no hotwords -- so under
that engine the list only helps during summarisation.

The vocabulary lives in a plain text file rather than in code, so it is data a
user owns rather than something shipped with the tool. One term per line; blank
lines and `#` comments are ignored.
"""

from pathlib import Path

from meet.config import GLOSSARY_EXAMPLE, GLOSSARY_FILE, ROOT


def example_file() -> Path:
    """The vocabulary template shipped with the project."""
    return ROOT / GLOSSARY_EXAMPLE


def ensure_glossary_file(path: Path | None = None) -> Path:
    """Create the glossary from the shipped example if it does not exist yet."""
    target = path or GLOSSARY_FILE
    if not target.exists():
        source = example_file()
        template = source.read_text(encoding="utf-8") if source.exists() else ""
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(template, encoding="utf-8")
    return target


def parse_terms(text: str) -> list[str]:
    """Extract terms from glossary contents, dropping comments and blanks."""
    terms = []
    for raw in text.splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            terms.append(line)
    return terms


def load_terms(path: Path | None = None) -> list[str]:
    """Read the vocabulary, falling back to the shipped example, then to none."""
    for candidate in (path or GLOSSARY_FILE, example_file()):
        if candidate.exists():
            return parse_terms(candidate.read_text(encoding="utf-8"))
    return []


def context_prompt(terms: list[str] | None = None) -> str:
    """Render terms as the context string the recogniser expects."""
    words = terms if terms is not None else load_terms()
    if not words:
        return ""
    return "Vocabulary: " + ", ".join(words) + "."
