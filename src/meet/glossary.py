"""Domain vocabulary for recognition and correction.

Qwen3-ASR takes a free-text context string, and injecting these terms measurably
fixes errors: on a code-switched test clip "Parlia" was recognised as "Party"
without the glossary and correctly with it.

SenseVoice has no equivalent hook -- CTC models accept no hotwords -- so under
that engine the same list is only used to correct the transcript during
summarisation.

The list lives in a plain text file so it can be edited without touching code.
One term per line; blank lines and `#` comments are ignored.
"""

from pathlib import Path

from meet.config import ROOT

GLOSSARY_FILE = ROOT / "glossary.txt"

DEFAULT_TERMS = (
    "# Terms fed to the recogniser as context. One per line.",
    "# Add colleagues' names and project jargon as you encounter them.",
    "",
    "# --- BNB Chain ---",
    "BNB Chain",
    "New L1",
    "BSC",
    "opBNB",
    "Greenfield",
    "Parlia",
    "BEP",
    "validator",
    "slashing",
    "epoch",
    "relayer",
    "mempool",
    "gas limit",
    "gateway",
    "subblock",
    "preconfirmation",
    "PerpDex",
    "MultiDB",
    "ShardingDB",
    "",
    "# --- Execution / tooling ---",
    "Reth",
    "Geth",
    "EVM",
    "opcode",
    "MPT",
    "BLAKE3",
    "lattice hash",
    "Foundry",
    "devnet",
    "testnet",
    "mainnet",
    "account abstraction",
    "parallel execution",
    "MEV",
    "builder",
    "searcher",
    "CEX-DEX",
    "",
    "# --- AI / speech ---",
    "Qwen",
    "Qwen3",
    "Qwen3-ASR",
    "Whisper",
    "SenseVoice",
    "VibeVoice",
    "ASR",
    "diarization",
    "MLX",
    "LLM",
)


def ensure_glossary_file() -> Path:
    """Create the glossary with sensible defaults if it does not exist yet."""
    if not GLOSSARY_FILE.exists():
        GLOSSARY_FILE.parent.mkdir(parents=True, exist_ok=True)
        GLOSSARY_FILE.write_text("\n".join(DEFAULT_TERMS) + "\n", encoding="utf-8")
    return GLOSSARY_FILE


def parse_terms(text: str) -> list[str]:
    """Extract terms from glossary file contents, dropping comments and blanks."""
    terms = []
    for raw in text.splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            terms.append(line)
    return terms


def load_terms() -> list[str]:
    """Read the user's glossary, falling back to the built-in defaults."""
    if GLOSSARY_FILE.exists():
        return parse_terms(GLOSSARY_FILE.read_text(encoding="utf-8"))
    return parse_terms("\n".join(DEFAULT_TERMS))


def context_prompt(terms: list[str] | None = None) -> str:
    """Render terms as the context string Qwen3-ASR expects."""
    words = terms if terms is not None else load_terms()
    if not words:
        return ""
    return "Vocabulary: " + ", ".join(words) + "."
