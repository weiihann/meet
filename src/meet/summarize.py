"""Summarising a transcript into an Obsidian note.

The summariser doubles as the transcript's proofreader. Recognition errors on
domain jargon survive into the transcript -- SenseVoice has no hotword hook at
all -- so the glossary is handed to the model with instructions to correct
obvious mishearings using sentence context, which it does far better than a
string-replacement pass could.

Output matches the vault's own Meeting Template: `created` / `org` frontmatter,
tagged `note` and `meeting`, then Notes and Action Items.
"""

import re
import subprocess
from datetime import date
from pathlib import Path

from meet.config import NOTES_SUBDIR, SUMMARY_MODEL_ID, VAULT

#: Local Qwen via MLX. The default, so a meeting need never leave the machine.
QWEN = "qwen"
#: Claude Code in print mode. Better notes, but the transcript leaves the device.
CLAUDE = "claude"
SUMMARISERS = (QWEN, CLAUDE)

#: mlx-lm cannot share an environment with the recogniser (see local_summary),
#: so the local summariser runs isolated with its own dependency set.
_MLX_LM_PIN = "mlx-lm==0.31.3"
_LOCAL_SCRIPT = Path(__file__).with_name("local_summary.py")

#: Characters macOS and Obsidian both dislike in filenames.
_UNSAFE = re.compile(r'[/\\:*?"<>|]+')

PROMPT_TEMPLATE = """\
You are turning a raw, machine-generated meeting transcript into meeting notes.

The transcript comes from automatic speech recognition of a bilingual
Mandarin/English meeting, so expect two kinds of error:

1. Mis-recognised technical jargon and product names.
2. Missing or wrong punctuation, and occasional dropped short words.

Correct those silently using sentence context and this vocabulary list:
{glossary}

Rules:
- Do NOT invent content. If something is unclear, put it under Open Questions
  rather than guessing.
- Keep Mandarin in Mandarin. Do not translate it to English.
- Speaker labels are approximate; "Me" is the person taking the notes. If the
  labels are obviously wrong, write around them instead of repeating them.
- Attribute action items to a person only when the transcript makes it clear.

Produce exactly these Markdown sections, and nothing else:

## Notes
Bullet points grouped under `###` subheadings by topic. Terse and factual.

## Decisions
What was actually settled. Omit the section entirely if nothing was.

## Action Items
`- [ ]` checkboxes. Prefix mine with **(me)**. Omit if there are none.

## Open Questions
Things left unresolved. Omit if there are none.

Transcript:
---
{transcript}
---
"""


def build_prompt(transcript: str, glossary: str) -> str:
    """Assemble the summarisation prompt."""
    return PROMPT_TEMPLATE.format(glossary=glossary or "(none)", transcript=transcript)


def safe_title(title: str) -> str:
    """Make `title` usable as a filename, without collapsing it to nothing."""
    cleaned = _UNSAFE.sub("-", title).strip().strip(".")
    return cleaned or "Untitled"


def note_path(when: date, title: str) -> str:
    """Where the finished note belongs inside the vault."""
    return str(VAULT / NOTES_SUBDIR / f"{when.isoformat()} Meeting - {safe_title(title)}.md")


def render_note(when: date, title: str, body: str, transcript: str, org: str = "") -> str:
    """Wrap the model's summary in the vault's meeting-note format."""
    return (
        "---\n"
        f'created: "{when.isoformat()}"\n'
        f"org: {org}\n"
        "tags:\n  - note\n  - meeting\n"
        "---\n"
        f"# {title}\n\n"
        f"{body.strip()}\n\n"
        "## Transcript\n\n"
        "> [!note]- Full transcript\n"
        + "\n".join(f"> {line}" for line in transcript.splitlines())
        + "\n"
    )


def _run(args: list[str], prompt: str) -> str:
    result = subprocess.run(args, input=prompt, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{args[0]} failed: {result.stderr.strip() or 'no error output'}")
    return result.stdout.strip()


def summarise(transcript: str, glossary: str, summariser: str = QWEN, model: str = "") -> str:
    """Summarise `transcript`, returning Markdown sections.

    Args:
        transcript: The rendered, speaker-tagged transcript.
        glossary: Domain vocabulary for error correction.
        summariser: `qwen` (default, fully local) or `claude`.
        model: Model override.

    Raises:
        ValueError: If `summariser` is unknown.
        RuntimeError: If the underlying command fails.
    """
    prompt = build_prompt(transcript, glossary)
    if summariser == QWEN:
        return _run(
            [
                "uv",
                "run",
                "--isolated",
                "--quiet",
                "--with",
                _MLX_LM_PIN,
                "python",
                str(_LOCAL_SCRIPT),
                "--model",
                model or SUMMARY_MODEL_ID,
            ],
            prompt,
        )
    if summariser == CLAUDE:
        # --strict-mcp-config keeps the summariser hermetic: this is a pure
        # text task, and it has no business holding Gmail or Drive tools.
        # It also cuts startup from ~15s to ~6s by skipping MCP server boot.
        args = [CLAUDE, "-p", "--strict-mcp-config"]
        if model:
            args += ["--model", model]
        return _run(args, prompt)
    raise ValueError(f"unknown summariser {summariser!r}; choose from {', '.join(SUMMARISERS)}")
