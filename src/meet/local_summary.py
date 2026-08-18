"""Standalone local summariser, run in its own Python environment.

mlx-lm and qwen3-asr-mlx cannot share an environment: transformers 5.x requires
``tokenizers<=0.23.0`` while qwen3-asr-mlx requires ``>=0.23.0``, and tokenizers
0.23.0 was never released, so the intersection is empty. Rather than downgrade
the recogniser, this module runs under ``uv run --isolated --with mlx-lm``.

It therefore must not import anything from `meet`. Prompt arrives on stdin, the
summary leaves on stdout.
"""

import argparse
import re
import sys

#: Qwen3 emits reasoning in <think> blocks that must not reach the note.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_ORPHAN_OPEN = re.compile(r"<think>.*", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove Qwen reasoning blocks, including one left unclosed by truncation."""
    cleaned = _THINK_BLOCK.sub("", text)
    return _ORPHAN_OPEN.sub("", cleaned).strip()


def render_prompt(tokenizer, prompt: str) -> str:
    """Apply the model's chat template, disabling thinking where supported."""
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
    except TypeError:
        # Older or non-Qwen templates do not accept enable_thinking.
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()

    prompt = sys.stdin.read()
    if not prompt.strip():
        print("error: empty prompt on stdin", file=sys.stderr)
        return 1

    # ty: ignore[unresolved-import]
    # mlx-lm is deliberately absent from this project's environment; it is
    # supplied by `uv run --isolated --with mlx-lm` when this script is invoked.
    from mlx_lm import generate, load

    model, tokenizer = load(args.model)
    output = generate(
        model,
        tokenizer,
        prompt=render_prompt(tokenizer, prompt),
        max_tokens=args.max_tokens,
        verbose=False,
    )
    summary = strip_thinking(output)
    if not summary:
        print("error: model returned no summary", file=sys.stderr)
        return 1
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
