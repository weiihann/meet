"""Turning recorded audio into a transcript.

Two entry points share one pipeline.

`transcribe_file` handles a single mixed recording -- a Zoom export, a screen
recording, a voice memo -- and produces timestamped utterances with *no* speaker
labels. Nothing in a single mixed track reliably reveals who is talking, and
guessed labels proved worse than none.

`transcribe_tracks` handles a `meet start` recording, where the two tracks carry
the attribution for free: the microphone is the user, the system audio is
everyone on the far end.
"""

from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from meet.asr import Engine
from meet.audio import decode
from meet.config import SAMPLE_RATE
from meet.segments import ME, THEM, Segment, coalesce, merge_tracks
from meet.vad import detect_speech

ProgressFn = Callable[[str, int, int], None]

#: Speech-detection boundaries routinely clip the first phoneme of an utterance.
#: Widening each slice recovers it. The cost is slight bleed from neighbouring
#: speech, which a recogniser handles far better than a missing word onset.
SEGMENT_PAD = 0.2

#: Utterances are already silence-bounded, so only spans that touch are joined.
CONTIGUOUS_ONLY = 0.0


def _slice(samples: np.ndarray, span: Segment, pad: float = SEGMENT_PAD) -> np.ndarray:
    start = max(0, int((span.start - pad) * SAMPLE_RATE))
    end = min(len(samples), int((span.end + pad) * SAMPLE_RATE))
    return samples[start:end]


def recognise(
    samples: np.ndarray,
    spans: Sequence[Segment],
    engine: Engine,
    progress: ProgressFn | None = None,
    stage: str = "transcribe",
) -> list[Segment]:
    """Fill in the text of each span using `engine`."""
    filled: list[Segment] = []
    for done, span in enumerate(spans, start=1):
        filled.append(replace(span, text=engine.transcribe(_slice(samples, span))))
        if progress:
            progress(stage, done, len(spans))
    return filled


def transcribe_file(
    path: Path,
    engine: Engine,
    *,
    progress: ProgressFn | None = None,
) -> list[Segment]:
    """Transcribe one mixed recording as unattributed timestamped utterances."""
    samples = decode(path)
    spans = detect_speech(samples)
    return coalesce(recognise(samples, spans, engine, progress), max_gap=CONTIGUOUS_ONLY)


def transcribe_tracks(
    mic_path: Path,
    system_path: Path,
    engine: Engine,
    *,
    progress: ProgressFn | None = None,
) -> list[Segment]:
    """Transcribe a two-track recording onto a single timeline."""
    mic_samples = decode(mic_path)
    system_samples = decode(system_path)

    mine = recognise(
        mic_samples,
        detect_speech(mic_samples, label=ME),
        engine,
        progress,
        stage=ME,
    )
    theirs = recognise(
        system_samples,
        detect_speech(system_samples, label=THEM),
        engine,
        progress,
        stage=THEM,
    )
    return coalesce(merge_tracks(mine, theirs))
