"""Speech detection.

Splits a waveform into speech spans and discards silence.

This replaces speaker diarisation, which was removed. On a real 18-minute
two-person call, clustering reported 38 speakers at one threshold and 13 at
another: labels that confident and that wrong are worse than no labels, because
they invite you to trust them. Tuning did not fix it -- the value that recovered
exactly 4 speakers on a clean reference recording was the same value that
invented 39 here.

Speaker attribution now comes only from where the audio was captured: the
microphone track is the user, the system track is everyone else. That is a
property of the recording rather than a model's guess, so it cannot be wrong.
"""

import numpy as np

from meet.config import SAMPLE_RATE, VAD_MODEL, require
from meet.segments import Segment

#: Longest span handed to the recogniser in one call.
#:
#: An autoregressive recogniser's cost grows superlinearly with input length: on
#: an 18-minute meeting, un-split spans took 61 minutes to transcribe against 7
#: for shorter spans covering the same audio. 30s also matches the window these
#: models are trained on.
MAX_SPEECH_SECONDS = 30.0

#: Silence shorter than this does not end an utterance.
MIN_SILENCE_SECONDS = 0.35

#: Ignore blips shorter than this; they are noise, not speech.
MIN_SPEECH_SECONDS = 0.25

#: Silero consumes fixed 512-sample windows at 16 kHz.
WINDOW_SIZE = 512

#: How much audio the detector may buffer while deciding.
BUFFER_SECONDS = 60


def _build(max_speech: float):
    import sherpa_onnx

    model = require(VAD_MODEL, "run `meet doctor` to see which models are missing")
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = str(model)
    config.silero_vad.min_silence_duration = MIN_SILENCE_SECONDS
    config.silero_vad.min_speech_duration = MIN_SPEECH_SECONDS
    config.silero_vad.max_speech_duration = max_speech
    config.silero_vad.window_size = WINDOW_SIZE
    config.sample_rate = SAMPLE_RATE
    if not config.validate():
        raise RuntimeError("VAD config rejected by sherpa-onnx; check the model path")
    return sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=BUFFER_SECONDS)


def detect_speech(
    samples: np.ndarray,
    *,
    label: str = "",
    max_speech: float = MAX_SPEECH_SECONDS,
) -> list[Segment]:
    """Find the spans of `samples` that contain speech.

    Args:
        samples: Mono waveform at `SAMPLE_RATE`.
        label: Speaker name to attach to every span. Empty when unknown.
        max_speech: Longest span to emit; longer speech is cut into pieces.

    Returns:
        Speech spans in chronological order, with empty `text`.
    """
    if len(samples) == 0:
        return []
    detector = _build(max_speech)
    spans: list[Segment] = []

    def drain() -> None:
        while not detector.empty():
            speech = detector.front
            start = speech.start / SAMPLE_RATE
            end = (speech.start + len(speech.samples)) / SAMPLE_RATE
            spans.append(Segment(start=start, end=end, speaker=label))
            detector.pop()

    for offset in range(0, len(samples), WINDOW_SIZE):
        detector.accept_waveform(samples[offset : offset + WINDOW_SIZE])
        drain()
    detector.flush()
    drain()
    return spans
