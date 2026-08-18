"""Speech recognition engines.

Two engines behind one interface.

Qwen3-ASR is the default. Measured on Mandarin/English code-switched audio it
transcribes embedded English correctly and punctuates naturally, where
SenseVoice-Small mangles or drops English words outright ("debug" became "D",
"relayer" became "RE LAYER"). It runs at roughly 8x realtime.

SenseVoice-Small is kept because it runs at roughly 90x realtime, which makes it
useful for a quick first pass on a long recording. Note that its `use_itn`
option corrupts Chinese -- it drops leading characters -- so it is left off.
"""

from typing import Protocol

import numpy as np

from meet.config import ASR_MODEL_ID, LANGUAGE, SAMPLE_RATE, SENSE_VOICE_DIR, require

#: Segments shorter than this are noise, not speech, and confuse both engines.
MIN_SEGMENT_SECONDS = 0.25

#: Qwen3-ASR is pinned to exactly one language rather than detecting per chunk.
#:
#: Left on auto-detection the model drifts: on a noisy English clip it emitted
#: Devanagari mid-sentence. Pinning removes that entirely, so `auto` exists only
#: for audio that is neither English nor Mandarin.
#:
#: For **mixed Mandarin/English** audio, `zh` is the better pin even for speakers
#: who mostly use English: Chinese decoding transcribes embedded English
#: correctly ("不 sure if you can hear me") and, on an English-only track,
#: produced output identical to pinning English. Set `MEET_LANGUAGE=zh` in `.env`
#: to make that the default.
#:
#: Pinning `en` carries one hazard: on an unintelligible segment the model has
#: echoed the glossary back as though it were speech. `echoes_context` catches it.
AUTO_LANGUAGE = "auto"
LANGUAGES = ("en", "zh", AUTO_LANGUAGE)

#: Resolved from `MEET_LANGUAGE`, defaulting to English.
DEFAULT_LANGUAGE = LANGUAGE

#: Enough leading context to recognise the model parroting its own prompt.
_ECHO_PREFIX = 24


def echoes_context(text: str, context: str) -> bool:
    """Whether the recogniser returned its own prompt instead of a transcript.

    Qwen3-ASR occasionally regurgitates the vocabulary list it was given when a
    segment carries no intelligible speech. That output must be discarded rather
    than written into a transcript.
    """
    if not text or not context:
        return False
    if "Vocabulary:" in text:
        return True
    return text[:_ECHO_PREFIX].strip() == context[:_ECHO_PREFIX].strip()


QWEN3 = "qwen3"
SENSEVOICE = "sensevoice"
ENGINE_NAMES = (QWEN3, SENSEVOICE)


class Engine(Protocol):
    """Turns a mono 16 kHz waveform into text."""

    name: str

    def transcribe(self, samples: np.ndarray) -> str:
        """Recognise speech in `samples`, returning '' when there is none."""
        ...


def _too_short(samples: np.ndarray) -> bool:
    return len(samples) < MIN_SEGMENT_SECONDS * SAMPLE_RATE


class Qwen3Engine:
    """Qwen3-ASR-1.7B via MLX, with glossary terms supplied as context."""

    name = QWEN3

    def __init__(
        self,
        context: str = "",
        model_id: str = ASR_MODEL_ID,
        language: str = DEFAULT_LANGUAGE,
    ) -> None:
        from qwen3_asr_mlx import Qwen3ASR

        self._model = Qwen3ASR.from_pretrained(model_id)
        self._context = context or None
        self._language = None if language == AUTO_LANGUAGE else language

    def transcribe(self, samples: np.ndarray) -> str:
        if _too_short(samples):
            return ""
        result = self._model.transcribe(
            np.ascontiguousarray(samples, dtype=np.float32),
            context=self._context,
            language=self._language,
        )
        text = str(result.text).strip()
        return "" if echoes_context(text, self._context or "") else text


class SenseVoiceEngine:
    """SenseVoice-Small via sherpa-onnx: much faster, weaker on English."""

    name = SENSEVOICE

    def __init__(self, num_threads: int = 4) -> None:
        import sherpa_onnx

        model = require(
            SENSE_VOICE_DIR / "model.onnx",
            "run `meet setup --with-sensevoice` to download it (929 MB)",
        )
        self._recogniser = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model),
            tokens=str(SENSE_VOICE_DIR / "tokens.txt"),
            num_threads=num_threads,
            language="",  # auto; forcing a language changes nothing measurable
            use_itn=False,  # ITN drops leading Chinese characters
        )

    def transcribe(self, samples: np.ndarray) -> str:
        if _too_short(samples):
            return ""
        stream = self._recogniser.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples)
        self._recogniser.decode_stream(stream)
        return str(stream.result.text).strip()


def resolve_language(value: str) -> str:
    """Validate a configured language.

    Checked explicitly because `MEET_LANGUAGE` bypasses argparse's `choices`:
    argparse does not validate a default it was handed.

    Raises:
        ValueError: If `value` is not one of `LANGUAGES`.
    """
    if value not in LANGUAGES:
        raise ValueError(f"unknown language {value!r}; choose from {', '.join(LANGUAGES)}")
    return value


def load_engine(name: str, context: str = "", language: str = DEFAULT_LANGUAGE) -> Engine:
    """Instantiate an ASR engine by name.

    Args:
        name: One of `ENGINE_NAMES`.
        context: Glossary text; only Qwen3 can use it.
        language: One of `LANGUAGES`; only Qwen3 honours it.

    Raises:
        ValueError: If `name` or `language` is not recognised.
    """
    language = resolve_language(language)
    if name == QWEN3:
        return Qwen3Engine(context=context, language=language)
    if name == SENSEVOICE:
        return SenseVoiceEngine()
    raise ValueError(f"unknown engine {name!r}; choose from {', '.join(ENGINE_NAMES)}")
