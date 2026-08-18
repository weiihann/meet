"""Transcript assembly.

Pure data manipulation: no models, no filesystem, no clock. Everything here is
deterministic so the transcript shape can be tested without running inference.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

#: Same-speaker turns separated by more than this are kept as distinct turns.
DEFAULT_MAX_GAP = 30.0

#: Speaker label for the local microphone track.
ME = "Me"

#: Speaker label for the system-audio track: everyone on the far end.
THEM = "Them"


@dataclass(frozen=True, slots=True)
class Segment:
    """One contiguous stretch of speech.

    Attributes:
        start: Offset in seconds from the beginning of the meeting.
        end: Offset in seconds at which the speech stops.
        speaker: Display label, e.g. ``"Me"``. Empty when unknown, which is the
            normal case for a single mixed recording.
        text: Recognised words, empty until an ASR engine fills it in.
    """

    start: float
    end: float
    speaker: str = ""
    text: str = ""


def timestamp(seconds: float) -> str:
    """Render `seconds` as ``MM:SS``, widening to ``H:MM:SS`` past an hour.

    Raises:
        ValueError: If `seconds` is negative.
    """
    if seconds < 0:
        raise ValueError(f"negative time: {seconds}")
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def coalesce(
    segments: Iterable[Segment],
    max_gap: float = DEFAULT_MAX_GAP,
) -> list[Segment]:
    """Join runs of same-speaker segments into readable turns.

    Segments whose text is blank are dropped -- those are spans where the engine
    recognised nothing.

    Args:
        segments: Spans in chronological order.
        max_gap: Seconds of silence that still count as the same turn. Pass 0 to
            join only spans that touch, which is what a single mixed recording
            wants: its spans are already utterances and should stay separate.
    """
    turns: list[Segment] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        previous = turns[-1] if turns else None
        if (
            previous is not None
            and previous.speaker == segment.speaker
            and segment.start - previous.end <= max_gap
        ):
            turns[-1] = replace(previous, end=segment.end, text=f"{previous.text} {text}")
        else:
            turns.append(replace(segment, text=text))
    return turns


def merge_tracks(mine: Sequence[Segment], theirs: Sequence[Segment]) -> list[Segment]:
    """Interleave the microphone and system-audio tracks onto one clock.

    Both tracks start at the same wall-clock instant, so their offsets are
    directly comparable. Ties resolve in favour of the microphone track: when I
    start talking at the same moment as someone else, my words read first.
    """
    return sorted([*mine, *theirs], key=lambda s: (s.start, s.speaker != ME))


def format_transcript(segments: Sequence[Segment]) -> str:
    """Render segments as timestamped Markdown lines.

    The speaker prefix is omitted when the speaker is unknown, rather than
    printing a placeholder label that implies knowledge we do not have.
    """
    if not segments:
        return "_(no speech detected)_"
    lines = []
    for segment in segments:
        prefix = f"[{timestamp(segment.start)}]"
        speaker = f" **{segment.speaker}:**" if segment.speaker else ""
        lines.append(f"{prefix}{speaker} {segment.text}")
    return "\n".join(lines)
