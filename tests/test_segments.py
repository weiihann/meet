"""Behaviour of the pure transcript-assembly logic."""

import pytest

from meet.segments import ME, THEM, Segment, coalesce, format_transcript, merge_tracks, timestamp


class TestTimestamp:
    def test_formats_under_an_hour_as_mm_ss(self):
        assert timestamp(0) == "00:00"
        assert timestamp(65) == "01:05"
        assert timestamp(599.9) == "09:59"

    def test_switches_to_h_mm_ss_past_an_hour(self):
        assert timestamp(3600) == "1:00:00"
        assert timestamp(3725) == "1:02:05"

    def test_rejects_negative_time(self):
        with pytest.raises(ValueError, match="negative"):
            timestamp(-1)


class TestCoalesce:
    def test_joins_consecutive_segments_from_one_speaker(self):
        out = coalesce(
            [
                Segment(0.0, 2.0, ME, "let's start"),
                Segment(2.1, 4.0, ME, "with the BEP"),
            ]
        )
        assert len(out) == 1
        assert out[0].text == "let's start with the BEP"
        assert out[0].start == 0.0
        assert out[0].end == 4.0

    def test_keeps_different_speakers_apart(self):
        out = coalesce(
            [
                Segment(0.0, 2.0, ME, "hello"),
                Segment(2.0, 4.0, THEM, "你好"),
                Segment(4.0, 6.0, ME, "ok"),
            ]
        )
        assert [s.speaker for s in out] == [ME, THEM, ME]

    def test_drops_segments_with_no_recognised_text(self):
        out = coalesce(
            [
                Segment(0.0, 2.0, ME, "real words"),
                Segment(2.0, 3.0, ME, "   "),
                Segment(3.0, 4.0, THEM, ""),
            ]
        )
        assert len(out) == 1
        assert out[0].text == "real words"

    def test_empty_input_gives_empty_output(self):
        assert coalesce([]) == []

    def test_does_not_join_across_a_long_pause(self):
        """A 40s gap is a new turn even from the same speaker."""
        out = coalesce(
            [
                Segment(0.0, 2.0, ME, "first thought"),
                Segment(42.0, 44.0, ME, "second thought"),
            ],
            max_gap=30.0,
        )
        assert len(out) == 2

    def test_zero_gap_keeps_separate_utterances_apart(self):
        """The single-file path relies on this: utterances must not become a blob."""
        out = coalesce(
            [
                Segment(0.0, 2.0, "", "first utterance"),
                Segment(2.5, 4.0, "", "second utterance"),
            ],
            max_gap=0.0,
        )
        assert len(out) == 2

    def test_zero_gap_still_joins_spans_that_touch(self):
        """A long utterance cut at the 30s cap must read as one turn again."""
        out = coalesce(
            [
                Segment(0.0, 30.0, "", "first half"),
                Segment(30.0, 45.0, "", "second half"),
            ],
            max_gap=0.0,
        )
        assert len(out) == 1
        assert out[0].text == "first half second half"


class TestMergeTracks:
    def test_orders_both_tracks_onto_one_clock(self):
        mine = [Segment(1.0, 2.0, ME, "a"), Segment(5.0, 6.0, ME, "c")]
        theirs = [Segment(3.0, 4.0, THEM, "b")]
        assert [s.text for s in merge_tracks(mine, theirs)] == ["a", "b", "c"]

    def test_my_track_wins_ties_so_my_words_read_first(self):
        mine = [Segment(1.0, 2.0, ME, "mine")]
        theirs = [Segment(1.0, 2.0, THEM, "theirs")]
        assert [s.text for s in merge_tracks(mine, theirs)] == ["mine", "theirs"]


class TestFormatTranscript:
    def test_renders_timestamp_speaker_and_text(self):
        out = format_transcript([Segment(65.0, 70.0, THEM, "好，我先 share screen")])
        assert out == "[01:05] **Them:** 好，我先 share screen"

    def test_omits_the_speaker_when_it_is_unknown(self):
        """A single mixed recording has no speaker info; do not fake a label."""
        out = format_transcript([Segment(5.0, 7.0, "", "just the words")])
        assert out == "[00:05] just the words"

    def test_blank_transcript_is_stated_not_silently_empty(self):
        assert "no speech" in format_transcript([]).lower()
