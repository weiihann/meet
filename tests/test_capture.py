"""Behaviour of recording bookkeeping."""

from meet.capture import tracks_diverged


class TestTracksDiverged:
    def test_equal_length_tracks_are_fine(self):
        assert not tracks_diverged(600.0, 600.0)

    def test_small_drift_is_tolerated(self):
        """Captures start milliseconds apart; that is not a failure."""
        assert not tracks_diverged(600.0, 597.0)

    def test_a_mic_that_died_early_is_reported(self):
        """The symptom of a headset disconnecting mid-meeting."""
        assert tracks_diverged(120.0, 600.0)

    def test_direction_does_not_matter(self):
        assert tracks_diverged(600.0, 120.0)

    def test_two_empty_tracks_do_not_look_like_drift(self):
        """Both empty is a different problem, reported by its own warning."""
        assert not tracks_diverged(0.0, 0.0)

    def test_very_short_recordings_are_not_judged(self):
        """A one-second test recording should not raise a scary warning."""
        assert not tracks_diverged(0.4, 0.9)

    def test_startup_skew_on_a_short_recording_is_not_a_warning(self):
        """ffmpeg opens its device ~1s slower than audiotee; 8s vs 9s is normal."""
        assert not tracks_diverged(8.0, 9.0)

    def test_a_real_truncation_still_trips_at_short_durations(self):
        assert tracks_diverged(2.0, 30.0)
