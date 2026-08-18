"""Behaviour of audio device discovery and format handling."""

import numpy as np
import pytest

from meet.audio import (
    ffmpeg_decode_args,
    is_digital_silence,
    parse_avfoundation_devices,
    pick_input_index,
)

FFMPEG_DEVICE_DUMP = """\
[AVFoundation indev @ 0x14e018140] AVFoundation video devices:
[AVFoundation indev @ 0x14e018140] [0] FaceTime HD Camera
[AVFoundation indev @ 0x14e018140] AVFoundation audio devices:
[AVFoundation indev @ 0x14e018140] [0] MacBook Pro Microphone
[AVFoundation indev @ 0x14e018140] [1] WH-1000XM6
[AVFoundation indev @ 0x14e018140] [2] UGREEN Camera 4K
[in#0 @ 0x14e018000] Error opening input: Input/output error
"""


class TestParseAvfoundationDevices:
    def test_reads_only_the_audio_section(self):
        """Video devices reuse the same index numbers and must not leak in."""
        assert parse_avfoundation_devices(FFMPEG_DEVICE_DUMP) == {
            "MacBook Pro Microphone": 0,
            "WH-1000XM6": 1,
            "UGREEN Camera 4K": 2,
        }

    def test_no_audio_section_gives_nothing(self):
        assert parse_avfoundation_devices("[foo] AVFoundation video devices:\n") == {}

    def test_empty_input_gives_nothing(self):
        assert parse_avfoundation_devices("") == {}


class TestPickInputIndex:
    devices = {"MacBook Pro Microphone": 0, "WH-1000XM6": 1}

    def test_prefers_the_system_default_device(self):
        assert pick_input_index(self.devices, "WH-1000XM6") == 1

    def test_falls_back_to_built_in_when_default_is_not_capturable(self):
        """A default device absent from avfoundation must not silently yield zeros."""
        assert pick_input_index(self.devices, "Some Virtual Device") == 0

    def test_raises_when_there_is_nothing_to_record_from(self):
        with pytest.raises(RuntimeError, match="no audio input"):
            pick_input_index({}, "WH-1000XM6")


class TestFfmpegDecodeArgs:
    def test_declares_the_format_for_headerless_pcm(self):
        """Raw PCM has no header, so the input format must be stated or ffmpeg guesses."""
        args = ffmpeg_decode_args("rec.pcm")
        assert "s16le" in args
        assert args.index("-f") < args.index("-i")

    def test_lets_ffmpeg_sniff_container_formats(self):
        args = ffmpeg_decode_args("zoom.m4a")
        assert "s16le" not in args

    def test_always_outputs_float32_mono_16k(self):
        for name in ("rec.pcm", "zoom.m4a"):
            args = ffmpeg_decode_args(name)
            assert "f32le" in args
            assert "16000" in args

    def test_discards_video_so_containers_like_mp4_decode_to_audio(self):
        """Screen recordings and Zoom .mp4 exports must yield their audio track."""
        for name in ("meeting.mp4", "screen.mov", "call.mkv"):
            assert "-vn" in ffmpeg_decode_args(name)

    def test_is_case_insensitive_about_extensions(self):
        assert "s16le" in ffmpeg_decode_args("REC.PCM")


class TestIsDigitalSilence:
    def test_all_zeros_is_silence(self):
        """What macOS returns for a disabled or unpermitted input."""
        assert is_digital_silence(np.zeros(16000, dtype=np.float32))

    def test_empty_track_is_silence(self):
        assert is_digital_silence(np.zeros(0, dtype=np.float32))

    def test_a_very_quiet_but_live_mic_is_not_silence(self):
        """A Bluetooth mic in a quiet room peaks around -60 dB; that is working."""
        quiet = np.full(16000, 1e-3, dtype=np.float32)
        assert not is_digital_silence(quiet)

    def test_even_a_single_nonzero_sample_counts_as_signal(self):
        track = np.zeros(16000, dtype=np.float32)
        track[500] = 1e-6
        assert not is_digital_silence(track)
