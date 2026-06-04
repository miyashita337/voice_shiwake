"""音声抽出ユーティリティの単体テスト。"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from voice_shiwake.audio import (
    AudioSlice,
    FFmpegNotFoundError,
    ensure_ffmpeg,
    extract_audio,
)


class TestAudioSlice:
    def test_duration(self):
        sl = AudioSlice(start_sec=1.0, end_sec=4.5)
        assert sl.duration == pytest.approx(3.5)

    def test_negative_duration_clamped_to_zero(self):
        sl = AudioSlice(start_sec=5.0, end_sec=2.0)
        assert sl.duration == 0.0


class TestEnsureFfmpeg:
    @patch("shutil.which")
    def test_raises_when_missing(self, mock_which):
        mock_which.return_value = None
        with pytest.raises(FFmpegNotFoundError):
            ensure_ffmpeg()

    @patch("shutil.which")
    def test_returns_path_when_present(self, mock_which):
        mock_which.return_value = "/usr/bin/ffmpeg"
        assert ensure_ffmpeg() == "/usr/bin/ffmpeg"


class TestExtractAudio:
    @patch("voice_shiwake.audio.subprocess.run")
    @patch("voice_shiwake.audio.ensure_ffmpeg", return_value="/usr/bin/ffmpeg")
    def test_calls_ffmpeg_with_expected_args(self, _mock_ensure, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        video = tmp_path / "in.mp4"
        video.touch()
        out = tmp_path / "out.wav"
        out.touch()  # extract_audio が「存在チェック」で見るので先に touch

        extract_audio(video, out)

        cmd = mock_run.call_args.args[0]
        assert cmd[0] == "ffmpeg"
        assert "-ar" in cmd and "16000" in cmd
        assert "-ac" in cmd and "1" in cmd
        assert "pcm_s16le" in cmd

    @patch("voice_shiwake.audio.subprocess.run")
    @patch("voice_shiwake.audio.ensure_ffmpeg", return_value="/usr/bin/ffmpeg")
    def test_raises_on_nonzero_returncode(self, _mock_ensure, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=1, stderr="boom")
        video = tmp_path / "in.mp4"
        video.touch()
        out = tmp_path / "out.wav"
        with pytest.raises(RuntimeError, match="ffmpeg 失敗"):
            extract_audio(video, out)
