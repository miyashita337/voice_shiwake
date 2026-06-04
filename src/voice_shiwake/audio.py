"""音声抽出・切り出しユーティリティ。

動画 → 16kHz mono WAV を ffmpeg で抽出する。
AssemblyAI / Resemblyzer の両方で扱いやすい単一フォーマットに正規化する。
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class FFmpegNotFoundError(RuntimeError):
    """ffmpeg が PATH に無い場合の例外。"""


def ensure_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise FFmpegNotFoundError(
            "ffmpeg が見つかりません。macOS: brew install ffmpeg / Ubuntu: apt install ffmpeg"
        )
    return path


def extract_audio(video_path: Path, output_path: Path, sample_rate: int = 16000) -> Path:
    """動画から 16kHz mono WAV を抽出する。

    Args:
        video_path: 入力動画
        output_path: 出力 WAV
        sample_rate: 既定 16kHz（AssemblyAI / Resemblyzer 共通対応）

    Returns:
        出力 WAV のパス
    """
    ensure_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg 失敗 (returncode={result.returncode})\nstderr:\n{result.stderr}"
        )
    if not output_path.exists():
        raise RuntimeError(f"ffmpeg 完了したが出力ファイルが存在しない: {output_path}")
    return output_path


@dataclass
class AudioSlice:
    """音声の一区間（秒指定）。"""

    start_sec: float
    end_sec: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


def slice_wav(src_wav: Path, slice_: AudioSlice, dst_wav: Path) -> Path:
    """WAV を秒指定で切り出す。

    Resemblyzer に渡す代表音声の生成に使う。
    """
    ensure_ffmpeg()
    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src_wav),
        "-ss",
        f"{slice_.start_sec:.3f}",
        "-to",
        f"{slice_.end_sec:.3f}",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        str(dst_wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg slice 失敗 (returncode={result.returncode})\nstderr:\n{result.stderr}"
        )
    return dst_wav
