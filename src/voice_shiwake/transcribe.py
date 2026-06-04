"""AssemblyAI による日本語 ASR + 話者分離。

AssemblyAI の `speaker_labels=True` は事前登録不要の anonymous diarization
（SPEAKER_A / SPEAKER_B / ...）を返す。個人名への紐付けは voiceprint.py で行う。
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class Utterance:
    """1人の発話セグメント。"""

    speaker: str  # SPEAKER_A など、後で個人名に置換される
    start_sec: float
    end_sec: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TranscriptResult:
    utterances: list[Utterance]
    raw_text: str
    language_code: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "language_code": self.language_code,
            "raw_text": self.raw_text,
            "utterances": [u.to_dict() for u in self.utterances],
        }


class AssemblyAIError(RuntimeError):
    pass


def transcribe(audio_path: Path, *, api_key: str | None = None) -> TranscriptResult:
    """AssemblyAI に音声を投げて diarization 付き文字起こしを取得する。

    Args:
        audio_path: 16kHz mono WAV を推奨
        api_key: 省略時は環境変数 ASSEMBLYAI_API_KEY を使用
    """
    api_key = api_key or os.environ.get("ASSEMBLYAI_API_KEY")
    if not api_key:
        raise AssemblyAIError(
            "ASSEMBLYAI_API_KEY が設定されていません。.env を確認してください。"
        )

    try:
        import assemblyai as aai
    except ImportError as e:
        raise AssemblyAIError(
            "assemblyai パッケージが未インストール。`pip install -e .` を実行してください。"
        ) from e

    aai.settings.api_key = api_key

    config = aai.TranscriptionConfig(
        language_code="ja",
        speaker_labels=True,
        punctuate=True,
        format_text=True,
    )

    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(str(audio_path))

    if transcript.status == aai.TranscriptStatus.error:
        raise AssemblyAIError(f"AssemblyAI 失敗: {transcript.error}")

    utterances: list[Utterance] = []
    for utt in transcript.utterances or []:
        utterances.append(
            Utterance(
                speaker=f"SPEAKER_{utt.speaker}",
                start_sec=utt.start / 1000.0,  # ms → sec
                end_sec=utt.end / 1000.0,
                text=utt.text.strip(),
            )
        )

    return TranscriptResult(
        utterances=utterances,
        raw_text=transcript.text or "",
        language_code="ja",
    )


def format_as_dialogue(result: TranscriptResult, speaker_names: dict[str, str] | None = None) -> str:
    """話者ラベル付きの読みやすいテキストに変換する。

    Args:
        result: TranscriptResult
        speaker_names: {"SPEAKER_A": "田中"} のマッピング。未指定の話者はラベルのまま

    Returns:
        "田中: こんにちは。\n佐藤: よろしくお願いします。\n..." 形式
    """
    speaker_names = speaker_names or {}
    lines: list[str] = []
    for utt in result.utterances:
        name = speaker_names.get(utt.speaker, utt.speaker)
        lines.append(f"{name}: {utt.text}")
    return "\n".join(lines)
