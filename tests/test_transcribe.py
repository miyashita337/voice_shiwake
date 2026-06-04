"""文字起こしモジュールの単体テスト。AssemblyAI SDK は mock。"""

from __future__ import annotations

from pathlib import Path

import pytest

from voice_shiwake.transcribe import (
    AssemblyAIError,
    TranscriptResult,
    Utterance,
    format_as_dialogue,
    transcribe,
)


class TestUtterance:
    def test_to_dict(self):
        u = Utterance(speaker="SPEAKER_A", start_sec=1.0, end_sec=2.5, text="こんにちは")
        d = u.to_dict()
        assert d["speaker"] == "SPEAKER_A"
        assert d["start_sec"] == 1.0
        assert d["end_sec"] == 2.5
        assert d["text"] == "こんにちは"


class TestFormatAsDialogue:
    def test_without_mapping_keeps_speaker_label(self):
        result = TranscriptResult(
            utterances=[
                Utterance("SPEAKER_A", 0.0, 1.0, "こんにちは"),
                Utterance("SPEAKER_B", 1.0, 2.0, "よろしく"),
            ],
            raw_text="こんにちは よろしく",
            language_code="ja",
        )
        out = format_as_dialogue(result)
        assert "SPEAKER_A: こんにちは" in out
        assert "SPEAKER_B: よろしく" in out

    def test_with_mapping_substitutes_names(self):
        result = TranscriptResult(
            utterances=[
                Utterance("SPEAKER_A", 0.0, 1.0, "こんにちは"),
                Utterance("SPEAKER_B", 1.0, 2.0, "よろしく"),
            ],
            raw_text="",
            language_code="ja",
        )
        out = format_as_dialogue(result, {"SPEAKER_A": "田中", "SPEAKER_B": "佐藤"})
        assert "田中: こんにちは" in out
        assert "佐藤: よろしく" in out

    def test_partial_mapping_keeps_unmapped_as_label(self):
        result = TranscriptResult(
            utterances=[Utterance("SPEAKER_A", 0.0, 1.0, "a"), Utterance("SPEAKER_C", 1.0, 2.0, "c")],
            raw_text="",
            language_code="ja",
        )
        out = format_as_dialogue(result, {"SPEAKER_A": "田中"})
        assert "田中: a" in out
        assert "SPEAKER_C: c" in out


class TestTranscribe:
    def test_raises_without_api_key(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
        dummy = tmp_path / "x.wav"
        dummy.touch()
        with pytest.raises(AssemblyAIError):
            transcribe(dummy)
