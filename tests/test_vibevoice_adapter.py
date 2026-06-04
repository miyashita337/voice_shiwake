"""VibeVoice アダプタの単体テスト。torch/transformers 実体はロードしない。"""

from __future__ import annotations

from pathlib import Path

import pytest

from voice_shiwake.vibevoice_adapter import (
    VibeVoiceError,
    check_availability,
    markdown_to_speech_text,
)


class TestCheckAvailability:
    def test_no_model_path_returns_false(self, monkeypatch):
        monkeypatch.delenv("VIBEVOICE_MODEL_PATH", raising=False)
        ok, reason = check_availability()
        assert ok is False
        assert "VIBEVOICE_MODEL_PATH" in reason

    def test_nonexistent_path_returns_false(self, monkeypatch):
        monkeypatch.setenv("VIBEVOICE_MODEL_PATH", "/nonexistent/path/xyz")
        ok, reason = check_availability()
        assert ok is False
        assert "存在しない" in reason

    def test_existing_path_checks_libs(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VIBEVOICE_MODEL_PATH", str(tmp_path))
        ok, reason = check_availability()
        # torch / transformers がこの env に無ければ False、あれば True
        if not ok:
            assert "torch" in reason or "transformers" in reason
        else:
            assert reason == "OK"


class TestMarkdownToSpeechText:
    def test_strips_h1_marker(self):
        result = markdown_to_speech_text("# タイトル")
        assert result == "タイトル。"

    def test_strips_h2_marker(self):
        result = markdown_to_speech_text("## 議題サマリー")
        assert result == "議題サマリー。"

    def test_removes_bullet_marker(self):
        result = markdown_to_speech_text("- 項目1\n- 項目2")
        assert "項目1" in result
        assert "項目2" in result
        assert "-" not in result.split("\n")[0]

    def test_removes_table_separator_row(self):
        md = "| 担当 | 内容 |\n|---|---|\n| 田中 | 資料作成 |"
        result = markdown_to_speech_text(md)
        assert "---" not in result
        assert "田中" in result
        assert "資料作成" in result

    def test_converts_table_row_to_japanese_punctuation(self):
        md = "| 田中 | 資料作成 | 来週まで |"
        result = markdown_to_speech_text(md)
        assert "田中、資料作成、来週まで" in result

    def test_unwraps_markdown_link(self):
        md = "詳細は [公式ドキュメント](https://example.com) を参照"
        result = markdown_to_speech_text(md)
        assert "公式ドキュメント" in result
        assert "https://example.com" not in result

    def test_truncates_long_text(self):
        long = "あ" * 5000
        result = markdown_to_speech_text(long, max_chars=100)
        assert len(result) <= 200  # truncate + 「以下省略」 を加味
        assert "以下省略" in result


class TestVibeVoiceBackendUnavailable:
    """check_availability で弾かれるケースの動作確認。"""

    def test_init_raises_when_unavailable(self, monkeypatch):
        monkeypatch.delenv("VIBEVOICE_MODEL_PATH", raising=False)
        from voice_shiwake.vibevoice_adapter import VibeVoiceBackend

        with pytest.raises(VibeVoiceError):
            VibeVoiceBackend()
