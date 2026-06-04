"""議事録整形モジュールの単体テスト。Anthropic SDK は mock。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from voice_shiwake.minutes import ClaudeError, summarize_to_minutes


class TestSummarizeToMinutes:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ClaudeError):
            summarize_to_minutes("田中: こんにちは")

    @patch("anthropic.Anthropic")
    def test_returns_markdown_from_claude(self, mock_anthropic_cls):
        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "# テスト議事録\n\n## 議題サマリー\nテスト内容"

        mock_message = MagicMock()
        mock_message.content = [mock_block]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message
        mock_anthropic_cls.return_value = mock_client

        result = summarize_to_minutes(
            "田中: こんにちは\n佐藤: よろしく",
            title="テスト議事録",
            api_key="sk-test-fake",
        )

        assert "# テスト議事録" in result.markdown
        assert result.title == "テスト議事録"
        assert result.input_chars > 0
        mock_client.messages.create.assert_called_once()
