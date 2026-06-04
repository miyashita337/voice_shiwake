"""Slack 投稿モジュールの単体テスト。SDK は mock。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from voice_shiwake.slack import (
    SLACK_MAX_BLOCK_CHARS,
    SlackError,
    _chunk_markdown,
    post_to_slack,
)


class TestChunkMarkdown:
    def test_short_markdown_returns_one_chunk(self):
        md = "# タイトル\n\n短い本文。"
        chunks = _chunk_markdown(md)
        assert len(chunks) == 1
        assert chunks[0] == md

    def test_splits_on_h2_headers(self):
        md = "# T\n\n## A\n本文A\n\n## B\n本文B"
        chunks = _chunk_markdown(md, max_chars=100)
        assert len(chunks) >= 2
        assert any("## A" in c for c in chunks)
        assert any("## B" in c for c in chunks)

    def test_hard_cut_when_paragraph_too_long(self):
        long_para = "あ" * (SLACK_MAX_BLOCK_CHARS * 2 + 100)
        chunks = _chunk_markdown(long_para)
        assert len(chunks) >= 2
        for c in chunks:
            assert len(c) <= SLACK_MAX_BLOCK_CHARS

    def test_empty_input(self):
        assert _chunk_markdown("") == []


class TestPostToSlack:
    def test_raises_when_webhook_missing(self, monkeypatch):
        monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
        with pytest.raises(SlackError):
            post_to_slack("hello")

    @patch("slack_sdk.webhook.WebhookClient")
    def test_posts_with_explicit_webhook(self, mock_client_cls):
        mock_client = MagicMock()
        mock_resp = MagicMock(status_code=200)
        mock_client.send.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        result = post_to_slack("# T\n\n本文", webhook_url="https://example.invalid/hook")

        assert result.ok is True
        assert result.message_count == 1
        mock_client.send.assert_called_once()

    @patch("slack_sdk.webhook.WebhookClient")
    def test_header_prepended_to_first_chunk(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.send.return_value = MagicMock(status_code=200)
        mock_client_cls.return_value = mock_client

        post_to_slack("本文", webhook_url="https://example.invalid/hook", header=":memo: *Title*")

        first_call_text = mock_client.send.call_args.kwargs.get("text") or mock_client.send.call_args.args[0]
        assert ":memo: *Title*" in first_call_text

    @patch("slack_sdk.webhook.WebhookClient")
    def test_non_200_returns_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.send.return_value = MagicMock(status_code=500, body="server error")
        mock_client_cls.return_value = mock_client

        result = post_to_slack("本文", webhook_url="https://example.invalid/hook")
        assert result.ok is False
        assert "500" in (result.error or "")
