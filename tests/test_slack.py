"""Slack 投稿モジュールの単体テスト。SDK は mock。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from voice_shiwake.slack import (
    SLACK_MAX_BLOCK_CHARS,
    SlackError,
    _chunk_markdown,
    post_to_slack,
    post_to_slack_auto,
    post_to_slack_threaded,
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


class TestPostToSlackThreaded:
    def test_raises_when_bot_token_missing(self, monkeypatch):
        monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
        monkeypatch.setenv("SLACK_CHANNEL_ID", "C123")
        with pytest.raises(SlackError, match="SLACK_BOT_TOKEN"):
            post_to_slack_threaded("hello")

    def test_raises_when_channel_missing(self, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-fake")
        monkeypatch.delenv("SLACK_CHANNEL_ID", raising=False)
        with pytest.raises(SlackError, match="SLACK_CHANNEL_ID"):
            post_to_slack_threaded("hello")

    @patch("slack_sdk.WebClient")
    def test_posts_parent_then_thread_replies(self, mock_web_cls, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-fake")
        monkeypatch.setenv("SLACK_CHANNEL_ID", "C123")

        mock_client = MagicMock()
        # 親メッセージのレスポンス: ts と channel を返す
        mock_client.chat_postMessage.return_value = {"ts": "1234567890.000100", "channel": "C123"}
        mock_web_cls.return_value = mock_client

        result = post_to_slack_threaded(
            "# T\n\n## A\n本文A\n\n## B\n本文B",
            parent_text=":memo: *タイトル*",
        )

        assert result.ok is True
        assert result.thread_ts == "1234567890.000100"
        assert result.channel == "C123"
        # 親 + 子chunks（実数を assert すると _chunk_markdown 仕様変更で壊れるので >= 2 を確認）
        assert result.message_count >= 2
        # 親メッセージは thread_ts なしで投稿
        first_call = mock_client.chat_postMessage.call_args_list[0]
        assert "thread_ts" not in first_call.kwargs
        assert first_call.kwargs["text"] == ":memo: *タイトル*"
        # 2通目以降は thread_ts="1234567890.000100" 指定
        second_call = mock_client.chat_postMessage.call_args_list[1]
        assert second_call.kwargs["thread_ts"] == "1234567890.000100"

    @patch("slack_sdk.WebClient")
    def test_returns_error_on_slack_api_error(self, mock_web_cls, monkeypatch):
        from slack_sdk.errors import SlackApiError

        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-fake")
        monkeypatch.setenv("SLACK_CHANNEL_ID", "C123")

        mock_client = MagicMock()
        err_response = MagicMock()
        err_response.get.return_value = "channel_not_found"
        mock_client.chat_postMessage.side_effect = SlackApiError("test", err_response)
        mock_web_cls.return_value = mock_client

        result = post_to_slack_threaded("hello")
        assert result.ok is False
        assert "channel_not_found" in (result.error or "")


class TestPostToSlackAuto:
    def test_prefers_bot_token_when_both_set(self, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-fake")
        monkeypatch.setenv("SLACK_CHANNEL_ID", "C123")
        monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://example.invalid/hook")

        with patch("voice_shiwake.slack.post_to_slack_threaded") as mock_threaded, \
             patch("voice_shiwake.slack.post_to_slack") as mock_webhook:
            mock_threaded.return_value = MagicMock(ok=True, message_count=3, thread_ts="ts", channel="C123")
            post_to_slack_auto("test", header="h")
            mock_threaded.assert_called_once()
            mock_webhook.assert_not_called()

    def test_falls_back_to_webhook_when_no_bot(self, monkeypatch):
        monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
        monkeypatch.delenv("SLACK_CHANNEL_ID", raising=False)
        monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://example.invalid/hook")

        with patch("voice_shiwake.slack.post_to_slack_threaded") as mock_threaded, \
             patch("voice_shiwake.slack.post_to_slack") as mock_webhook:
            mock_webhook.return_value = MagicMock(ok=True, message_count=2, thread_ts=None)
            post_to_slack_auto("test", header="h")
            mock_threaded.assert_not_called()
            mock_webhook.assert_called_once()

    def test_raises_when_nothing_set(self, monkeypatch):
        monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
        monkeypatch.delenv("SLACK_CHANNEL_ID", raising=False)
        monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
        with pytest.raises(SlackError, match="Slack 投稿先未設定"):
            post_to_slack_auto("test")
