"""Slack への議事録投稿。

2つの投稿方式をサポート:
  1. Incoming Webhook: 単純な投稿のみ、スレッド化不可
  2. Bot Token (xoxb-...) + chat.postMessage: スレッド化可能

長文議事録は Slack の 1メッセージ上限（およそ 3000〜4000 文字）を考慮して
見出し単位で分割投稿する。Bot Token 経由なら 2通目以降は親メッセージへの
スレッド返信として投稿される（「議事録さん」と同じ方式）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

SLACK_MAX_BLOCK_CHARS = 2900  # Slack の block 上限より安全側


@dataclass
class SlackPostResult:
    ok: bool
    message_count: int
    error: str | None = None
    thread_ts: str | None = None  # Bot Token 方式の場合の親メッセージ ts
    channel: str | None = None


class SlackError(RuntimeError):
    pass


def _chunk_markdown(markdown: str, max_chars: int = SLACK_MAX_BLOCK_CHARS) -> list[str]:
    """Markdown を見出し境界優先で分割する。

    1. まず `\n## ` で分割
    2. 各セクションが max_chars を超える場合は段落（空行）で再分割
    3. それでも超える場合は max_chars でハードカット
    """
    sections: list[str] = []
    current = ""
    for line in markdown.splitlines(keepends=True):
        if line.startswith("## ") and current:
            sections.append(current)
            current = line
        else:
            current += line
    if current:
        sections.append(current)

    chunks: list[str] = []
    for section in sections:
        if len(section) <= max_chars:
            chunks.append(section)
            continue
        # 段落分割
        paragraphs = section.split("\n\n")
        buf = ""
        for para in paragraphs:
            if len(buf) + len(para) + 2 <= max_chars:
                buf += (para + "\n\n") if para else ""
            else:
                if buf:
                    chunks.append(buf.rstrip())
                if len(para) > max_chars:
                    # ハードカット
                    for i in range(0, len(para), max_chars):
                        chunks.append(para[i : i + max_chars])
                    buf = ""
                else:
                    buf = para + "\n\n"
        if buf.strip():
            chunks.append(buf.rstrip())
    return [c for c in chunks if c.strip()]


def post_to_slack(
    markdown: str,
    *,
    webhook_url: str | None = None,
    header: str | None = None,
) -> SlackPostResult:
    """Markdown 議事録を Slack に投稿する。

    Args:
        markdown: 議事録本文
        webhook_url: 省略時は SLACK_WEBHOOK_URL 環境変数
        header: 先頭メッセージに付加するヘッダ（任意）

    Returns:
        SlackPostResult
    """
    webhook_url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        raise SlackError(
            "SLACK_WEBHOOK_URL が設定されていません。.env を確認してください。"
        )

    try:
        from slack_sdk.webhook import WebhookClient
    except ImportError as e:
        raise SlackError("slack_sdk 未インストール") from e

    client = WebhookClient(webhook_url)
    chunks = _chunk_markdown(markdown)
    if header:
        chunks[0] = f"{header}\n\n{chunks[0]}"

    sent = 0
    for i, chunk in enumerate(chunks):
        try:
            resp = client.send(text=chunk)
        except Exception as e:  # noqa: BLE001 - SDK 側の多様な例外を握って報告
            return SlackPostResult(
                ok=False,
                message_count=sent,
                error=f"chunk {i} 送信失敗: {e}",
            )
        if resp.status_code != 200:
            return SlackPostResult(
                ok=False,
                message_count=sent,
                error=f"chunk {i} status={resp.status_code} body={resp.body}",
            )
        sent += 1
    return SlackPostResult(ok=True, message_count=sent)


def post_to_slack_threaded(
    markdown: str,
    *,
    bot_token: str | None = None,
    channel: str | None = None,
    parent_text: str | None = None,
    header: str | None = None,
) -> SlackPostResult:
    """Bot Token + chat.postMessage でスレッド化投稿する。

    動作:
      1. parent_text（指定なければ header） を **親メッセージ**として投稿
      2. 議事録本文を chunk 分割し、**親への返信スレッド**として連続投稿
      3. 「議事録さん」と同じ見た目

    Args:
        markdown: 議事録本文
        bot_token: 省略時は SLACK_BOT_TOKEN 環境変数
        channel: 投稿先 channel ID (C0XXX) or 名前 (#meeting-minutes)
                 省略時は SLACK_CHANNEL_ID
        parent_text: 親メッセージのテキスト（任意）。指定なし時は header を流用、
                     どちらも無ければ「📝 議事録」とする
        header: 互換用（parent_text 未指定時のフォールバック）

    Returns:
        SlackPostResult（thread_ts, channel が埋まる）
    """
    bot_token = bot_token or os.environ.get("SLACK_BOT_TOKEN")
    if not bot_token:
        raise SlackError(
            "SLACK_BOT_TOKEN が設定されていません。.env に xoxb-... を設定してください。"
        )
    channel = channel or os.environ.get("SLACK_CHANNEL_ID")
    if not channel:
        raise SlackError(
            "SLACK_CHANNEL_ID が設定されていません。.env に C0XXX 形式の channel ID を設定してください。"
        )

    try:
        from slack_sdk import WebClient
        from slack_sdk.errors import SlackApiError
    except ImportError as e:
        raise SlackError("slack_sdk 未インストール") from e

    client = WebClient(token=bot_token)

    # 親メッセージ
    parent = parent_text or header or "📝 議事録"
    try:
        parent_resp = client.chat_postMessage(channel=channel, text=parent)
    except SlackApiError as e:
        err = e.response.get("error", str(e))
        return SlackPostResult(
            ok=False,
            message_count=0,
            error=f"親メッセージ投稿失敗: {err}",
        )

    thread_ts = parent_resp["ts"]
    posted_channel = parent_resp["channel"]
    sent = 1  # 親メッセージカウント

    # 議事録本文をスレッドに連投
    chunks = _chunk_markdown(markdown)
    for i, chunk in enumerate(chunks):
        try:
            client.chat_postMessage(
                channel=posted_channel,
                thread_ts=thread_ts,
                text=chunk,
            )
        except SlackApiError as e:
            err = e.response.get("error", str(e))
            return SlackPostResult(
                ok=False,
                message_count=sent,
                error=f"chunk {i} スレッド投稿失敗: {err}",
                thread_ts=thread_ts,
                channel=posted_channel,
            )
        sent += 1

    return SlackPostResult(
        ok=True,
        message_count=sent,
        thread_ts=thread_ts,
        channel=posted_channel,
    )


def post_to_slack_auto(
    markdown: str,
    *,
    header: str | None = None,
    parent_text: str | None = None,
) -> SlackPostResult:
    """環境変数に応じて Bot Token 経路 or Webhook 経路を自動選択する。

    優先順:
      1. SLACK_BOT_TOKEN + SLACK_CHANNEL_ID 両方あれば → スレッド化（推奨）
      2. SLACK_WEBHOOK_URL があれば → Webhook フラット投稿
      3. どちらも無ければ SlackError
    """
    has_bot = bool(os.environ.get("SLACK_BOT_TOKEN")) and bool(
        os.environ.get("SLACK_CHANNEL_ID")
    )
    has_webhook = bool(os.environ.get("SLACK_WEBHOOK_URL"))

    if has_bot:
        return post_to_slack_threaded(
            markdown, header=header, parent_text=parent_text
        )
    if has_webhook:
        return post_to_slack(markdown, header=header)
    raise SlackError(
        "Slack 投稿先未設定。.env に SLACK_BOT_TOKEN+SLACK_CHANNEL_ID "
        "（スレッド化）または SLACK_WEBHOOK_URL（フラット）を設定してください。"
    )
