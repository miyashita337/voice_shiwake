"""Slack Incoming Webhook への議事録投稿。

Slack の 1メッセージ上限（およそ 3000〜4000 文字）を考慮して、
長い議事録は見出し単位で分割投稿する。
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
