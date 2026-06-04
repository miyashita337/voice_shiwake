"""Claude API による議事録整形。

話者ラベル付きの生文字起こしを、要約・決定事項・ActionItem 構造の
Markdown 議事録に変換する。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

SYSTEM_PROMPT = """あなたは優秀な議事録作成アシスタントです。
入力された会議の文字起こしから、以下の構成で Markdown 議事録を作成してください。

# {title}

- 日時: {datetime}
- 参加者: （文字起こしから登場した話者名を列挙）

## 議題サマリー
（3〜5行で会議全体の論点を要約）

## 主な議論
（時系列に沿って論点を箇条書き。話者名を「（田中）」のように補足）

## 決定事項
（明確に合意・決定された事項のみ。無ければ「特になし」）

## ActionItem
（誰が・何を・いつまでに、の3点セット。無ければ「特になし」）
| 担当 | 内容 | 期限 |
|---|---|---|

## メモ
（補足・未解決事項）

ルール:
- 推測で情報を補わず、文字起こしに無いことは書かない
- 話者名が「SPEAKER_A」など匿名のままなら、その表記で続ける
- [Unknown:数値] は「不明話者」と表記
- 冗長な相槌は省略してよい
"""


@dataclass
class MinutesResult:
    markdown: str
    model: str
    input_chars: int
    title: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "model": self.model,
            "input_chars": self.input_chars,
            "markdown": self.markdown,
        }


class ClaudeError(RuntimeError):
    pass


def summarize_to_minutes(
    dialogue_text: str,
    *,
    title: str = "会議議事録",
    meeting_datetime: datetime | None = None,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> MinutesResult:
    """Claude API で議事録を生成する。

    Args:
        dialogue_text: "話者名: 発話\n..." 形式
        title: 議事録タイトル
        meeting_datetime: 会議日時（未指定なら現在時刻）
        model: Claude モデル名
        api_key: 省略時は ANTHROPIC_API_KEY 環境変数

    Returns:
        MinutesResult
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ClaudeError(
            "ANTHROPIC_API_KEY が設定されていません。.env を確認してください。"
        )

    try:
        from anthropic import Anthropic
    except ImportError as e:
        raise ClaudeError("anthropic 未インストール") from e

    meeting_datetime = meeting_datetime or datetime.now()
    sys_prompt = SYSTEM_PROMPT.format(
        title=title,
        datetime=meeting_datetime.strftime("%Y-%m-%d %H:%M"),
    )

    client = Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=sys_prompt,
        messages=[
            {
                "role": "user",
                "content": f"以下の文字起こしを議事録にしてください。\n\n{dialogue_text}",
            }
        ],
    )

    # SDK の content は list[ContentBlock]
    text_parts: list[str] = []
    for block in message.content:
        if getattr(block, "type", None) == "text":
            text_parts.append(block.text)
    markdown = "".join(text_parts).strip()

    return MinutesResult(
        markdown=markdown,
        model=model,
        input_chars=len(dialogue_text),
        title=title,
    )
