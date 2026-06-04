"""videodb 統合アダプタ。

videodb.io は動画用のインデックス＋意味検索プラットフォーム。
本ツールでは「過去の議事録を意味検索する」レイヤとして利用する:

1. process 完了後に動画 + 議事録テキストを videodb に upload
2. index_spoken_words で発話インデックス生成
3. semantic search で過去会議を横断検索

公式 SDK: https://docs.videodb.io/
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class VideodbError(RuntimeError):
    pass


def _get_client(api_key: str | None = None) -> Any:
    """videodb SDK のクライアントを取得（遅延 import）。"""
    api_key = api_key or os.environ.get("VIDEODB_API_KEY")
    if not api_key:
        raise VideodbError(
            "VIDEODB_API_KEY が設定されていません。"
            "https://console.videodb.io/ で取得し .env に設定してください。"
        )
    try:
        import videodb  # type: ignore
    except ImportError as e:
        raise VideodbError(
            "videodb 未インストール。`pip install videodb` を実行してください。"
        ) from e
    return videodb.connect(api_key=api_key)


@dataclass
class IndexResult:
    video_id: str
    collection_id: str
    indexed: bool
    transcript_chars: int


@dataclass
class SearchHit:
    video_id: str
    text: str
    start_sec: float
    end_sec: float
    score: float


def upload_and_index(
    video_path: Path,
    *,
    collection_name: str = "meetings",
    api_key: str | None = None,
) -> IndexResult:
    """動画を videodb にアップロードし、発話インデックスを作成する。

    Args:
        video_path: ローカル動画ファイル
        collection_name: コレクション名（既定 "meetings"）
        api_key: 省略時は VIDEODB_API_KEY 環境変数

    Returns:
        IndexResult: アップロード成功と発話インデックスの状況
    """
    conn = _get_client(api_key)

    # collection を取得 or 作成
    collection = None
    try:
        collections = conn.get_collections()
        for c in collections:
            if getattr(c, "name", None) == collection_name:
                collection = c
                break
    except Exception:  # noqa: BLE001
        # SDK 仕様差を許容
        pass

    if collection is None:
        # 既定 collection を使う（SDK の現行仕様: conn.get_collection() で取得可）
        collection = conn.get_collection()

    # 動画 upload
    video = collection.upload(file_path=str(video_path))

    transcript_chars = 0
    indexed = False
    try:
        # 発話インデックス（音声→テキスト + 意味インデックス）
        video.index_spoken_words()
        indexed = True
        # 任意: transcript を取得して文字数を返す
        try:
            transcript = video.get_transcript_text()
            transcript_chars = len(transcript or "")
        except Exception:  # noqa: BLE001
            transcript_chars = 0
    except Exception as e:  # noqa: BLE001
        raise VideodbError(f"index_spoken_words 失敗: {e}") from e

    return IndexResult(
        video_id=str(video.id),
        collection_id=str(getattr(collection, "id", "default")),
        indexed=indexed,
        transcript_chars=transcript_chars,
    )


def semantic_search(
    query: str,
    *,
    collection_name: str = "meetings",
    api_key: str | None = None,
    limit: int = 5,
) -> list[SearchHit]:
    """過去のインデックス済み動画から query に意味的に近い発話を探す。

    Args:
        query: 自然言語クエリ（例: "予算の話"）
        collection_name: 検索対象 collection
        api_key: 省略時は VIDEODB_API_KEY 環境変数
        limit: 返す件数上限

    Returns:
        SearchHit のリスト（スコア順）
    """
    conn = _get_client(api_key)
    collection = conn.get_collection()

    try:
        results = collection.search(query=query)
    except Exception as e:  # noqa: BLE001
        raise VideodbError(f"semantic_search 失敗: {e}") from e

    hits: list[SearchHit] = []
    # SDK のバージョンによって shots 構造が変わるため defensive に
    raw_shots = []
    if hasattr(results, "get_shots"):
        raw_shots = list(results.get_shots())
    elif hasattr(results, "shots"):
        raw_shots = list(results.shots)

    for shot in raw_shots[:limit]:
        hits.append(
            SearchHit(
                video_id=str(getattr(shot, "video_id", "")),
                text=str(getattr(shot, "text", "")),
                start_sec=float(getattr(shot, "start", 0.0)),
                end_sec=float(getattr(shot, "end", 0.0)),
                score=float(getattr(shot, "score", 0.0)),
            )
        )
    return hits
