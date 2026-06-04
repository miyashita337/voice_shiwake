"""videodb アダプタの単体テスト（SDK は mock）。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from voice_shiwake.videodb_adapter import (
    VideodbError,
    semantic_search,
    upload_and_index,
)


class TestUploadAndIndex:
    def test_raises_without_api_key(self, monkeypatch, tmp_path):
        monkeypatch.delenv("VIDEODB_API_KEY", raising=False)
        video = tmp_path / "v.mp4"
        video.touch()
        with pytest.raises(VideodbError, match="VIDEODB_API_KEY"):
            upload_and_index(video)

    def test_calls_index_spoken_words_on_uploaded_video(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VIDEODB_API_KEY", "fake-key")

        # videodb モジュールを mock として注入
        mock_videodb = MagicMock()
        mock_video = MagicMock()
        mock_video.id = "video-123"
        mock_video.get_transcript_text.return_value = "テスト文字起こし"
        mock_collection = MagicMock()
        mock_collection.id = "col-1"
        mock_collection.name = "meetings"
        mock_collection.upload.return_value = mock_video

        mock_conn = MagicMock()
        mock_conn.get_collections.return_value = [mock_collection]
        mock_conn.get_collection.return_value = mock_collection
        mock_videodb.connect.return_value = mock_conn

        monkeypatch.setitem(__import__("sys").modules, "videodb", mock_videodb)

        video = tmp_path / "v.mp4"
        video.touch()
        result = upload_and_index(video)

        # get_collections で見つかった collection が再利用される（get_collection 不要）
        mock_collection.upload.assert_called_once_with(file_path=str(video))
        mock_video.index_spoken_words.assert_called_once()
        assert result.video_id == "video-123"
        assert result.indexed is True
        assert result.transcript_chars > 0

    def test_propagates_index_failure(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VIDEODB_API_KEY", "fake-key")
        mock_videodb = MagicMock()
        mock_video = MagicMock()
        mock_video.index_spoken_words.side_effect = RuntimeError("boom")
        mock_collection = MagicMock()
        mock_collection.upload.return_value = mock_video
        mock_conn = MagicMock()
        mock_conn.get_collection.return_value = mock_collection
        mock_conn.get_collections.return_value = []
        mock_videodb.connect.return_value = mock_conn
        monkeypatch.setitem(__import__("sys").modules, "videodb", mock_videodb)

        video = tmp_path / "v.mp4"
        video.touch()
        with pytest.raises(VideodbError, match="index_spoken_words"):
            upload_and_index(video)


class TestSemanticSearch:
    def test_returns_hits_from_shots(self, monkeypatch):
        monkeypatch.setenv("VIDEODB_API_KEY", "fake-key")

        mock_videodb = MagicMock()
        # shots は属性アクセスで揃える
        shot1 = MagicMock()
        shot1.video_id = "v1"
        shot1.text = "予算について"
        shot1.start = 10.0
        shot1.end = 15.0
        shot1.score = 0.92

        mock_search_result = MagicMock()
        mock_search_result.get_shots.return_value = [shot1]

        mock_collection = MagicMock()
        mock_collection.search.return_value = mock_search_result

        mock_conn = MagicMock()
        mock_conn.get_collection.return_value = mock_collection
        mock_videodb.connect.return_value = mock_conn

        monkeypatch.setitem(__import__("sys").modules, "videodb", mock_videodb)

        hits = semantic_search("予算")
        assert len(hits) == 1
        assert hits[0].video_id == "v1"
        assert hits[0].text == "予算について"
        assert hits[0].score == pytest.approx(0.92)

    def test_returns_empty_when_no_shots(self, monkeypatch):
        monkeypatch.setenv("VIDEODB_API_KEY", "fake-key")
        mock_videodb = MagicMock()
        mock_search_result = MagicMock()
        mock_search_result.get_shots.return_value = []
        mock_collection = MagicMock()
        mock_collection.search.return_value = mock_search_result
        mock_conn = MagicMock()
        mock_conn.get_collection.return_value = mock_collection
        mock_videodb.connect.return_value = mock_conn
        monkeypatch.setitem(__import__("sys").modules, "videodb", mock_videodb)

        assert semantic_search("無関係なクエリ") == []
