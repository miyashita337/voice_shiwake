"""声紋DB ロジック単体テスト（multi-sample 対応版）。Resemblyzer 実体はロードしない。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from voice_shiwake.transcribe import Utterance
from voice_shiwake.voiceprint import (
    VoiceprintDB,
    _representative_slice,
    cosine_similarity,
)


# ---------------------------------------------------------------------------
# 基本ユーティリティ
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# multi-sample enrollment
# ---------------------------------------------------------------------------


class TestMultiSampleEnroll:
    def test_default_mode_appends_samples(self, tmp_path, monkeypatch):
        """同名 enroll は追加モード（sample_id が増えていく）。"""
        from voice_shiwake import voiceprint as vp_mod

        embs = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.9, 0.1], dtype=np.float32),
            np.array([0.95, 0.05], dtype=np.float32),
        ]
        idx = {"i": 0}

        def fake_compute(_p):
            v = embs[idx["i"]]
            idx["i"] += 1
            return v

        monkeypatch.setattr(vp_mod, "compute_embedding", fake_compute)

        db = VoiceprintDB(db_path=tmp_path / "t.sqlite3")
        try:
            v1 = db.enroll("田中", Path("/dev/null"), source="meeting-1")
            v2 = db.enroll("田中", Path("/dev/null"), source="meeting-2")
            v3 = db.enroll("田中", Path("/dev/null"), source="meeting-3")
            assert v1.sample_id == 0
            assert v2.sample_id == 1
            assert v3.sample_id == 2
            assert len(db.list_all()) == 3
        finally:
            db.close()

    def test_replace_clears_existing(self, tmp_path, monkeypatch):
        from voice_shiwake import voiceprint as vp_mod
        monkeypatch.setattr(vp_mod, "compute_embedding", lambda p: np.array([1.0, 0.0], dtype=np.float32))

        db = VoiceprintDB(db_path=tmp_path / "t.sqlite3")
        try:
            db.enroll("田中", Path("/dev/null"))
            db.enroll("田中", Path("/dev/null"))
            assert len(db.list_all()) == 2
            db.enroll("田中", Path("/dev/null"), replace=True)
            assert len(db.list_all()) == 1
        finally:
            db.close()

    def test_stats_aggregates_per_name(self, tmp_path, monkeypatch):
        from voice_shiwake import voiceprint as vp_mod
        monkeypatch.setattr(vp_mod, "compute_embedding", lambda p: np.array([1.0, 0.0], dtype=np.float32))

        db = VoiceprintDB(db_path=tmp_path / "t.sqlite3")
        try:
            db.enroll("田中", Path("/dev/null"), source="weekly-1")
            db.enroll("田中", Path("/dev/null"), source="weekly-2")
            db.enroll("佐藤", Path("/dev/null"), source="weekly-1")
            stats = db.stats()
            stats_by_name = {s.name: s for s in stats}
            assert stats_by_name["田中"].sample_count == 2
            assert stats_by_name["佐藤"].sample_count == 1
            assert "weekly-1" in stats_by_name["田中"].sources
            assert "weekly-2" in stats_by_name["田中"].sources
        finally:
            db.close()


# ---------------------------------------------------------------------------
# best_match (centroid + max similarity)
# ---------------------------------------------------------------------------


class TestBestMatchMultiSample:
    def test_max_similarity_picks_best_per_name(self, tmp_path, monkeypatch):
        """田中に2サンプル、佐藤に1サンプルあり、クエリは田中の片方に近い。"""
        from voice_shiwake import voiceprint as vp_mod

        # 田中: [1,0,0] と [0.9,0.4,0] / 佐藤: [0,1,0]
        embs = [
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.9, 0.4, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        ]
        idx = {"i": 0}

        def fake_compute(_p):
            v = embs[idx["i"]]
            idx["i"] += 1
            return v

        monkeypatch.setattr(vp_mod, "compute_embedding", fake_compute)

        db = VoiceprintDB(db_path=tmp_path / "t.sqlite3")
        try:
            db.enroll("田中", Path("/dev/null"))
            db.enroll("田中", Path("/dev/null"))
            db.enroll("佐藤", Path("/dev/null"))

            # 田中の1個目に完全一致するクエリ
            name, sim = db.best_match(np.array([1.0, 0.0, 0.0]), threshold=0.5)
            assert name == "田中"
            assert sim == pytest.approx(1.0, abs=1e-6)
        finally:
            db.close()

    def test_match_detail_returns_centroid(self, tmp_path, monkeypatch):
        from voice_shiwake import voiceprint as vp_mod

        embs = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ]
        idx = {"i": 0}
        monkeypatch.setattr(vp_mod, "compute_embedding", lambda _p: embs[idx.setdefault("i", 0)] if (idx.update({"i": idx["i"] + 1}) or True) else embs[0])

        # シンプルに直接 enroll_embedding
        db = VoiceprintDB(db_path=tmp_path / "t.sqlite3")
        try:
            db.enroll_embedding("田中", np.array([1.0, 0.0], dtype=np.float32))
            db.enroll_embedding("田中", np.array([0.0, 1.0], dtype=np.float32))
            # centroid = [0.5, 0.5]、クエリ = [0.7, 0.7]
            result = db.match_detail(np.array([0.7, 0.7], dtype=np.float32))
            assert result.name == "田中"
            assert result.sample_count == 2
            # max は [0.7,0.7] と [1,0] or [0,1] の類似度 = 0.707
            assert result.max_similarity == pytest.approx(0.7071, abs=1e-3)
            # centroid 類似度は 1.0 (向きが同じ)
            assert result.centroid_similarity == pytest.approx(1.0, abs=1e-3)
        finally:
            db.close()

    def test_empty_db_returns_none(self, tmp_path):
        db = VoiceprintDB(db_path=tmp_path / "t.sqlite3")
        try:
            name, sim = db.best_match(np.array([1.0, 0.0]), threshold=0.5)
            assert name is None
            assert sim == -1.0
        finally:
            db.close()


# ---------------------------------------------------------------------------
# correction_history
# ---------------------------------------------------------------------------


class TestCorrectionHistory:
    def test_record_and_read_history(self, tmp_path):
        db = VoiceprintDB(db_path=tmp_path / "t.sqlite3")
        try:
            db.record_correction(
                source="weekly-2026-06-01",
                original_label="SPEAKER_A",
                new_label="山田太郎",
                max_similarity=0.45,
                note="低類似度で誤識別、訂正",
            )
            db.record_correction(
                source="weekly-2026-06-01",
                original_label="SPEAKER_B",
                new_label="小川",
            )
            history = db.correction_history()
            assert len(history) == 2
            # 新しいものが先頭
            assert history[0]["new_label"] == "小川"
            assert history[1]["new_label"] == "山田太郎"
            assert history[1]["max_similarity"] == pytest.approx(0.45)
        finally:
            db.close()


# ---------------------------------------------------------------------------
# legacy schema migration
# ---------------------------------------------------------------------------


class TestLegacyMigration:
    def test_migrates_legacy_schema(self, tmp_path):
        """旧スキーマ（name PRIMARY KEY）から新スキーマへ自動移行する。"""
        db_path = tmp_path / "legacy.sqlite3"

        # 旧スキーマを直接作成
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE voiceprints ("
            "name TEXT PRIMARY KEY, embedding_json TEXT, note TEXT, created_at TEXT"
            ")"
        )
        import json as _json
        conn.execute(
            "INSERT INTO voiceprints (name, embedding_json, note) VALUES (?, ?, ?)",
            ("田中", _json.dumps([1.0, 0.0]), "古いノート"),
        )
        conn.commit()
        conn.close()

        # 新クラスで開く → マイグレーション走るはず
        db = VoiceprintDB(db_path=db_path)
        try:
            all_vps = db.list_all()
            assert len(all_vps) == 1
            assert all_vps[0].name == "田中"
            assert all_vps[0].sample_id == 0
            assert all_vps[0].source == "legacy"
            np.testing.assert_array_almost_equal(all_vps[0].embedding, [1.0, 0.0])
        finally:
            db.close()


# ---------------------------------------------------------------------------
# representative slice (回帰テスト)
# ---------------------------------------------------------------------------


class TestRepresentativeSlice:
    def test_returns_none_if_no_speaker(self):
        utts = [Utterance("SPEAKER_A", 0.0, 1.0, "x")]
        assert _representative_slice(utts, "SPEAKER_B") is None

    def test_skips_short_utterances_when_long_ones_exist(self):
        utts = [
            Utterance("SPEAKER_A", 0.0, 0.5, "うん"),
            Utterance("SPEAKER_A", 1.0, 5.0, "長めの発話"),
            Utterance("SPEAKER_A", 10.0, 15.0, "もうひとつ"),
        ]
        sl = _representative_slice(utts, "SPEAKER_A", target_sec=30.0)
        assert sl is not None
        assert sl.start_sec == 1.0

    def test_falls_back_to_short_when_no_long_utterances(self):
        utts = [Utterance("SPEAKER_A", 0.0, 0.5, "うん")]
        sl = _representative_slice(utts, "SPEAKER_A")
        assert sl is not None
        assert sl.start_sec == 0.0
