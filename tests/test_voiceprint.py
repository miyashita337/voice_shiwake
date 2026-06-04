"""声紋DB ロジック単体テスト。Resemblyzer 実体はロードしない。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voice_shiwake.voiceprint import (
    VoiceprintDB,
    _representative_slice,
    cosine_similarity,
)
from voice_shiwake.transcribe import Utterance


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


class TestVoiceprintDB:
    def test_enroll_and_list_uses_mocked_embedding(self, tmp_path, monkeypatch):
        """compute_embedding を mock して DB ロジックのみ検証。"""
        from voice_shiwake import voiceprint as vp_mod

        fake_emb = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        monkeypatch.setattr(vp_mod, "compute_embedding", lambda p: fake_emb)

        db_path = tmp_path / "test.sqlite3"
        db = VoiceprintDB(db_path=db_path)
        try:
            db.enroll("田中", Path("/dev/null"), note="テスト")
            all_vps = db.list_all()
            assert len(all_vps) == 1
            assert all_vps[0].name == "田中"
            assert all_vps[0].note == "テスト"
            np.testing.assert_array_almost_equal(all_vps[0].embedding, fake_emb)
        finally:
            db.close()

    def test_enroll_same_name_overwrites(self, tmp_path, monkeypatch):
        from voice_shiwake import voiceprint as vp_mod

        embeddings = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ]
        idx = {"i": 0}

        def fake_compute(_p):
            v = embeddings[idx["i"]]
            idx["i"] += 1
            return v

        monkeypatch.setattr(vp_mod, "compute_embedding", fake_compute)

        db = VoiceprintDB(db_path=tmp_path / "t.sqlite3")
        try:
            db.enroll("田中", Path("/dev/null"))
            db.enroll("田中", Path("/dev/null"))
            assert len(db.list_all()) == 1
            np.testing.assert_array_almost_equal(db.list_all()[0].embedding, embeddings[1])
        finally:
            db.close()

    def test_delete_returns_true_when_exists(self, tmp_path, monkeypatch):
        from voice_shiwake import voiceprint as vp_mod
        monkeypatch.setattr(vp_mod, "compute_embedding", lambda p: np.array([1.0]))
        db = VoiceprintDB(db_path=tmp_path / "t.sqlite3")
        try:
            db.enroll("X", Path("/dev/null"))
            assert db.delete("X") is True
            assert db.delete("X") is False
        finally:
            db.close()

    def test_best_match_above_threshold(self, tmp_path, monkeypatch):
        from voice_shiwake import voiceprint as vp_mod
        monkeypatch.setattr(vp_mod, "compute_embedding", lambda p: np.array([1.0, 0.0]))
        db = VoiceprintDB(db_path=tmp_path / "t.sqlite3")
        try:
            db.enroll("田中", Path("/dev/null"))
            name, sim = db.best_match(np.array([1.0, 0.0]), threshold=0.5)
            assert name == "田中"
            assert sim == pytest.approx(1.0, abs=1e-6)
        finally:
            db.close()

    def test_best_match_below_threshold_returns_none(self, tmp_path, monkeypatch):
        from voice_shiwake import voiceprint as vp_mod
        monkeypatch.setattr(vp_mod, "compute_embedding", lambda p: np.array([1.0, 0.0]))
        db = VoiceprintDB(db_path=tmp_path / "t.sqlite3")
        try:
            db.enroll("田中", Path("/dev/null"))
            name, sim = db.best_match(np.array([0.0, 1.0]), threshold=0.5)
            assert name is None
            assert sim == pytest.approx(0.0, abs=1e-6)
        finally:
            db.close()


class TestRepresentativeSlice:
    def test_returns_none_if_no_speaker(self):
        utts = [Utterance("SPEAKER_A", 0.0, 1.0, "x")]
        assert _representative_slice(utts, "SPEAKER_B") is None

    def test_skips_short_utterances_when_long_ones_exist(self):
        utts = [
            Utterance("SPEAKER_A", 0.0, 0.5, "うん"),  # 短い相槌
            Utterance("SPEAKER_A", 1.0, 5.0, "長めの発話"),  # 4秒
            Utterance("SPEAKER_A", 10.0, 15.0, "もうひとつ"),  # 5秒
        ]
        sl = _representative_slice(utts, "SPEAKER_A", target_sec=30.0)
        assert sl is not None
        # 短い相槌は除外され、1.0 秒から始まる
        assert sl.start_sec == 1.0

    def test_falls_back_to_short_when_no_long_utterances(self):
        utts = [Utterance("SPEAKER_A", 0.0, 0.5, "うん")]
        sl = _representative_slice(utts, "SPEAKER_A")
        assert sl is not None
        assert sl.start_sec == 0.0
