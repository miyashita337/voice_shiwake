"""声紋（speaker embedding）の登録と照合。

Resemblyzer の ECAPA 系 embedding（256次元）を使い、
コサイン類似度で「SPEAKER_A = 田中」と紐付ける。
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .audio import AudioSlice, slice_wav
from .transcribe import TranscriptResult, Utterance

def _default_db_path() -> Path:
    return Path(os.environ.get("VOICEPRINT_DB_PATH", "./voiceprints.sqlite3"))


def _default_threshold() -> float:
    return float(os.environ.get("VOICEPRINT_THRESHOLD", "0.75"))


# 後方互換のための定数（テストで参照されるケースに備える）
DEFAULT_DB_PATH = _default_db_path()
DEFAULT_THRESHOLD = _default_threshold()


@dataclass
class Voiceprint:
    name: str
    embedding: np.ndarray  # shape: (256,)
    note: str = ""


def _get_encoder() -> Any:
    """Resemblyzer の VoiceEncoder を遅延ロード（重いため）。"""
    try:
        from resemblyzer import VoiceEncoder
    except ImportError as e:
        raise RuntimeError(
            "resemblyzer 未インストール。`pip install -e .` を実行してください。"
        ) from e
    return VoiceEncoder()


def _load_wav_for_resemblyzer(wav_path: Path) -> np.ndarray:
    """Resemblyzer に渡す前処理（16kHz mono float32）。"""
    try:
        from resemblyzer import preprocess_wav
    except ImportError as e:
        raise RuntimeError("resemblyzer 未インストール") from e
    return preprocess_wav(str(wav_path))


def compute_embedding(audio_path: Path) -> np.ndarray:
    """音声ファイルから embedding (256-d) を計算する。"""
    encoder = _get_encoder()
    wav = _load_wav_for_resemblyzer(audio_path)
    return encoder.embed_utterance(wav)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """コサイン類似度（-1〜1）。embedding 既に正規化されていれば内積と等価。"""
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


class VoiceprintDB:
    """SQLite ベースの声紋データベース。"""

    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path) if db_path is not None else _default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS voiceprints (
                name TEXT PRIMARY KEY,
                embedding_json TEXT NOT NULL,
                note TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.commit()

    def enroll(self, name: str, audio_path: Path, note: str = "") -> Voiceprint:
        """声紋を登録（同名は上書き）。"""
        emb = compute_embedding(audio_path)
        self._conn.execute(
            "INSERT OR REPLACE INTO voiceprints (name, embedding_json, note) VALUES (?, ?, ?)",
            (name, json.dumps(emb.tolist()), note),
        )
        self._conn.commit()
        return Voiceprint(name=name, embedding=emb, note=note)

    def list_all(self) -> list[Voiceprint]:
        cur = self._conn.execute("SELECT name, embedding_json, note FROM voiceprints ORDER BY name")
        result = []
        for name, emb_json, note in cur.fetchall():
            emb = np.array(json.loads(emb_json), dtype=np.float32)
            result.append(Voiceprint(name=name, embedding=emb, note=note))
        return result

    def delete(self, name: str) -> bool:
        cur = self._conn.execute("DELETE FROM voiceprints WHERE name = ?", (name,))
        self._conn.commit()
        return cur.rowcount > 0

    def best_match(self, query_embedding: np.ndarray, threshold: float | None = None) -> tuple[str | None, float]:
        if threshold is None:
            threshold = _default_threshold()
        """登録された声紋から最も近いものを返す。

        Returns:
            (名前 or None, cos類似度)。threshold 未満なら名前は None
        """
        best_name: str | None = None
        best_sim: float = -1.0
        for vp in self.list_all():
            sim = cosine_similarity(query_embedding, vp.embedding)
            if sim > best_sim:
                best_sim = sim
                best_name = vp.name
        if best_sim < threshold:
            return None, best_sim
        return best_name, best_sim

    def close(self) -> None:
        self._conn.close()


def _representative_slice(utterances: list[Utterance], speaker: str, target_sec: float = 30.0) -> AudioSlice | None:
    """ある speaker の代表音声の区間（最大 target_sec 秒）を返す。

    戦略: 同一 speaker の utterance を時系列順に並べ、長さ合計が target_sec を
    超えるまでの最初の連続区間を返す。
    2秒未満の短い相槌は候補からスキップする。
    """
    candidates = [u for u in utterances if u.speaker == speaker and (u.end_sec - u.start_sec) >= 2.0]
    if not candidates:
        # 緩めて短いものも許容
        candidates = [u for u in utterances if u.speaker == speaker]
    if not candidates:
        return None

    accumulated = 0.0
    start = candidates[0].start_sec
    end = candidates[0].end_sec
    for utt in candidates:
        accumulated += utt.end_sec - utt.start_sec
        end = utt.end_sec
        if accumulated >= target_sec:
            break
    return AudioSlice(start_sec=start, end_sec=end)


def identify_speakers(
    transcript: TranscriptResult,
    source_wav: Path,
    db: VoiceprintDB,
    *,
    threshold: float | None = None,
    representative_sec: float = 30.0,
) -> dict[str, str]:
    if threshold is None:
        threshold = _default_threshold()
    """transcript の各 SPEAKER_X を登録済み声紋と照合して個人名にマッピング。

    Args:
        transcript: AssemblyAI の結果
        source_wav: 会議全体の音声ファイル（16kHz mono WAV）
        db: 声紋DB
        threshold: cos類似度の閾値
        representative_sec: 各話者から抽出する代表音声の長さ

    Returns:
        {"SPEAKER_A": "田中太郎", "SPEAKER_B": "[Unknown]"} のマッピング
    """
    speakers = sorted({u.speaker for u in transcript.utterances})
    mapping: dict[str, str] = {}

    if not db.list_all():
        # 登録ゼロなら何もしない
        return {s: s for s in speakers}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for speaker in speakers:
            slice_ = _representative_slice(transcript.utterances, speaker, representative_sec)
            if slice_ is None or slice_.duration < 1.0:
                mapping[speaker] = f"{speaker}[音声不足]"
                continue
            slice_wav_path = tmp / f"{speaker}.wav"
            slice_wav(source_wav, slice_, slice_wav_path)
            emb = compute_embedding(slice_wav_path)
            name, sim = db.best_match(emb, threshold=threshold)
            if name is None:
                mapping[speaker] = f"[Unknown:{sim:.2f}]"
            else:
                mapping[speaker] = name
    return mapping
