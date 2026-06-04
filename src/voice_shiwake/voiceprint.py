"""声紋（speaker embedding）の登録と照合（multi-sample 対応）。

Resemblyzer の embedding（256次元）を 1名あたり複数保存し、
識別時は「全サンプル中の最大コサイン類似度」で判定する。

継続学習ループ:
1. 初回 enroll で 1〜N サンプル登録
2. 会議で誤識別が出たら `correct` コマンドで正解を教える
   → 該当 SPEAKER_X.wav が「正解名前」の追加サンプルとして登録される
3. サンプル数が増えるほど、声質・体調・録音環境の変動に強くなる

スキーマ:
- voiceprints(name, sample_id, embedding_json, source, note, created_at)
- correction_history(corrected_at, source, original_label, new_label, similarity)
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .audio import AudioSlice, slice_wav
from .transcribe import TranscriptResult, Utterance


def _default_db_path() -> Path:
    return Path(os.environ.get("VOICEPRINT_DB_PATH", "./voiceprints.sqlite3"))


def _default_threshold() -> float:
    return float(os.environ.get("VOICEPRINT_THRESHOLD", "0.75"))


DEFAULT_DB_PATH = _default_db_path()
DEFAULT_THRESHOLD = _default_threshold()


@dataclass
class Voiceprint:
    """1サンプルの声紋。1名あたり複数 Voiceprint が紐づく。"""

    name: str
    embedding: np.ndarray  # shape: (256,)
    sample_id: int = 0
    source: str = ""
    note: str = ""


@dataclass
class SpeakerStats:
    """名前ごとの登録状況。"""

    name: str
    sample_count: int
    sources: list[str] = field(default_factory=list)


@dataclass
class MatchResult:
    """識別結果の詳細（centroid と max sim を分けて返す）。"""

    name: str | None
    max_similarity: float
    centroid_similarity: float
    sample_count: int


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
    """コサイン類似度（-1〜1）。"""
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


@dataclass
class HomogeneityResult:
    """サンプル wav の内部一貫性判定。"""

    mean_similarity: float
    min_similarity: float
    n_chunks: int
    homogeneous: bool  # 既定閾値 (0.85) を超えていれば True
    suspicion: str  # "ok" / "borderline" / "mixed"

    @property
    def warning_message(self) -> str | None:
        if self.suspicion == "ok":
            return None
        if self.suspicion == "borderline":
            return (
                f"⚠ サンプル一貫性スコア {self.mean_similarity:.2f} (推奨 ≥ 0.85)。"
                f"複数話者混在の可能性あり、念のため中身を確認してください。"
            )
        return (
            f"❌ サンプル一貫性スコア {self.mean_similarity:.2f} (危険域 < 0.70)。"
            f"複数人の声が混在している可能性が極めて高い。SPEAKER_X.wav の中身を"
            f"確認し、被り発話・他人の声・BGMを除去するか、別会議のサンプルで再取得してください。"
        )


HOMOGENEITY_OK_THRESHOLD = 0.85
HOMOGENEITY_DANGER_THRESHOLD = 0.70


def _classify_homogeneity(mean_sim: float) -> str:
    if mean_sim >= HOMOGENEITY_OK_THRESHOLD:
        return "ok"
    if mean_sim >= HOMOGENEITY_DANGER_THRESHOLD:
        return "borderline"
    return "mixed"


def check_sample_homogeneity(
    audio_path: Path,
    *,
    partial_rate: float = 1.0,
) -> HomogeneityResult:
    """音声サンプルが「単一話者の声」になっているか検査する。

    アルゴリズム:
      1. Resemblyzer の `embed_utterance(..., return_partials=True)` で
         1.6秒ウィンドウごとの partial embedding を取得
      2. 全ペアのコサイン類似度を計算
      3. 平均類似度を「内部一貫性スコア」として返す

    Args:
        audio_path: 検査対象 WAV
        partial_rate: partial embedding のサンプリングレート (default 1.0/s)

    Returns:
        HomogeneityResult（mean/min/分類）
    """
    encoder = _get_encoder()
    wav = _load_wav_for_resemblyzer(audio_path)
    # return_partials=True で (embed, partial_embeds, wav_splits) を取得
    _, partial_embeds, _ = encoder.embed_utterance(
        wav, return_partials=True, rate=partial_rate
    )
    return _evaluate_partials(np.asarray(partial_embeds))


def _evaluate_partials(partial_embeds: np.ndarray) -> HomogeneityResult:
    """partial embeddings 行列から一貫性スコアを計算する（テスト容易化のため分離）。"""
    n = int(partial_embeds.shape[0]) if partial_embeds.ndim == 2 else 0
    if n < 2:
        # チャンク 1 個以下では判定不能 → ok 扱い（短すぎサンプルは別途警告）
        return HomogeneityResult(
            mean_similarity=1.0,
            min_similarity=1.0,
            n_chunks=n,
            homogeneous=True,
            suspicion="ok",
        )
    sims: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(cosine_similarity(partial_embeds[i], partial_embeds[j]))
    mean_sim = float(np.mean(sims))
    min_sim = float(np.min(sims))
    suspicion = _classify_homogeneity(mean_sim)
    return HomogeneityResult(
        mean_similarity=mean_sim,
        min_similarity=min_sim,
        n_chunks=n,
        homogeneous=(suspicion == "ok"),
        suspicion=suspicion,
    )


class VoiceprintDB:
    """SQLite ベースの multi-sample 声紋データベース。"""

    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path) if db_path is not None else _default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._init_schema()
        self._migrate_if_needed()

    def _init_schema(self) -> None:
        """新スキーマ作成（既存テーブルがあれば触らない）。"""
        # voiceprints: name + sample_id 複合主キー
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS voiceprints (
                name TEXT NOT NULL,
                sample_id INTEGER NOT NULL,
                embedding_json TEXT NOT NULL,
                source TEXT DEFAULT '',
                note TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (name, sample_id)
            )
            """
        )
        # 訂正履歴（フィードバックループのトレース）
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS correction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                corrected_at TEXT DEFAULT CURRENT_TIMESTAMP,
                source TEXT NOT NULL,
                original_label TEXT,
                new_label TEXT NOT NULL,
                max_similarity REAL,
                note TEXT DEFAULT ''
            )
            """
        )
        self._conn.commit()

    def _migrate_if_needed(self) -> None:
        """旧スキーマ（name PRIMARY KEY のみ）からの自動マイグレーション。"""
        cur = self._conn.execute("PRAGMA table_info(voiceprints)")
        cols = {row[1] for row in cur.fetchall()}
        if "sample_id" in cols:
            return  # 既に新スキーマ
        # 旧 → 新へコピー
        self._conn.execute("ALTER TABLE voiceprints RENAME TO voiceprints_legacy")
        self._init_schema()
        # legacy 行を sample_id=0 として移行
        try:
            cur = self._conn.execute(
                "SELECT name, embedding_json, note FROM voiceprints_legacy"
            )
            rows = cur.fetchall()
            for name, emb_json, note in rows:
                self._conn.execute(
                    "INSERT INTO voiceprints (name, sample_id, embedding_json, source, note) "
                    "VALUES (?, 0, ?, 'legacy', ?)",
                    (name, emb_json, note or ""),
                )
            self._conn.execute("DROP TABLE voiceprints_legacy")
            self._conn.commit()
        except sqlite3.OperationalError:
            # 旧テーブルが部分的に壊れていても続行
            self._conn.rollback()

    def _next_sample_id(self, name: str) -> int:
        cur = self._conn.execute(
            "SELECT COALESCE(MAX(sample_id), -1) + 1 FROM voiceprints WHERE name = ?",
            (name,),
        )
        return int(cur.fetchone()[0])

    def enroll(
        self,
        name: str,
        audio_path: Path,
        *,
        source: str = "",
        note: str = "",
        replace: bool = False,
    ) -> Voiceprint:
        """声紋サンプルを登録する。

        既定は **追加モード**（同名でも別 sample_id で蓄積される）。
        `replace=True` で当該 name の既存サンプルを全削除してから登録。
        """
        emb = compute_embedding(audio_path)
        if replace:
            self._conn.execute("DELETE FROM voiceprints WHERE name = ?", (name,))
        sample_id = self._next_sample_id(name)
        self._conn.execute(
            "INSERT INTO voiceprints (name, sample_id, embedding_json, source, note) "
            "VALUES (?, ?, ?, ?, ?)",
            (name, sample_id, json.dumps(emb.tolist()), source, note),
        )
        self._conn.commit()
        return Voiceprint(name=name, embedding=emb, sample_id=sample_id, source=source, note=note)

    def enroll_embedding(
        self,
        name: str,
        embedding: np.ndarray,
        *,
        source: str = "",
        note: str = "",
    ) -> Voiceprint:
        """既に計算済みの embedding を登録する（correct で複数 sample 一括登録時に使用）。"""
        sample_id = self._next_sample_id(name)
        self._conn.execute(
            "INSERT INTO voiceprints (name, sample_id, embedding_json, source, note) "
            "VALUES (?, ?, ?, ?, ?)",
            (name, sample_id, json.dumps(np.asarray(embedding).tolist()), source, note),
        )
        self._conn.commit()
        return Voiceprint(name=name, embedding=embedding, sample_id=sample_id, source=source, note=note)

    def list_all(self) -> list[Voiceprint]:
        """全サンプルを返す。"""
        cur = self._conn.execute(
            "SELECT name, sample_id, embedding_json, source, note FROM voiceprints ORDER BY name, sample_id"
        )
        result = []
        for name, sid, emb_json, source, note in cur.fetchall():
            emb = np.array(json.loads(emb_json), dtype=np.float32)
            result.append(Voiceprint(name=name, embedding=emb, sample_id=sid, source=source or "", note=note or ""))
        return result

    def stats(self) -> list[SpeakerStats]:
        """名前ごとの登録状況を集計する。"""
        cur = self._conn.execute(
            "SELECT name, COUNT(*), GROUP_CONCAT(COALESCE(source, ''), '|') "
            "FROM voiceprints GROUP BY name ORDER BY name"
        )
        result = []
        for name, count, sources_concat in cur.fetchall():
            sources = [s for s in (sources_concat or "").split("|") if s]
            result.append(SpeakerStats(name=name, sample_count=int(count), sources=sources))
        return result

    def delete(self, name: str) -> bool:
        """name の全サンプルを削除。"""
        cur = self._conn.execute("DELETE FROM voiceprints WHERE name = ?", (name,))
        self._conn.commit()
        return cur.rowcount > 0

    def best_match(
        self,
        query_embedding: np.ndarray,
        threshold: float | None = None,
    ) -> tuple[str | None, float]:
        """登録された名前の中から最も近いものを返す。

        判定: 各 name の **全サンプル中の最大コサイン類似度** で比較。
        外れ値（古い/ノイジーな1サンプル）に強く、サンプル数が増えると単調に
        精度向上しやすい。

        Returns:
            (名前 or None, max cos類似度)
        """
        if threshold is None:
            threshold = _default_threshold()
        result = self.match_detail(query_embedding)
        if result.max_similarity < threshold:
            return None, result.max_similarity
        return result.name, result.max_similarity

    def match_detail(self, query_embedding: np.ndarray) -> MatchResult:
        """centroid 類似度も含めた詳細結果を返す（デバッグ・分析用）。"""
        all_samples = self.list_all()
        if not all_samples:
            return MatchResult(name=None, max_similarity=-1.0, centroid_similarity=-1.0, sample_count=0)

        # 名前ごとに集約
        per_name: dict[str, list[np.ndarray]] = {}
        for vp in all_samples:
            per_name.setdefault(vp.name, []).append(vp.embedding)

        best_name: str | None = None
        best_max: float = -1.0
        best_centroid: float = -1.0
        best_count = 0
        for name, embs in per_name.items():
            sims = [cosine_similarity(query_embedding, e) for e in embs]
            max_sim = max(sims)
            centroid = np.mean(np.stack(embs, axis=0), axis=0)
            centroid_sim = cosine_similarity(query_embedding, centroid)
            if max_sim > best_max:
                best_name = name
                best_max = max_sim
                best_centroid = centroid_sim
                best_count = len(embs)
        return MatchResult(
            name=best_name,
            max_similarity=best_max,
            centroid_similarity=best_centroid,
            sample_count=best_count,
        )

    def record_correction(
        self,
        *,
        source: str,
        original_label: str | None,
        new_label: str,
        max_similarity: float | None = None,
        note: str = "",
    ) -> None:
        """訂正履歴を記録する（フィードバックループのトレース）。"""
        self._conn.execute(
            "INSERT INTO correction_history (source, original_label, new_label, max_similarity, note) "
            "VALUES (?, ?, ?, ?, ?)",
            (source, original_label, new_label, max_similarity, note),
        )
        self._conn.commit()

    def correction_history(self, limit: int = 50) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT corrected_at, source, original_label, new_label, max_similarity, note "
            "FROM correction_history ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [
            {
                "corrected_at": row[0],
                "source": row[1],
                "original_label": row[2],
                "new_label": row[3],
                "max_similarity": row[4],
                "note": row[5],
            }
            for row in cur.fetchall()
        ]

    def close(self) -> None:
        self._conn.close()


def _representative_slice(utterances: list[Utterance], speaker: str, target_sec: float = 30.0) -> AudioSlice | None:
    """ある speaker の代表音声の区間（最大 target_sec 秒）を返す。"""
    candidates = [u for u in utterances if u.speaker == speaker and (u.end_sec - u.start_sec) >= 2.0]
    if not candidates:
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
    """transcript の各 SPEAKER_X を登録済み声紋と照合して個人名にマッピング。"""
    if threshold is None:
        threshold = _default_threshold()
    speakers = sorted({u.speaker for u in transcript.utterances})
    mapping: dict[str, str] = {}

    if not db.list_all():
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
