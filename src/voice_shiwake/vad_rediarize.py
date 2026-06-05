"""VAD ベースの細粒度セグメント化 + 声紋による話者再分離。

videodb の盲目 diarization が話者を取りこぼした場合の後処理として使用する。
オフライン1マイクで複数名が話す録音で、既知の候補2-3名を指定すると、
WebRTC VAD で speech 区間を抽出し、Resemblyzer の embedding を声紋DBの
候補と比較して話者ラベルを付け直す。

videodb のセグメントを使わず音声から直接 segmentation するため、ターン交替を
より細かく捉えられる可能性があるが、オフライン1マイクの音響条件では本質的な
分離限界が存在する点に注意。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .voiceprint import VoiceprintDB, _get_encoder, cosine_similarity


@dataclass
class VadSegment:
    """1つの speech 区間。"""

    start: float
    end: float
    text: str = ""
    label: str | None = None
    sims: dict[str, float] = field(default_factory=dict)
    embedded: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start


def detect_speech_segments(
    audio: np.ndarray,
    sr: int,
    *,
    frame_ms: int = 30,
    vad_mode: int = 2,
    min_pause_ms: int = 300,
    min_seg_ms: int = 500,
) -> list[tuple[float, float]]:
    """WebRTC VAD で speech 区間を抽出する。

    Args:
        audio: mono の音声配列 (float32 or int16)
        sr: サンプリングレート (8000/16000/32000/48000 のいずれか、webrtcvad 制約)
        frame_ms: 1フレームの長さ ms (10/20/30 のみ受け付け、webrtcvad 制約)
        vad_mode: 0 (緩) - 3 (厳格)。0 は雑音も拾い、3 は話者の声でも捨てがち
        min_pause_ms: この長さ以上の無音で segment を切る (短ければ吸収して連続発話扱い)
        min_seg_ms: この長さ未満の segment は捨てる (短いノイズの誤検出を除外)

    Returns:
        [(start_sec, end_sec), ...] の昇順リスト
    """
    try:
        import webrtcvad
    except ImportError as e:
        raise RuntimeError("webrtcvad 未インストール") from e

    if sr not in (8000, 16000, 32000, 48000):
        raise ValueError(f"sr={sr} は webrtcvad 非対応 (8000/16000/32000/48000 のみ)")
    if frame_ms not in (10, 20, 30):
        raise ValueError(f"frame_ms={frame_ms} は webrtcvad 非対応 (10/20/30 のみ)")

    vad = webrtcvad.Vad(vad_mode)
    frame_samples = int(sr * frame_ms / 1000)

    # int16 に正規化（webrtcvad は int16 PCM を要求）
    if audio.dtype != np.int16:
        audio_int = (audio * 32767.0).clip(-32768, 32767).astype(np.int16)
    else:
        audio_int = audio

    n_frames = len(audio_int) // frame_samples
    flags = [False] * n_frames
    for i in range(n_frames):
        chunk = audio_int[i * frame_samples : (i + 1) * frame_samples].tobytes()
        try:
            flags[i] = vad.is_speech(chunk, sr)
        except Exception:
            flags[i] = False

    max_pause_frames = max(1, int(min_pause_ms / frame_ms))
    min_seg_frames = max(1, int(min_seg_ms / frame_ms))

    segments: list[tuple[float, float]] = []
    in_speech = False
    seg_start = 0
    pause = 0
    for i, is_sp in enumerate(flags):
        if is_sp:
            if not in_speech:
                seg_start = i
                in_speech = True
            pause = 0
        else:
            if in_speech:
                pause += 1
                if pause >= max_pause_frames:
                    seg_end = i - pause + 1
                    if seg_end - seg_start >= min_seg_frames:
                        segments.append(
                            (
                                seg_start * frame_ms / 1000.0,
                                seg_end * frame_ms / 1000.0,
                            )
                        )
                    in_speech = False
                    pause = 0
    if in_speech:
        seg_end = n_frames
        if seg_end - seg_start >= min_seg_frames:
            segments.append(
                (seg_start * frame_ms / 1000.0, seg_end * frame_ms / 1000.0)
            )
    return segments


def assign_text_to_segments(
    segments: list[VadSegment],
    word_entries: list[dict[str, Any]],
) -> None:
    """transcript_structured.json の単語を時間範囲で各 segment に割り当てる。

    word_entries が start でソート済み前提。"-" や空 text は無視する。
    """
    j = 0
    for seg in segments:
        while j < len(word_entries) and float(word_entries[j].get("end", 0.0)) <= seg.start:
            j += 1
        parts: list[str] = []
        k = j
        while k < len(word_entries):
            w_start = float(word_entries[k].get("start", 0.0))
            if w_start >= seg.end:
                break
            text = word_entries[k].get("text", "")
            if text and text != "-":
                parts.append(text)
            k += 1
        seg.text = "".join(parts)


def label_segments(
    segments: list[VadSegment],
    audio: np.ndarray,
    sr: int,
    candidate_embeddings: dict[str, list[np.ndarray]],
    *,
    min_dur_for_embed: float = 1.6,
) -> None:
    """各 segment の音声を embed し、候補との cos 類似度で最大のラベルを付ける。"""
    try:
        from resemblyzer import preprocess_wav
    except ImportError as e:
        raise RuntimeError("resemblyzer 未インストール") from e

    encoder = _get_encoder()
    for seg in segments:
        if seg.duration < min_dur_for_embed:
            continue
        a = max(0, int(seg.start * sr))
        b = min(len(audio), int(seg.end * sr))
        clip = audio[a:b]
        if len(clip) < int(sr * 0.5):
            continue
        try:
            wav = preprocess_wav(clip.astype(np.float32), source_sr=sr)
            emb = encoder.embed_utterance(wav)
        except Exception:
            continue
        sims = {
            name: float(max(cosine_similarity(emb, e) for e in embs))
            for name, embs in candidate_embeddings.items()
        }
        winner = max(sims, key=sims.get)
        seg.sims = sims
        seg.label = winner
        seg.embedded = True


def smooth_and_merge(segments: list[VadSegment]) -> list[VadSegment]:
    """ラベル未確定（embedding スキップ）segment は隣接継承、連続同一話者は merge。"""
    last: str | None = None
    for s in segments:
        if s.label is None:
            s.label = last
        else:
            last = s.label
    nxt: str | None = None
    for s in reversed(segments):
        if s.label is None:
            s.label = nxt
        else:
            nxt = s.label
    for s in segments:
        if s.label is None:
            s.label = "不明"

    merged: list[VadSegment] = []
    for s in segments:
        if merged and merged[-1].label == s.label:
            merged[-1].text += s.text
            merged[-1].end = s.end
        else:
            merged.append(
                VadSegment(start=s.start, end=s.end, text=s.text, label=s.label)
            )
    return merged


def rediarize_with_vad(
    audio_path: Path,
    transcript_json_path: Path,
    candidates: list[str],
    *,
    db_path: Path | None = None,
    vad_mode: int = 2,
    min_pause_ms: int = 300,
    min_seg_ms: int = 500,
    min_dur_for_embed: float = 1.6,
) -> dict[str, Any]:
    """VAD + 声紋で再分離するエントリーポイント。

    Returns:
        {
            'segments': merge後の VadSegment list,
            'raw_segments': merge前の VadSegment list,
            'stats': {label: {'count': int, 'duration_s': float}},
            'avg_margin': float (winner-loser の平均 cos 差),
            'embedded_count': int,
        }
    """
    audio, sr = sf.read(str(audio_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    word_entries = json.loads(transcript_json_path.read_text(encoding="utf-8"))

    db = VoiceprintDB(db_path)
    all_vps = db.list_all()
    db.close()
    cand_embs: dict[str, list[np.ndarray]] = {name: [] for name in candidates}
    for vp in all_vps:
        if vp.name in cand_embs:
            cand_embs[vp.name].append(vp.embedding)
    missing = [n for n in candidates if not cand_embs[n]]
    if missing:
        raise ValueError(f"声紋が DB に未登録: {missing}")

    raw_pairs = detect_speech_segments(
        audio,
        sr,
        vad_mode=vad_mode,
        min_pause_ms=min_pause_ms,
        min_seg_ms=min_seg_ms,
    )
    segments = [VadSegment(start=s, end=e) for s, e in raw_pairs]
    assign_text_to_segments(segments, word_entries)
    label_segments(
        segments, audio, sr, cand_embs, min_dur_for_embed=min_dur_for_embed
    )

    embedded_count = sum(1 for s in segments if s.embedded)

    merged = smooth_and_merge(segments)

    stats: dict[str, dict[str, float]] = {}
    margins: list[float] = []
    for s in segments:
        if s.label and s.label in candidates:
            stats.setdefault(s.label, {"count": 0, "duration_s": 0.0})
            stats[s.label]["count"] += 1
            stats[s.label]["duration_s"] += s.duration
            if s.sims:
                margin = s.sims[s.label] - min(s.sims.values())
                margins.append(margin)

    return {
        "segments": merged,
        "raw_segments": segments,
        "stats": stats,
        "avg_margin": float(np.mean(margins)) if margins else 0.0,
        "embedded_count": embedded_count,
    }


def format_rediarized_dialogue(segments: list[VadSegment]) -> str:
    """merge 済み segments を「ラベル: 発話\\n」形式に整形する。"""
    lines = [f"{s.label}: {s.text}" for s in segments if s.text.strip()]
    return "\n".join(lines)
