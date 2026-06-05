"""VAD ベース再分離のユニットテスト。

webrtcvad の実機呼び出しを含む結合相当のテストと、smoothing/assign のロジック単体テスト。
"""
from __future__ import annotations

import numpy as np
import pytest

from voice_shiwake.vad_rediarize import (
    VadSegment,
    assign_text_to_segments,
    detect_speech_segments,
    format_rediarized_dialogue,
    smooth_and_merge,
)


SR = 16000


def _make_signal(duration_s: float, freq_hz: float = 440.0, amp: float = 0.3) -> np.ndarray:
    """指定秒数のサイン波（speech とは判定されないが波形として有効）。"""
    n = int(SR * duration_s)
    t = np.arange(n) / SR
    return (amp * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _make_silence(duration_s: float) -> np.ndarray:
    return np.zeros(int(SR * duration_s), dtype=np.float32)


def test_detect_speech_segments_all_silence_returns_empty() -> None:
    audio = _make_silence(3.0)
    segs = detect_speech_segments(audio, SR, vad_mode=2)
    assert segs == []


def test_detect_speech_segments_rejects_bad_frame_ms() -> None:
    audio = _make_silence(1.0)
    with pytest.raises(ValueError, match="frame_ms"):
        detect_speech_segments(audio, SR, frame_ms=25)


def test_detect_speech_segments_rejects_bad_sr() -> None:
    audio = _make_silence(1.0)
    with pytest.raises(ValueError, match="webrtcvad"):
        detect_speech_segments(audio, 22050)


def test_assign_text_to_segments_within_range() -> None:
    segs = [VadSegment(start=1.0, end=3.0), VadSegment(start=5.0, end=7.0)]
    words = [
        {"start": 0.5, "end": 0.9, "text": "おはよう"},  # 範囲外（前）
        {"start": 1.2, "end": 1.5, "text": "こんに"},  # seg1
        {"start": 1.6, "end": 2.0, "text": "ちは"},  # seg1
        {"start": 3.5, "end": 4.0, "text": "-"},  # ハイフンは無視
        {"start": 5.2, "end": 5.5, "text": "さよ"},  # seg2
        {"start": 5.6, "end": 6.0, "text": "うなら"},  # seg2
        {"start": 8.0, "end": 8.5, "text": "またね"},  # 範囲外（後）
    ]
    assign_text_to_segments(segs, words)
    assert segs[0].text == "こんにちは"
    assert segs[1].text == "さようなら"


def test_assign_text_to_segments_empty_words() -> None:
    segs = [VadSegment(start=0.0, end=1.0)]
    assign_text_to_segments(segs, [])
    assert segs[0].text == ""


def test_smooth_and_merge_forward_fills_none_label() -> None:
    segs = [
        VadSegment(start=0.0, end=1.0, text="a", label="宮下"),
        VadSegment(start=1.0, end=2.0, text="b", label=None),  # 前を継承
        VadSegment(start=2.0, end=3.0, text="c", label="酒井"),
        VadSegment(start=3.0, end=4.0, text="d", label=None),  # 前を継承
    ]
    merged = smooth_and_merge(segs)
    # 0-2s: 宮下 (a+b), 2-4s: 酒井 (c+d)
    assert len(merged) == 2
    assert merged[0].label == "宮下" and merged[0].text == "ab"
    assert merged[1].label == "酒井" and merged[1].text == "cd"


def test_smooth_and_merge_backward_fills_leading_none() -> None:
    segs = [
        VadSegment(start=0.0, end=1.0, text="a", label=None),  # 後ろから継承
        VadSegment(start=1.0, end=2.0, text="b", label="宮下"),
    ]
    merged = smooth_and_merge(segs)
    assert len(merged) == 1
    assert merged[0].label == "宮下"


def test_smooth_and_merge_all_none_marks_unknown() -> None:
    segs = [VadSegment(start=0.0, end=1.0, label=None)]
    merged = smooth_and_merge(segs)
    assert merged[0].label == "不明"


def test_format_rediarized_dialogue_skips_empty_text() -> None:
    segs = [
        VadSegment(start=0.0, end=1.0, text="ハロー", label="宮下"),
        VadSegment(start=1.0, end=2.0, text="", label="酒井"),  # スキップ
        VadSegment(start=2.0, end=3.0, text="お疲れ", label="宮下"),
    ]
    out = format_rediarized_dialogue(segs)
    assert out == "宮下: ハロー\n宮下: お疲れ"
