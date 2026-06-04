"""videodb_pipeline の純粋ロジックの単体テスト（SDK は使わない）。"""

from __future__ import annotations

from voice_shiwake.videodb_pipeline import (
    aggregate_utterances,
    build_minutes_markdown,
)


SAMPLE_STRUCTURED = [
    {"start": 0.0, "end": 1.0, "text": "-"},  # placeholder
    {"start": 1.0, "end": 2.0, "speaker": "A", "text": "おはよう"},
    {"start": 2.0, "end": 3.0, "speaker": "A", "text": "ございます"},
    {"start": 4.0, "end": 5.0, "speaker": "B", "text": "はい"},
    {"start": 10.0, "end": 11.0, "speaker": "A", "text": "じゃあ"},  # 5秒以上開いたので別発話
    {"start": 11.0, "end": 12.0, "speaker": "A", "text": "始めます"},
]


class TestAggregateUtterances:
    def test_groups_consecutive_words_into_utterance(self):
        utterances, time_, count = aggregate_utterances(SAMPLE_STRUCTURED)
        # A は 2 発話、B は 1 発話のはず
        assert len(utterances) == 3
        # 1個目: Aの「おはようございます」
        assert utterances[0]["speaker"] == "A"
        assert utterances[0]["text"] == "おはようございます"
        assert utterances[0]["start"] == 1.0
        assert utterances[0]["end"] == 3.0
        # 2個目: Bの「はい」
        assert utterances[1]["speaker"] == "B"
        # 3個目: Aの「じゃあ始めます」(silence_gap=2.0 で別発話)
        assert utterances[2]["speaker"] == "A"
        assert utterances[2]["text"] == "じゃあ始めます"

    def test_speaker_time_aggregation(self):
        _, time_, _ = aggregate_utterances(SAMPLE_STRUCTURED)
        # A: (2-1)+(3-2)+(11-10)+(12-11) = 4 秒
        assert time_["A"] == 4.0
        # B: 1秒
        assert time_["B"] == 1.0

    def test_speaker_word_count(self):
        _, _, count = aggregate_utterances(SAMPLE_STRUCTURED)
        # A: おはよう/ございます/じゃあ/始めます = 4
        assert count["A"] == 4
        # B: はい = 1
        assert count["B"] == 1

    def test_empty_input(self):
        utts, time_, count = aggregate_utterances([])
        assert utts == []
        assert time_ == {}
        assert count == {}

    def test_no_speaker_field_ignored(self):
        data = [{"start": 0.0, "end": 1.0, "text": "x"}]
        utts, time_, count = aggregate_utterances(data)
        assert utts == []
        assert time_ == {}


class TestBuildMinutesMarkdown:
    def test_includes_speaker_stats_table(self):
        utts, time_, count = aggregate_utterances(SAMPLE_STRUCTURED)
        md = build_minutes_markdown(
            title="テスト議事録",
            video_name="test.mp4",
            duration_sec=12.0,
            videodb_id="m-test",
            utterances=utts,
            speaker_time=time_,
            speaker_word_count=count,
        )
        assert "# テスト議事録" in md
        assert "SPEAKER_A" in md
        assert "SPEAKER_B" in md
        assert "発話率" in md
        assert "m-test" in md

    def test_applies_speaker_name_map(self):
        utts, time_, count = aggregate_utterances(SAMPLE_STRUCTURED)
        md = build_minutes_markdown(
            title="t",
            video_name="t.mp4",
            duration_sec=10.0,
            videodb_id="id",
            utterances=utts,
            speaker_time=time_,
            speaker_word_count=count,
            speaker_name_map={"A": "山田太郎", "B": "山田"},
        )
        assert "山田太郎" in md
        assert "山田" in md
        # 文字起こし行にも適用される
        assert "山田太郎**: おはようございます" in md
