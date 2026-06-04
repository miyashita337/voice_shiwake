"""videodb 単独で動画→音声→ASR+diarization→議事録→声紋サンプル抽出 を完結させる。

AssemblyAI / Anthropic を使わずに videodb の structured transcript（speaker フィールド付き）
だけで動作するパイプライン。batch-process で大量動画を一括処理するベース。
"""

from __future__ import annotations

import json
import os
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .audio import ensure_ffmpeg, extract_audio


@dataclass
class SpeakerSummary:
    """1動画における話者ごとの集計。"""

    label: str  # "A", "B", "C", ...
    total_sec: float
    word_count: int
    sample_path: Path | None = None  # 抽出した代表音声 WAV


@dataclass
class PipelineResult:
    """1動画の処理結果。"""

    video_path: Path
    output_dir: Path
    audio_path: Path
    transcript_text_path: Path
    transcript_json_path: Path
    minutes_md_path: Path
    samples_dir: Path
    videodb_id: str
    duration_sec: float
    n_utterances: int
    speakers: list[SpeakerSummary] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Path → str
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        d["speakers"] = [
            {
                "label": s.label,
                "total_sec": s.total_sec,
                "word_count": s.word_count,
                "sample_path": str(s.sample_path) if s.sample_path else None,
            }
            for s in self.speakers
        ]
        return d


class VideodbPipelineError(RuntimeError):
    pass


def _get_videodb_client(api_key: str | None = None) -> Any:
    api_key = api_key or os.environ.get("VIDEODB_API_KEY")
    if not api_key:
        raise VideodbPipelineError(
            "VIDEODB_API_KEY が未設定。.env または環境変数を確認してください。"
        )
    try:
        import videodb
    except ImportError as e:
        raise VideodbPipelineError("videodb 未インストール。`pip install videodb`") from e
    return videodb.connect(api_key=api_key)


def upload_and_transcribe(
    video_path: Path,
    *,
    api_key: str | None = None,
) -> tuple[str, float, str, list[dict[str, Any]]]:
    """videodb に動画をアップロードして speaker 付き transcript を取得する。

    Returns:
        (video_id, length_sec, transcript_text, structured_transcript)
    """
    conn = _get_videodb_client(api_key)
    coll = conn.get_collection()
    video = coll.upload(file_path=str(video_path))
    try:
        video.index_spoken_words()
    except Exception as e:  # noqa: BLE001
        raise VideodbPipelineError(f"index_spoken_words 失敗: {e}") from e

    try:
        transcript_text = video.get_transcript_text() or ""
    except Exception:  # noqa: BLE001
        transcript_text = ""

    try:
        structured = video.get_transcript()
    except Exception:  # noqa: BLE001
        structured = []

    return str(video.id), float(video.length), transcript_text, list(structured or [])


def aggregate_utterances(
    structured: list[dict[str, Any]],
    *,
    silence_gap: float = 2.0,
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, int]]:
    """単語列から発話列を生成し、話者別統計を集計する。

    Args:
        structured: videodb の get_transcript() 戻り値
        silence_gap: 同一話者でも N秒以上開いたら別発話とみなす

    Returns:
        (utterances, speaker_time, speaker_word_count)
    """
    utterances: list[dict[str, Any]] = []
    speaker_time: defaultdict[str, float] = defaultdict(float)
    speaker_word_count: Counter[str] = Counter()

    current_speaker: str | None = None
    current_words: list[str] = []
    current_start: float | None = None
    current_end: float | None = None

    def flush() -> None:
        nonlocal current_speaker, current_words, current_start, current_end
        if (
            current_speaker
            and current_words
            and current_start is not None
            and current_end is not None
        ):
            utterances.append(
                {
                    "speaker": current_speaker,
                    "start": current_start,
                    "end": current_end,
                    "text": "".join(current_words),
                }
            )
        current_speaker = None
        current_words = []
        current_start = None
        current_end = None

    for entry in structured:
        spk = entry.get("speaker")
        text = entry.get("text", "")
        start = float(entry.get("start", 0.0))
        end = float(entry.get("end", start))
        if spk is None or not text or text == "-":
            flush()
            continue
        speaker_time[spk] += end - start
        speaker_word_count[spk] += 1

        if current_speaker is None:
            current_speaker = spk
            current_start = start
            current_end = end
            current_words = [text]
        elif spk == current_speaker and current_end is not None and start - current_end < silence_gap:
            current_words.append(text)
            current_end = end
        else:
            flush()
            current_speaker = spk
            current_start = start
            current_end = end
            current_words = [text]
    flush()
    return utterances, dict(speaker_time), dict(speaker_word_count)


def _fmt_time(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def build_minutes_markdown(
    *,
    title: str,
    video_name: str,
    duration_sec: float,
    videodb_id: str,
    utterances: list[dict[str, Any]],
    speaker_time: dict[str, float],
    speaker_word_count: dict[str, int],
    speaker_name_map: dict[str, str] | None = None,
) -> str:
    """話者ラベル付き議事録 Markdown を構築する。"""
    speaker_name_map = speaker_name_map or {}
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- 動画: {video_name}")
    lines.append(f"- 長さ: {_fmt_time(duration_sec)}")
    lines.append(f"- videodb ID: `{videodb_id}`")
    lines.append(f"- 検出話者数: {len(speaker_time)}")
    lines.append("")
    lines.append("## 話者別発話統計")
    lines.append("")
    lines.append("| 話者 | 発話時間 | 発話単語数 | 発話率 |")
    lines.append("|---|---|---|---|")
    total = sum(speaker_time.values()) or 1.0
    for spk, sec in sorted(speaker_time.items(), key=lambda x: -x[1]):
        name = speaker_name_map.get(spk, f"SPEAKER_{spk}")
        pct = sec / total * 100
        lines.append(f"| {name} | {_fmt_time(sec)} | {speaker_word_count.get(spk, 0)} | {pct:.1f}% |")
    lines.append("")
    lines.append("## 文字起こし（話者ラベル付き）")
    lines.append("")
    for u in utterances:
        name = speaker_name_map.get(u["speaker"], f"SPEAKER_{u['speaker']}")
        timestamp = _fmt_time(u["start"])
        text = u["text"].replace("\n", " ").strip()
        if text:
            lines.append(f"**[{timestamp}] {name}**: {text}")
            lines.append("")
    return "\n".join(lines)


def extract_speaker_samples(
    source_wav: Path,
    structured: list[dict[str, Any]],
    samples_dir: Path,
    *,
    target_sec: float = 30.0,
    min_utterance_sec: float = 2.0,
) -> dict[str, Path]:
    """各話者の代表音声を WAV として保存する。

    Args:
        source_wav: 元音声 (16kHz mono WAV)
        structured: videodb structured transcript
        samples_dir: 出力先
        target_sec: 1話者あたりの目標長さ
        min_utterance_sec: これ未満の短い発話は候補から除外

    Returns:
        {"A": Path("samples/SPEAKER_A.wav"), ...}
    """
    ensure_ffmpeg()
    samples_dir.mkdir(parents=True, exist_ok=True)

    speakers = sorted({d.get("speaker") for d in structured if d.get("speaker")})
    result: dict[str, Path] = {}

    for spk in speakers:
        segs = _collect_long_segments(structured, spk, min_utterance_sec)
        if not segs:
            continue
        segs.sort(key=lambda x: -(x[1] - x[0]))
        accumulated = 0.0
        chosen: list[tuple[float, float]] = []
        for s, e in segs:
            chosen.append((s, e))
            accumulated += e - s
            if accumulated >= target_sec:
                break
        sample_wav = samples_dir / f"SPEAKER_{spk}.wav"
        _concat_slices(source_wav, chosen, sample_wav)
        result[spk] = sample_wav
    return result


def _collect_long_segments(
    structured: list[dict[str, Any]],
    target_speaker: str,
    min_sec: float,
) -> list[tuple[float, float]]:
    segs: list[tuple[float, float]] = []
    cur_start: float | None = None
    cur_end: float | None = None
    for entry in structured:
        spk = entry.get("speaker")
        start = float(entry.get("start", 0.0))
        end = float(entry.get("end", start))
        if spk == target_speaker:
            if cur_start is None:
                cur_start = start
                cur_end = end
            elif cur_end is not None and start - cur_end < 0.5:
                cur_end = end
            else:
                if cur_end is not None and cur_end - cur_start >= min_sec:
                    segs.append((cur_start, cur_end))
                cur_start = start
                cur_end = end
        else:
            if cur_start is not None and cur_end is not None and cur_end - cur_start >= min_sec:
                segs.append((cur_start, cur_end))
            cur_start = None
            cur_end = None
    if cur_start is not None and cur_end is not None and cur_end - cur_start >= min_sec:
        segs.append((cur_start, cur_end))
    return segs


def _concat_slices(
    source_wav: Path,
    slices: list[tuple[float, float]],
    output_wav: Path,
) -> None:
    """複数の (start, end) を切り出して 1 つの WAV に結合する。"""
    tmp_dir = output_wav.parent / f"_tmp_{output_wav.stem}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    seg_paths: list[Path] = []
    try:
        for i, (s, e) in enumerate(slices):
            seg = tmp_dir / f"seg_{i:03d}.wav"
            cmd = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(source_wav),
                "-ss",
                f"{s:.3f}",
                "-to",
                f"{e:.3f}",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-acodec",
                "pcm_s16le",
                str(seg),
            ]
            subprocess.run(cmd, check=True)
            seg_paths.append(seg)

        list_file = tmp_dir / "concat.txt"
        list_file.write_text(
            "\n".join(f"file '{p.resolve()}'" for p in seg_paths) + "\n",
            encoding="utf-8",
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            str(output_wav),
        ]
        subprocess.run(cmd, check=True)
    finally:
        for p in seg_paths:
            p.unlink(missing_ok=True)
        list_file = tmp_dir / "concat.txt"
        if list_file.exists():
            list_file.unlink()
        try:
            tmp_dir.rmdir()
        except OSError:
            pass


def process_video(
    video_path: Path,
    output_dir: Path,
    *,
    title: str | None = None,
    api_key: str | None = None,
    target_sample_sec: float = 30.0,
) -> PipelineResult:
    """1動画の完全パイプライン: 音声抽出 → videodb → 議事録 → サンプル抽出。

    Args:
        video_path: 入力動画
        output_dir: この動画の出力先（既存ならファイルが上書きされる）
        title: 議事録タイトル（省略時はファイル名）
        api_key: videodb キー（省略時は環境変数）
        target_sample_sec: 各話者の代表音声目標長さ
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    title = title or f"会議議事録 {video_path.stem}"

    # 1. 音声抽出
    audio_path = output_dir / "audio.wav"
    extract_audio(video_path, audio_path)

    # 2. videodb で文字起こし + diarization
    video_id, duration, transcript_text, structured = upload_and_transcribe(
        video_path, api_key=api_key
    )

    # 3. 集計
    utterances, speaker_time, speaker_word_count = aggregate_utterances(structured)

    # 4. 保存
    transcript_text_path = output_dir / "transcript_text.txt"
    transcript_text_path.write_text(transcript_text, encoding="utf-8")
    transcript_json_path = output_dir / "transcript_structured.json"
    transcript_json_path.write_text(
        json.dumps(structured, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # 5. 議事録 Markdown
    minutes_md = build_minutes_markdown(
        title=title,
        video_name=video_path.name,
        duration_sec=duration,
        videodb_id=video_id,
        utterances=utterances,
        speaker_time=speaker_time,
        speaker_word_count=speaker_word_count,
    )
    minutes_md_path = output_dir / "minutes.md"
    minutes_md_path.write_text(minutes_md, encoding="utf-8")

    # 6. 各話者の代表音声を抽出
    samples_dir = output_dir / "samples"
    sample_map = extract_speaker_samples(audio_path, structured, samples_dir, target_sec=target_sample_sec)

    speakers = []
    for spk in sorted(speaker_time.keys()):
        speakers.append(
            SpeakerSummary(
                label=spk,
                total_sec=speaker_time.get(spk, 0.0),
                word_count=speaker_word_count.get(spk, 0),
                sample_path=sample_map.get(spk),
            )
        )

    return PipelineResult(
        video_path=video_path,
        output_dir=output_dir,
        audio_path=audio_path,
        transcript_text_path=transcript_text_path,
        transcript_json_path=transcript_json_path,
        minutes_md_path=minutes_md_path,
        samples_dir=samples_dir,
        videodb_id=video_id,
        duration_sec=duration,
        n_utterances=len(utterances),
        speakers=speakers,
    )
