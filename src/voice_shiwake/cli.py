"""voice-shiwake コマンドラインインターフェース。

サブコマンド:
- enroll: 参加者の声紋を登録
- list:   登録済み声紋を一覧
- delete: 登録済み声紋を削除
- process: 動画を処理して議事録を生成（必要なら Slack 投稿）
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import click
from dotenv import load_dotenv

from .audio import extract_audio
from .minutes import summarize_to_minutes
from .slack import SlackError, post_to_slack_auto
from .transcribe import format_as_dialogue, transcribe
from .roster import RosterDB
from .videodb_adapter import VideodbError, semantic_search, upload_and_index
from .videodb_pipeline import VideodbPipelineError, process_video
from .voiceprint import (
    HomogeneityResult,
    VoiceprintDB,
    check_sample_homogeneity,
    identify_speakers,
)


def _run_quality_check(
    audio_path: Path,
    *,
    skip: bool,
    strict: bool,
) -> HomogeneityResult | None:
    """品質チェックを実行し、結果に応じて警告 or 終了する。

    Returns:
        実行した場合は HomogeneityResult、skip した場合は None
    """
    if skip:
        return None
    try:
        result = check_sample_homogeneity(audio_path)
    except Exception as e:  # noqa: BLE001
        click.echo(f"  [WARN] 品質チェック実行失敗（続行）: {e}", err=True)
        return None

    msg = result.warning_message
    if msg:
        click.echo(f"  {msg}", err=True)
        if strict and result.suspicion == "mixed":
            click.echo(
                "  --strict 指定により登録を中止します。"
                " --no-quality-check で強制登録可能。",
                err=True,
            )
            sys.exit(3)
    else:
        click.echo(
            f"  [OK] サンプル一貫性スコア {result.mean_similarity:.3f} "
            f"(chunks={result.n_chunks}, min={result.min_similarity:.3f})"
        )
    return result

load_dotenv()


@click.group()
@click.version_option(package_name="voice-shiwake")
def main() -> None:
    """会議動画から議事録を生成して Slack に投稿するツール。"""


@main.command()
@click.option("--name", required=True, help="参加者の表示名（例: 田中太郎）")
@click.option("--audio", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="30秒程度の音声サンプル")
@click.option("--source", default="", help="サンプル出所（例: 2026-06-01 weekly）")
@click.option("--note", default="", help="メモ（任意）")
@click.option("--replace", is_flag=True, help="既存サンプルを全削除してから登録（既定: 追加モード）")
@click.option("--no-quality-check", is_flag=True, help="サンプル品質チェックをスキップ")
@click.option("--strict", is_flag=True, help="品質チェックで「混在」判定なら exit 3")
def enroll(
    name: str,
    audio: Path,
    source: str,
    note: str,
    replace: bool,
    no_quality_check: bool,
    strict: bool,
) -> None:
    """参加者の声紋を登録する（既定は追加モード、同名で複数サンプル蓄積可能）。"""
    click.echo(f"[1/2] 品質チェック: {audio.name}")
    _run_quality_check(audio, skip=no_quality_check, strict=strict)
    click.echo("[2/2] 声紋DB に登録")
    db = VoiceprintDB()
    try:
        vp = db.enroll(name=name, audio_path=audio, source=source, note=note, replace=replace)
        sample_count = sum(1 for v in db.list_all() if v.name == name)
        mode = "上書き" if replace else "追加"
        click.echo(f"登録完了 ({mode}): {vp.name} sample_id={vp.sample_id} (累計 {sample_count} sample)")
    finally:
        db.close()


@main.command(name="list")
def list_voiceprints() -> None:
    """登録済み声紋を一覧表示（名前ごとのサンプル数）。"""
    db = VoiceprintDB()
    try:
        stats = db.stats()
        if not stats:
            click.echo("登録済み声紋なし。`voice-shiwake enroll` で追加してください。")
            return
        for s in stats:
            srcs = f" [{', '.join(s.sources)}]" if any(s.sources) else ""
            click.echo(f"- {s.name}: {s.sample_count} samples{srcs}")
    finally:
        db.close()


@main.command()
@click.option("--name", required=True, help="削除する声紋名（全サンプル削除）")
def delete(name: str) -> None:
    """登録済み声紋を削除（その名前の全サンプル）。"""
    db = VoiceprintDB()
    try:
        deleted = db.delete(name)
        if deleted:
            click.echo(f"削除: {name}")
        else:
            click.echo(f"見つからず: {name}", err=True)
            sys.exit(1)
    finally:
        db.close()


@main.command()
@click.option("--samples-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path), help="SPEAKER_*.wav が入ったディレクトリ")
@click.option("--map", "mappings", multiple=True, required=True, help="ラベル対応（複数指定可）。例: --map A=田中 --map B=佐藤")
@click.option("--source", default="", help="サンプル出所（既定: ディレクトリ名）")
@click.option("--no-quality-check", is_flag=True, help="サンプル品質チェックをスキップ")
@click.option("--strict", is_flag=True, help="混在判定のサンプルは登録せずスキップ")
def correct(
    samples_dir: Path,
    mappings: tuple[str, ...],
    source: str,
    no_quality_check: bool,
    strict: bool,
) -> None:
    """SPEAKER_X.wav に正解の人物名を紐付けて声紋DBに追加する（継続学習）。

    例:
        voice-shiwake correct --samples-dir work/samples \\
            --map A=山田太郎 --map B=小川 --map C=林
    """
    source = source or samples_dir.name
    parsed: list[tuple[str, str]] = []
    for m in mappings:
        if "=" not in m:
            click.echo(f"不正な --map 形式（A=name 必要）: {m}", err=True)
            sys.exit(1)
        label, person = m.split("=", 1)
        label = label.strip().upper()
        person = person.strip()
        if not label or not person:
            click.echo(f"空のラベル/人物名: {m}", err=True)
            sys.exit(1)
        parsed.append((label, person))

    # ロスターチェック（未登録名前を警告）
    try:
        roster_db = RosterDB()
        roster_names = {m.name for m in roster_db.list_all()}
        roster_db.close()
        if roster_names:
            unknown = [p for _, p in parsed if p not in roster_names]
            if unknown:
                click.echo(
                    f"⚠ ロスター未登録の名前: {', '.join(unknown)}（typo 注意）",
                    err=True,
                )
    except Exception:  # noqa: BLE001
        pass  # ロスター無くても correct は続行

    db = VoiceprintDB()
    try:
        for label, person in parsed:
            # SPEAKER_A.wav / A.wav どちらでも拾う
            candidates = [
                samples_dir / f"SPEAKER_{label}.wav",
                samples_dir / f"{label}.wav",
            ]
            audio = next((c for c in candidates if c.exists()), None)
            if audio is None:
                click.echo(f"  [SKIP] SPEAKER_{label}.wav が見つからず", err=True)
                continue

            # 品質チェック（strict 時は混在なら登録スキップ）
            click.echo(f"SPEAKER_{label} → {person}: 品質チェック ({audio.name})")
            if not no_quality_check:
                try:
                    quality = check_sample_homogeneity(audio)
                except Exception as e:  # noqa: BLE001
                    click.echo(f"  [WARN] 品質チェック失敗（続行）: {e}", err=True)
                    quality = None
                if quality is not None:
                    if quality.warning_message:
                        click.echo(f"  {quality.warning_message}", err=True)
                        if strict and quality.suspicion == "mixed":
                            click.echo(
                                f"  [SKIP] strict mode: SPEAKER_{label} → {person} 登録をスキップ",
                                err=True,
                            )
                            continue
                    else:
                        click.echo(f"  [OK] 一貫性 {quality.mean_similarity:.3f}")

            vp = db.enroll(name=person, audio_path=audio, source=source, note=f"correction from SPEAKER_{label}")
            db.record_correction(
                source=source,
                original_label=f"SPEAKER_{label}",
                new_label=person,
                note=str(audio),
            )
            sample_count = sum(1 for v in db.list_all() if v.name == person)
            click.echo(f"  [OK] SPEAKER_{label} → {person} (sample_id={vp.sample_id}, 累計 {sample_count})")
        click.echo(f"訂正完了: {len(parsed)} 件")
    finally:
        db.close()


@main.command()
@click.option("--limit", default=20, type=int, help="表示件数")
def history(limit: int) -> None:
    """訂正履歴を表示（誰がいつ何を修正したかのトレース）。"""
    db = VoiceprintDB()
    try:
        rows = db.correction_history(limit=limit)
        if not rows:
            click.echo("訂正履歴なし。")
            return
        for r in rows:
            orig = r.get("original_label") or "?"
            click.echo(f"[{r['corrected_at']}] {orig} → {r['new_label']} (source={r['source']})")
    finally:
        db.close()


@main.command()
@click.option("--video", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="入力動画")
@click.option("--title", default=None, help="議事録タイトル")
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Markdown 出力先")
@click.option("--post-slack", is_flag=True, help="Slack に投稿する")
@click.option("--no-identify", is_flag=True, help="声紋照合をスキップ")
@click.option("--no-summarize", is_flag=True, help="Claude 整形をスキップ（生文字起こしを出力）")
@click.option("--save-json", type=click.Path(dir_okay=False, path_type=Path), default=None, help="文字起こしJSONの保存先")
def process(
    video: Path,
    title: str | None,
    output: Path | None,
    post_slack: bool,
    no_identify: bool,
    no_summarize: bool,
    save_json: Path | None,
) -> None:
    """動画を処理して議事録を生成する。"""
    now = datetime.now()
    title = title or f"会議議事録 {now.strftime('%Y-%m-%d %H:%M')}"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        wav_path = tmp / "audio.wav"

        click.echo(f"[1/5] 音声抽出: {video.name} → 16kHz mono WAV")
        extract_audio(video, wav_path)

        click.echo("[2/5] AssemblyAI で文字起こし＋話者分離")
        transcript = transcribe(wav_path)
        click.echo(f"  utterances: {len(transcript.utterances)}, raw_chars: {len(transcript.raw_text)}")

        if save_json:
            save_json.parent.mkdir(parents=True, exist_ok=True)
            save_json.write_text(json.dumps(transcript.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            click.echo(f"  JSON保存: {save_json}")

        speaker_names: dict[str, str] = {}
        if not no_identify:
            click.echo("[3/5] 声紋照合で個人名にマッピング")
            db = VoiceprintDB()
            try:
                speaker_names = identify_speakers(transcript, wav_path, db)
            finally:
                db.close()
            for spk, name in speaker_names.items():
                click.echo(f"  {spk} → {name}")
        else:
            click.echo("[3/5] 声紋照合スキップ")

        dialogue = format_as_dialogue(transcript, speaker_names)

        if no_summarize:
            click.echo("[4/5] Claude 整形スキップ（生文字起こしを使用）")
            markdown = f"# {title}\n\n```\n{dialogue}\n```\n"
        else:
            click.echo("[4/5] Claude で議事録整形")
            minutes = summarize_to_minutes(dialogue, title=title, meeting_datetime=now)
            markdown = minutes.markdown

        # 保存
        if output is None:
            output_dir = Path(os.environ.get("MINUTES_DIR", "./minutes"))
            output = output_dir / f"{now.strftime('%Y%m%d-%H%M%S')}.md"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8")
        click.echo(f"  保存: {output}")

        # Slack
        if post_slack:
            click.echo("[5/5] Slack に投稿")
            channel_label = os.environ.get("SLACK_CHANNEL_LABEL", "")
            parent_text = f":memo: *{title}*" + (f"  ({channel_label})" if channel_label else "")
            try:
                result = post_to_slack_auto(markdown, parent_text=parent_text, header=parent_text)
            except SlackError as e:
                click.echo(f"  投稿失敗: {e}", err=True)
                sys.exit(2)
            if result.ok:
                mode = "スレッド化" if result.thread_ts else "フラット (Webhook)"
                click.echo(f"  投稿成功 ({mode}): {result.message_count} メッセージ")
                if result.thread_ts:
                    click.echo(f"  thread_ts: {result.thread_ts} channel: {result.channel}")
            else:
                click.echo(f"  投稿失敗: {result.error}", err=True)
                sys.exit(2)
        else:
            click.echo("[5/5] Slack 投稿スキップ（--post-slack で投稿）")


@main.group()
def roster() -> None:
    """参加者ロスター（名簿）の管理。

    batch-process 後の correct で「誰の声か」を判定する際の正規名リスト。
    """


@roster.command(name="add")
@click.argument("names", nargs=-1, required=True)
@click.option("--note", default="", help="メモ（任意）")
def roster_add(names: tuple[str, ...], note: str) -> None:
    """参加者を追加（複数可: roster add 山田 鈴木 田中）。"""
    db = RosterDB()
    try:
        for name in names:
            db.add(name, note=note)
            click.echo(f"  + {name}")
    finally:
        db.close()


@roster.command(name="list")
def roster_list() -> None:
    """登録参加者を一覧。"""
    db = RosterDB()
    try:
        members = db.list_all()
        if not members:
            click.echo("ロスター未登録。`voice-shiwake roster add 名前 ...` で登録してください。")
            return
        for m in members:
            note = f" - {m.note}" if m.note else ""
            click.echo(f"- {m.name}{note}")
        click.echo(f"\n計 {len(members)} 名")
    finally:
        db.close()


@roster.command(name="remove")
@click.argument("name")
def roster_remove(name: str) -> None:
    """参加者を削除。"""
    db = RosterDB()
    try:
        if db.remove(name):
            click.echo(f"削除: {name}")
        else:
            click.echo(f"見つからず: {name}", err=True)
            sys.exit(1)
    finally:
        db.close()


@main.command(name="batch-process")
@click.option(
    "--videos-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="動画ファイルが入ったディレクトリ",
)
@click.option(
    "--pattern",
    default="*.mp4,*.mov,*.m4a,*.mkv,*.webm",
    help="対象ファイルの glob パターン（カンマ区切り）",
)
@click.option(
    "--output-base",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("work/batch"),
    help="各動画の出力先のベースディレクトリ（既定: work/batch）",
)
@click.option("--skip-existing", is_flag=True, help="output に minutes.md があれば skip")
def batch_process(
    videos_dir: Path,
    pattern: str,
    output_base: Path,
    skip_existing: bool,
) -> None:
    """ディレクトリ内の全動画を videodb 経由で処理し、各動画ごとに samples/ + minutes.md を出力する。

    出力構造:
        output_base/
          <動画名>/
            audio.wav
            transcript_structured.json
            minutes.md
            samples/SPEAKER_A.wav, SPEAKER_B.wav, ...

    処理後は `voice-shiwake correct --samples-dir <出力先>/samples/ --source <タグ> --map ...`
    で正解人物名を紐付けて声紋DBに登録する。
    """
    patterns = [p.strip() for p in pattern.split(",") if p.strip()]
    videos: list[Path] = []
    for p in patterns:
        videos.extend(sorted(videos_dir.glob(p)))
    # 重複除外
    seen: set[str] = set()
    unique_videos: list[Path] = []
    for v in videos:
        if str(v) not in seen:
            seen.add(str(v))
            unique_videos.append(v)
    videos = unique_videos

    if not videos:
        click.echo(f"対象動画なし（{videos_dir} の pattern={pattern}）", err=True)
        sys.exit(1)

    click.echo(f"対象動画 {len(videos)} 件: {videos_dir}")
    output_base.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []

    for i, video in enumerate(videos, 1):
        out_dir = output_base / video.stem
        click.echo("=" * 60)
        click.echo(f"[{i}/{len(videos)}] {video.name} ({video.stat().st_size / 1024 / 1024:.1f}MB) → {out_dir}")
        if skip_existing and (out_dir / "minutes.md").exists():
            click.echo("  [SKIP] minutes.md 既存")
            continue
        try:
            result = process_video(video, out_dir)
            speakers_info = ", ".join(
                f"SPEAKER_{s.label}={s.total_sec:.0f}s" for s in result.speakers
            )
            click.echo(
                f"  ✓ 完了: 話者{len(result.speakers)}名, "
                f"{result.duration_sec:.0f}s音声, {result.n_utterances}発話 [{speakers_info}]"
            )
            summary.append(
                {
                    "video": str(video),
                    "output_dir": str(out_dir),
                    "videodb_id": result.videodb_id,
                    "duration_sec": result.duration_sec,
                    "n_speakers": len(result.speakers),
                    "n_utterances": result.n_utterances,
                }
            )
        except VideodbPipelineError as e:
            click.echo(f"  ✗ 失敗: {e}", err=True)
            failures.append((str(video), str(e)))
        except Exception as e:  # noqa: BLE001
            click.echo(f"  ✗ 予期せぬエラー: {e}", err=True)
            failures.append((str(video), str(e)))

    # サマリー JSON
    summary_path = output_base / "_batch_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "ran_at": datetime.now().isoformat(),
                "videos_dir": str(videos_dir),
                "n_total": len(videos),
                "n_success": len(summary),
                "n_failed": len(failures),
                "successes": summary,
                "failures": failures,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    click.echo("=" * 60)
    click.echo(f"完了: {len(summary)}/{len(videos)} 成功 ({len(failures)} 失敗)")
    click.echo(f"サマリー: {summary_path}")
    if summary:
        click.echo("\n次のステップ: 各動画の SPEAKER を correct で正解教示")
        for s in summary[:3]:
            out_dir = Path(s["output_dir"])
            click.echo(
                f"  voice-shiwake correct --samples-dir {out_dir}/samples \\"
            )
            click.echo(
                f"      --source \"<offline|online|hybrid>-{out_dir.name}\" \\"
            )
            click.echo(
                f"      --map A=<名前> --map B=<名前> ..."
            )
        if len(summary) > 3:
            click.echo(f"  ... 他 {len(summary) - 3} 件 (詳細は _batch_summary.json)")


@main.command()
@click.option("--video", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="インデックス対象動画")
@click.option("--collection", default="meetings", help="videodb の collection 名")
def index(video: Path, collection: str) -> None:
    """videodb に動画をアップロードして発話インデックスを作成。"""
    try:
        result = upload_and_index(video, collection_name=collection)
    except VideodbError as e:
        click.echo(f"videodb エラー: {e}", err=True)
        sys.exit(2)
    click.echo(f"video_id: {result.video_id}")
    click.echo(f"collection_id: {result.collection_id}")
    click.echo(f"indexed: {result.indexed} (transcript {result.transcript_chars} chars)")


@main.command()
@click.argument("query")
@click.option("--collection", default="meetings", help="検索対象 collection")
@click.option("--limit", default=5, type=int, help="返す件数")
def search(query: str, collection: str, limit: int) -> None:
    """過去のインデックス済み議事録を意味検索する。"""
    try:
        hits = semantic_search(query, collection_name=collection, limit=limit)
    except VideodbError as e:
        click.echo(f"videodb エラー: {e}", err=True)
        sys.exit(2)
    if not hits:
        click.echo("ヒットなし")
        return
    for i, hit in enumerate(hits, 1):
        click.echo(f"[{i}] video={hit.video_id} score={hit.score:.3f} ({hit.start_sec:.1f}s〜{hit.end_sec:.1f}s)")
        click.echo(f"    {hit.text}")


@main.command(name="check-deps")
def check_deps() -> None:
    """外部依存（ffmpeg / AssemblyAI / Claude / Slack / videodb）の利用可否を診断。"""
    import shutil as _shutil

    checks: list[tuple[str, bool, str]] = []

    checks.append(("ffmpeg", _shutil.which("ffmpeg") is not None, _shutil.which("ffmpeg") or "PATH に無し"))
    checks.append(("ASSEMBLYAI_API_KEY", bool(os.environ.get("ASSEMBLYAI_API_KEY")), "set" if os.environ.get("ASSEMBLYAI_API_KEY") else "未設定"))
    checks.append(("ANTHROPIC_API_KEY", bool(os.environ.get("ANTHROPIC_API_KEY")), "set" if os.environ.get("ANTHROPIC_API_KEY") else "未設定"))
    checks.append(("SLACK_WEBHOOK_URL", bool(os.environ.get("SLACK_WEBHOOK_URL")), "set (フラット投稿可)" if os.environ.get("SLACK_WEBHOOK_URL") else "未設定"))
    has_bot = bool(os.environ.get("SLACK_BOT_TOKEN")) and bool(os.environ.get("SLACK_CHANNEL_ID"))
    checks.append((
        "SLACK_BOT_TOKEN + SLACK_CHANNEL_ID",
        has_bot,
        "set (スレッド化投稿可、推奨)" if has_bot else "未設定（スレッド化したい場合は両方必要）",
    ))
    checks.append(("VIDEODB_API_KEY", bool(os.environ.get("VIDEODB_API_KEY")), "set" if os.environ.get("VIDEODB_API_KEY") else "未設定（index/search 不可）"))

    for name, available, detail in checks:
        mark = "OK" if available else "NG"
        click.echo(f"[{mark}] {name}: {detail}")


@main.command()
@click.option(
    "--audio",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="16kHz mono WAV (process_video の出力 audio.wav)",
)
@click.option(
    "--transcript-json",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="videodb の word-level JSON (transcript_structured.json)",
)
@click.option(
    "--candidates",
    required=True,
    help="カンマ区切りの候補名（例: 宮下,酒井）。声紋DBに登録済みである必要あり",
)
@click.option("--vad-mode", type=int, default=2, help="webrtcvad mode 0(緩)〜3(厳格)、既定 2")
@click.option("--min-pause-ms", type=int, default=300, help="この長さ以上の無音で segment 分割、既定 300ms")
@click.option("--min-seg-ms", type=int, default=500, help="この長さ未満の segment は捨てる、既定 500ms")
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="dialogue 出力先 Markdown",
)
def rediarize(
    audio: Path,
    transcript_json: Path,
    candidates: str,
    vad_mode: int,
    min_pause_ms: int,
    min_seg_ms: int,
    output: Path | None,
) -> None:
    """VADベースで話者再分離する（候補限定、後処理用）。

    videodb の盲目 diarization が話者をまとめて取りこぼした場合に、既知の候補名を
    指定して声紋DBから照合する。オフライン1マイクの音響条件では本質的な限界あり。
    """
    from .vad_rediarize import format_rediarized_dialogue, rediarize_with_vad

    cand_list = [c.strip() for c in candidates.split(",") if c.strip()]
    if len(cand_list) < 2:
        click.echo("候補は2名以上指定してください", err=True)
        sys.exit(2)

    click.echo(f"候補: {cand_list}")
    click.echo(f"VAD mode={vad_mode}, min_pause={min_pause_ms}ms, min_seg={min_seg_ms}ms")

    try:
        result = rediarize_with_vad(
            audio,
            transcript_json,
            cand_list,
            vad_mode=vad_mode,
            min_pause_ms=min_pause_ms,
            min_seg_ms=min_seg_ms,
        )
    except ValueError as e:
        click.echo(f"エラー: {e}", err=True)
        sys.exit(2)

    raw_n = len(result["raw_segments"])
    merged_n = len(result["segments"])
    embedded = result["embedded_count"]
    click.echo(f"VAD segment: {raw_n} (うち embed {embedded})、merge後ブロック: {merged_n}")
    for label, st in result["stats"].items():
        click.echo(f"  {label}: count={st['count']}, duration={st['duration_s']:.1f}s")
    click.echo(f"平均マージン (winner-loser cos差): {result['avg_margin']:.3f}")
    if result["avg_margin"] < 0.10:
        click.echo("  WARN: マージン < 0.10 は信頼度低（音響条件が不利）")

    dialogue = format_rediarized_dialogue(result["segments"])
    if output is None:
        output = audio.parent / "rediarized_dialogue.md"
    output.write_text(
        f"# 再分離 dialogue (候補: {', '.join(cand_list)})\n\n```\n{dialogue}\n```\n",
        encoding="utf-8",
    )
    click.echo(f"保存: {output}")


if __name__ == "__main__":
    main()
