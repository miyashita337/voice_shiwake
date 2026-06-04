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

import click
from dotenv import load_dotenv

from .audio import extract_audio
from .minutes import summarize_to_minutes
from .slack import post_to_slack
from .transcribe import format_as_dialogue, transcribe
from .videodb_adapter import VideodbError, semantic_search, upload_and_index
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
            channel = os.environ.get("SLACK_CHANNEL_LABEL", "")
            header = f":memo: *{title}*" + (f"  ({channel})" if channel else "")
            result = post_to_slack(markdown, header=header)
            if result.ok:
                click.echo(f"  投稿成功: {result.message_count} メッセージ")
            else:
                click.echo(f"  投稿失敗: {result.error}", err=True)
                sys.exit(2)
        else:
            click.echo("[5/5] Slack 投稿スキップ（--post-slack で投稿）")


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
    checks.append(("SLACK_WEBHOOK_URL", bool(os.environ.get("SLACK_WEBHOOK_URL")), "set" if os.environ.get("SLACK_WEBHOOK_URL") else "未設定"))
    checks.append(("VIDEODB_API_KEY", bool(os.environ.get("VIDEODB_API_KEY")), "set" if os.environ.get("VIDEODB_API_KEY") else "未設定（index/search 不可）"))

    for name, available, detail in checks:
        mark = "OK" if available else "NG"
        click.echo(f"[{mark}] {name}: {detail}")


if __name__ == "__main__":
    main()
