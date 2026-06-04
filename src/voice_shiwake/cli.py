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
from .voiceprint import VoiceprintDB, identify_speakers

load_dotenv()


@click.group()
@click.version_option(package_name="voice-shiwake")
def main() -> None:
    """会議動画から議事録を生成して Slack に投稿するツール。"""


@main.command()
@click.option("--name", required=True, help="参加者の表示名（例: 田中太郎）")
@click.option("--audio", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="30秒程度の音声サンプル")
@click.option("--note", default="", help="メモ（任意）")
def enroll(name: str, audio: Path, note: str) -> None:
    """参加者の声紋を登録する。"""
    db = VoiceprintDB()
    try:
        vp = db.enroll(name=name, audio_path=audio, note=note)
        click.echo(f"登録完了: {vp.name} (embedding dim={len(vp.embedding)})")
    finally:
        db.close()


@main.command(name="list")
def list_voiceprints() -> None:
    """登録済み声紋を一覧表示。"""
    db = VoiceprintDB()
    try:
        all_vps = db.list_all()
        if not all_vps:
            click.echo("登録済み声紋なし。`voice-shiwake enroll` で追加してください。")
            return
        for vp in all_vps:
            note = f" - {vp.note}" if vp.note else ""
            click.echo(f"- {vp.name}{note}")
    finally:
        db.close()


@main.command()
@click.option("--name", required=True, help="削除する声紋名")
def delete(name: str) -> None:
    """登録済み声紋を削除。"""
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
