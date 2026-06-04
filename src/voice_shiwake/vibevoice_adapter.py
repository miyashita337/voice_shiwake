"""VibeVoice TTS アダプタ（議事録の音声化）。

Microsoft VibeVoice は long-form / multi-speaker 対応の TTS モデル。
本ツールでは「生成した議事録を音声化して、移動中に耳でレビューする」
ユースケースで利用する（speaker identification 用途ではないので注意）。

⚠️ 注意:
- 公式リポジトリは 2025-09 に Microsoft により無効化済み
  (https://microsoft.github.io/VibeVoice/ の Status note 参照)
- 本アダプタは「ローカルに HuggingFace 互換重みが配置済み」前提で動作
- 使えない場合は明示的なエラーを返し、議事録パイプライン本体は影響を受けない
- 代替として pyttsx3 / Edge TTS / ElevenLabs などへの差し替えが容易になるよう
  TTSBackend Protocol で抽象化

設定:
- VIBEVOICE_MODEL_PATH: ローカルモデルディレクトリ
  (例: /models/vibevoice-1.5b)
- VIBEVOICE_DEVICE: "cuda" / "cpu" / "mps" (既定: 自動判定)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class VibeVoiceError(RuntimeError):
    pass


@dataclass
class TTSResult:
    output_path: Path
    sample_rate: int
    duration_sec: float
    backend: str


class TTSBackend(Protocol):
    """TTS バックエンドの最小インターフェース。"""

    name: str

    def synthesize(self, text: str, output_path: Path) -> TTSResult: ...


def check_availability(model_path: str | None = None) -> tuple[bool, str]:
    """VibeVoice が現環境で利用可能か検査する。

    Returns:
        (available, reason): available=True なら使用可能、False ならエラー理由
    """
    model_path = model_path or os.environ.get("VIBEVOICE_MODEL_PATH")
    if not model_path:
        return (
            False,
            "VIBEVOICE_MODEL_PATH 未設定。公式リポジトリは無効化済みのため、"
            "HuggingFace ミラー等から重みをローカル配置し、パスを指定してください。",
        )
    if not Path(model_path).exists():
        return False, f"モデルパスが存在しない: {model_path}"
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return (
            False,
            "torch / transformers 未インストール。"
            "`pip install torch transformers` を実行してください。",
        )
    return True, "OK"


class VibeVoiceBackend:
    """HuggingFace transformers 経由で VibeVoice をロードする。

    参考: https://huggingface.co/microsoft/VibeVoice-1.5B (リポ無効化以前の構造)
    本実装は AutoModel / AutoProcessor 互換 API を想定。
    実モデルの API 差異がある場合は本クラスを書き換えるだけで対応可能。
    """

    name = "vibevoice"

    def __init__(self, model_path: str | None = None, device: str | None = None):
        ok, reason = check_availability(model_path)
        if not ok:
            raise VibeVoiceError(reason)
        self.model_path = model_path or os.environ["VIBEVOICE_MODEL_PATH"]
        self.device = device or os.environ.get("VIBEVOICE_DEVICE") or self._auto_device()
        self._model = None
        self._processor = None

    @staticmethod
    def _auto_device() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as e:
            raise VibeVoiceError("transformers 未インストール") from e
        try:
            self._processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            self._model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True).to(self.device)
        except Exception as e:  # noqa: BLE001
            raise VibeVoiceError(
                f"VibeVoice モデルロード失敗 ({self.model_path}): {e}"
            ) from e

    def synthesize(self, text: str, output_path: Path) -> TTSResult:
        """テキストを WAV に変換する。

        Args:
            text: 読み上げ対象テキスト（Markdown は事前に plain text 化推奨）
            output_path: 出力 WAV のパス

        Returns:
            TTSResult
        """
        self._ensure_loaded()
        try:
            import soundfile as sf
            import torch
        except ImportError as e:
            raise VibeVoiceError(
                "soundfile / torch 未インストール"
            ) from e

        with torch.no_grad():
            inputs = self._processor(text=text, return_tensors="pt").to(self.device)  # type: ignore[union-attr]
            outputs = self._model(**inputs)  # type: ignore[misc]
            # API 差異: 多くの TTS モデルは waveform / audio フィールドを返す
            waveform = getattr(outputs, "waveform", None)
            if waveform is None:
                waveform = getattr(outputs, "audio", None)
            if waveform is None:
                raise VibeVoiceError(
                    "モデル出力に waveform / audio フィールドが見つかりません。"
                    "VibeVoice の API に合わせて vibevoice_adapter.synthesize を調整してください。"
                )
            waveform = waveform.detach().cpu().numpy().squeeze()

        sample_rate = int(getattr(self._processor, "sampling_rate", 24000))  # type: ignore[union-attr]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), waveform, sample_rate)
        duration = float(len(waveform)) / sample_rate

        return TTSResult(
            output_path=output_path,
            sample_rate=sample_rate,
            duration_sec=duration,
            backend=self.name,
        )


def markdown_to_speech_text(markdown: str, max_chars: int = 4000) -> str:
    """議事録 Markdown を TTS 向けの plain text に変換する。

    - Markdown 記号（`#`, `-`, `|`, `*`, `[]()`）を除去
    - テーブル区切り行を除去
    - max_chars を超える場合は先頭から切り詰め
    """
    lines: list[str] = []
    for raw in markdown.splitlines():
        line = raw.strip()
        if not line:
            lines.append("")
            continue
        # テーブル区切り行
        if set(line.replace(":", "").replace("|", "")) <= {"-", " "}:
            continue
        # 見出し記号を除去
        if line.startswith("#"):
            line = line.lstrip("#").strip()
            lines.append(line + "。")
            continue
        # 箇条書き
        if line.startswith(("- ", "* ", "+ ")):
            line = line[2:].strip()
        # テーブル行 → セルを「、」で連結
        if line.startswith("|") and line.endswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|") if c.strip()]
            line = "、".join(cells)
        # リンク [text](url) → text のみ
        while "[" in line and "](" in line and ")" in line:
            try:
                lb = line.index("[")
                rb = line.index("](", lb)
                rp = line.index(")", rb)
                line = line[:lb] + line[lb + 1 : rb] + line[rp + 1 :]
            except ValueError:
                break
        lines.append(line)
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "（以下省略）"
    return text


def synthesize_minutes(
    markdown: str,
    output_path: Path,
    *,
    backend: TTSBackend | None = None,
) -> TTSResult:
    """議事録 Markdown を音声化する。

    Args:
        markdown: 議事録本文
        output_path: 出力 WAV
        backend: TTS バックエンド（既定: VibeVoiceBackend）

    Returns:
        TTSResult
    """
    backend = backend or VibeVoiceBackend()
    text = markdown_to_speech_text(markdown)
    return backend.synthesize(text, output_path)
