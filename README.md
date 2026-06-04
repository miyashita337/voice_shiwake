# voice_shiwake

会議動画から **話者別議事録** を生成して **Slack に投稿** するツール。

## 何ができるか

1. 動画ファイル（Zoom/Teams録画など）を入力
2. AssemblyAI で日本語文字起こし + 話者分離（diarization）
3. 事前登録した参加者の声紋と照合 → `SPEAKER_A → 田中` のように個人名ラベル化
4. Claude API で議事録形式に整形（要約・決定事項・ActionItem）
5. Slack に Incoming Webhook で投稿

## アーキテクチャ

```
動画ファイル
  └─ ffmpeg ───────────────────┐
                              ↓
              音声(WAV/16kHz/mono)
                ├──────────────────────┐
                ↓                      ↓
       AssemblyAI                Resemblyzer
       (ASR + diarization)       (speaker embedding)
                │                      │
                └─── 名寄せ ───────────┘
                      ↓
              話者ラベル付き文字起こし
                      ↓
                Claude Sonnet 4.6
                (議事録整形)
                      ↓
              Slack Incoming Webhook
```

## セットアップ

```bash
git clone https://github.com/miyashita337/voice_shiwake.git
cd voice_shiwake

# 推奨: 仮想環境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 依存
pip install -e ".[dev]"

# ffmpeg を入れる（必須）
# macOS:   brew install ffmpeg
# Ubuntu:  sudo apt install -y ffmpeg
# Windows: winget install ffmpeg

# 環境変数
cp .env.example .env
# .env を編集して APIキー / Webhook URL を設定
```

## 使い方

### 1. 参加者の声紋を登録

各メンバーの **30秒程度** の音声サンプル（自己紹介など）を用意してから:

```bash
voice-shiwake enroll --name "田中太郎" --audio samples/tanaka.wav
voice-shiwake enroll --name "佐藤花子" --audio samples/sato.wav
voice-shiwake enroll --name "鈴木一郎" --audio samples/suzuki.wav
```

登録済みリスト:

```bash
voice-shiwake list
```

### 2. 会議動画を処理

```bash
voice-shiwake process --video meeting_2026_06_04.mp4 \
  --title "週次定例 2026-06-04" \
  --post-slack
```

オプション:

| フラグ | 説明 |
|---|---|
| `--video PATH` | 入力動画（必須） |
| `--title TEXT` | 議事録のタイトル |
| `--post-slack` | Slack に投稿（省略時は標準出力＋ローカル保存のみ） |
| `--output PATH` | Markdown 議事録の保存先（既定: `minutes/<datetime>.md`） |
| `--no-identify` | 声紋照合スキップ（`SPEAKER_A` ラベルのまま） |
| `--no-summarize` | Claude 整形スキップ（生の文字起こしを出力） |

## コスト感（月10本×1時間想定）

| 項目 | コスト |
|---|---|
| AssemblyAI | $0.37 × 10 ≒ **600円** |
| Claude Sonnet 4.6 | 数百円 |
| Slack / Resemblyzer / ffmpeg | 0円 |
| **合計** | **月1,000円以内** |

## 精度の目安

- ASR（日本語）: 実用十分
- 話者分離: クローズドマイク 90%+ / Zoom mix 録画 75-85%
- 声紋識別: cos類似度 0.75 以上で名寄せ。同性で似た声は誤りやすい

## 設計判断

- **VibeVoice 不採用**: Microsoft の VibeVoice は TTS（音声合成）専用で話者識別には使えない。さらに 2025-09 にリポジトリ無効化済み
- **videodb 不採用**: 月10時間規模ではオーバースペック。過去議事録検索が必要になった段階で追加検討
- **ローカルGPU 不採用**: 月10時間程度なら API 課金の方が安く運用負担も軽い

## ライセンス

MIT
