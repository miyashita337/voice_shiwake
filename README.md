# voice_shiwake

会議動画から **話者別議事録** を生成して **Slack に投稿** するツール。

## 何ができるか

1. 動画ファイル（Zoom/Teams録画など）を入力
2. AssemblyAI で日本語文字起こし + 話者分離（diarization）
3. 事前登録した参加者の声紋と照合 → `SPEAKER_A → 田中` のように個人名ラベル化
4. Claude API で議事録形式に整形（要約・決定事項・ActionItem）
5. Slack に Incoming Webhook で投稿

## 構成ツール役割マップ

| ツール | 役割 | 必須？ | 備考 |
|---|---|---|---|
| **AssemblyAI** | ASR (日本語文字起こし) + 話者分離 (diarization) | ✅ 必須 | 月数百円〜 |
| **Resemblyzer** | 声紋 embedding（個人名マッピング） | ✅ 必須 | ローカル、無料 |
| **Claude Sonnet 4.6** | 議事録形式に整形・要約 | ✅ 必須 | 月数百円〜 |
| **Slack Webhook** | 議事録の投稿先 | ✅ 必須 | 無料 |
| **videodb** | 過去会議の意味検索（`index` / `search` サブコマンド） | △ 任意 | 議事録履歴が貯まったら有用 |
| **VibeVoice (TTS)** | 議事録の音声化（`tts` サブコマンド、移動中の耳レビュー用） | △ 任意 | 公式リポ無効化済（後述） |

### VibeVoice について（注意）

- VibeVoice は **TTS（音声合成）モデル**であり、話者識別には使えない（事実）
- 2025-09 に Microsoft が公式リポジトリを無効化
- 本ツールでは「生成した議事録を音声化し、移動中に耳でレビューする」用途で組み込み済み
- 利用にはローカルに HuggingFace 互換のモデル重みを配置し、`VIBEVOICE_MODEL_PATH` を設定
- 未配置でもパイプライン本体（議事録生成→Slack投稿）は影響なく動作

### videodb について

- 本ツールでは `index_spoken_words` で発話インデックス化、`search` で意味検索
- 月10時間規模の運用なら必須ではないが、議事録アーカイブが半年以上貯まったとき有用
- 「予算の話どこの会議だっけ？」のような自然言語検索が可能

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

### 3. （任意）videodb でアーカイブ＋検索

```bash
# 動画を videodb にインデックス化
voice-shiwake index --video meeting_2026_06_04.mp4

# 過去議事録を意味検索
voice-shiwake search "予算の話"
```

### 4. （任意）VibeVoice で議事録を音声化

```bash
# 配置済みのモデルパスを .env に設定後
voice-shiwake tts --minutes-md minutes/20260604-1500.md --output minutes.wav
```

### 5. 環境診断

```bash
voice-shiwake check-deps
# ffmpeg / 各API キー / VibeVoice 利用可否を一覧表示
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

- **話者識別の主役は AssemblyAI + Resemblyzer**: VibeVoice は TTS であり、話者識別には根本的に不向きなため
- **VibeVoice は TTS アダプタとして採用**: 議事録の音声化（耳レビュー）用に組み込み。公式リポ無効化済のためローカル重み配置前提
- **videodb は意味検索レイヤとして採用**: 月10時間規模では必須ではないが、議事録の意味検索という独自価値を持つため `index` / `search` サブコマンドで利用可能
- **ローカルGPU 不採用**: 月10時間程度なら API 課金の方が安く運用負担も軽い

## ライセンス

MIT
