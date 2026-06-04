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

### 1. 参加者の声紋を登録（継続学習対応）

各メンバーの **30秒程度** の音声サンプルを用意して `enroll`:

```bash
voice-shiwake enroll --name "田中太郎" --audio samples/tanaka.wav
voice-shiwake enroll --name "佐藤花子" --audio samples/sato.wav
```

**同名で複数回 enroll すると追加サンプルとして蓄積されます**（既定動作）。

```bash
# 同じ田中さんの別会議音声を追加（精度向上）
voice-shiwake enroll --name "田中太郎" --audio samples/tanaka_meeting2.wav --source "weekly-2"
voice-shiwake list
# - 田中太郎: 2 samples [weekly-1, weekly-2]
```

サンプルを丸ごと差し替えたい場合は `--replace`:

```bash
voice-shiwake enroll --name "田中太郎" --audio samples/new.wav --replace
```

### 2. 会議処理後に正解を教える（フィードバックループ）

`process` 実行で `SPEAKER_A/B/C` のラベルが残った場合、各話者の代表音声を
ローカルに保存して `correct` で正解人物名を紐付けます:

```bash
# 会議の各話者の代表音声を抽出（process が自動で work/samples/ に保存）
ls work/samples/
# SPEAKER_A.wav  SPEAKER_B.wav  SPEAKER_C.wav

# Slack で議事録を見て「SPEAKER_A は渡辺、B は小川、C は林」と判明
voice-shiwake correct --samples-dir work/samples/ \
  --map A=山田太郎 \
  --map B=小川 \
  --map C=林

# 履歴で監査
voice-shiwake history
# [2026-06-04 17:30] SPEAKER_A → 山田太郎 (source=samples)
# [2026-06-04 17:30] SPEAKER_B → 小川 (source=samples)
```

**この訂正がそのまま追加サンプルとして DB に保存され、次回会議から識別精度が向上します。**

### 3. 識別ロジック

- 1名あたり N サンプル登録 → クエリ embedding と **全サンプル中の最大コサイン類似度** で判定
- 外れ値（古い・ノイジーな1サンプル）に強い
- サンプル数 ≥ 3 で精度大幅向上、≥ 5 で安定
- `centroid` 類似度も並行計算しており、`match_detail()` で取得可能（分析用）

### サンプル品質チェック（混在検出）

SPEAKER_X.wav に**複数人の声が混ざる**ケースは実際に起こります:

- diarization の誤分離（videodb が別人を同じラベルに）
- 同時発話の被り（A が話している裏で B の相槌）
- BGM・ハウリング・他人のマイク混信

`enroll` / `correct` は自動的に**内部一貫性スコア**を計算して警告します:

```
[1/2] 品質チェック: SPEAKER_B.wav
  [OK] サンプル一貫性スコア 0.912 (chunks=18, min=0.851)
[2/2] 声紋DB に登録
登録完了 (追加): 小川 sample_id=0 (累計 1 sample)
```

混在が疑われる場合:

```
  ⚠ サンプル一貫性スコア 0.62 (推奨 ≥ 0.85)。複数話者混在の可能性...
```

| スコア | 判定 | 既定動作 | `--strict` 動作 |
|---|---|---|---|
| ≥ 0.85 | 単一話者 | 続行 | 続行 |
| 0.70〜0.85 | グレー | 警告→続行 | 警告→続行 |
| < 0.70 | **混在の疑い** | 警告→続行 | **exit 3 で登録中止** |

検出後の対処:

1. SPEAKER_X.wav を Audacity 等で開き、他人の声を切り取って `enroll` し直す
2. 別の会議で再度 `process` してきれいな SPEAKER_X.wav を取り直し
3. どうしてもダメなら強制登録: `--no-quality-check`

### 大量動画から一気に声紋を蓄積する（batch ワークフロー）

参加者ロスター登録 → 動画一括処理 → 各動画の正解を順番に教える流れ:

```bash
# 1. 参加者を全員ロスター登録（typo チェックに使われる）
voice-shiwake roster add 山田 鈴木 田中 佐藤 高橋 渡辺 中村 小林 加藤 吉田 山口 松井

voice-shiwake roster list
# - 山田
# - 鈴木
# - 田中
# ...
# 計 12 名

# 2. 全動画を1つのディレクトリにまとめて batch-process
voice-shiwake batch-process \
  --videos-dir /path/to/all_meetings/ \
  --output-base work/batch/ \
  --skip-existing
# → 動画ごとに work/batch/<動画名>/samples/SPEAKER_*.wav と minutes.md を生成

# 3. 動画ごとに会議タイプ別タグを付けて correct
# オフライン会議
voice-shiwake correct --samples-dir work/batch/meeting1/samples/ \
  --source "offline-2026-06-01" \
  --map A=渡辺 --map B=山田 --map C=鈴木

# オンライン会議
voice-shiwake correct --samples-dir work/batch/meeting2/samples/ \
  --source "online-2026-06-08" \
  --map A=渡辺 --map B=田中 --map C=高橋

# ハイブリッド会議
voice-shiwake correct --samples-dir work/batch/meeting3/samples/ \
  --source "hybrid-2026-06-15" \
  --map A=渡辺 --map B=山田 --map C=中村

# 4. 蓄積状況の確認
voice-shiwake list
# - 渡辺: 5 samples [offline-2026-06-01, online-..., hybrid-...]
# - 山田: 3 samples [offline-2026-06-01, hybrid-...]
# ...
```

**会議タイプ別 source タグ**を `--source` で渡すことで、同一人物の embedding が
音響環境ごとに識別タグ付きで蓄積されます。`max similarity` 採用なので、
どの環境のサンプルでも「最も近い1個」が判定材料になります。

ハイブリッド会議の罠 (会議室マイク経由の声 vs 個人マイク) は、両方の環境の
サンプルを同じ名前で蓄積することで自然に解消されます。

### サンプルを増やすには（質問②への回答）

| 方法 | 工数 | 効果 |
|---|---|---|
| **A. 各人に30秒の自己紹介を録音してもらい enroll** | 1人5分 | 即時1サンプル、最確実 |
| **B. 過去の会議動画 N 本を `process` → `correct`** | 動画N本のAPIコスト | 一発でNサンプル蓄積 |
| **C. 毎週の定例で `process` → ズレてたら `correct`** | 0（自然蓄積） | 数週間で5サンプル到達 |

**推奨フロー**:
1. **初回**: A で 1サンプル登録（最初の精度ベースを作る）
2. **以降**: C で会議のたびに `correct` 実行 → 自然に蓄積
3. **B のショートカット**: 過去会議が手元にあれば一気に蓄積可能

```bash
# B案: 過去動画から一気に蓄積する例
for video in past_meetings/*.mp4; do
  voice-shiwake process --video "$video" --save-json work/$(basename "$video").json
  # 各動画の話者を Slack 等で確認後
  voice-shiwake correct --samples-dir work/$(basename "$video" .mp4)/samples \
    --map A=山田太郎 --map B=小川 --map C=林
done
voice-shiwake list
# - 山田太郎: 5 samples [video1, video2, video3, video4, video5]
```

### 4. 会議動画を処理

```bash
voice-shiwake process --video meeting_2026_06_04.mp4 \
  --title "週次定例 2026-06-04" \
  --post-slack
```

### 5. （任意）videodb でアーカイブ＋検索

```bash
# 動画を videodb にインデックス化
voice-shiwake index --video meeting_2026_06_04.mp4

# 過去議事録を意味検索
voice-shiwake search "予算の話"
```

### 6. 環境診断

```bash
voice-shiwake check-deps
# ffmpeg / 各API キーの利用可否を一覧表示
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

- **話者識別は AssemblyAI + Resemblyzer**: AssemblyAI で diarization、Resemblyzer の声紋 embedding で個人名マッピング
- **videodb は意味検索レイヤとして採用**: 月10時間規模では必須ではないが、議事録アーカイブの意味検索という独自価値を持つため `index` / `search` サブコマンドで利用可能
- **VibeVoice 不採用**: 当初検討したが TTS 専用モデルであり本ツールの用途（話者識別）には根本的に不適合。公式リポも 2025-09 に Microsoft が無効化済
- **ローカルGPU 不採用**: 月10時間程度なら API 課金の方が安く運用負担も軽い

## ライセンス

MIT
