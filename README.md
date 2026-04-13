# toolkit-audio

音声・動画ファイルの文字起こしツール。Windows（CUDA GPU）/ Mac（CPU）の両環境で動作する。

もともと [arano-bot](https://github.com/nomuraya-job-fde/arano-bot) の文字起こしスクリプト群として開発されたが、
プロジェクト横断で使えるようツールキットとして切り出した。

## 対応フォーマット

`m4a` `mp4` `mp3` `wav` `flac` `ogg` `webm`

## セットアップ

### 共通

```bash
# uv がなければインストール
# Mac:
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell):
irm https://astral.sh/uv/install.ps1 | iex

# 仮想環境作成
uv venv
```

### Mac（CPU）

```bash
uv pip install faster-whisper
```

### Windows（CUDA GPU推奨）

```bash
# CUDA 12.x 対応 PyTorch + faster-whisper
uv pip install faster-whisper torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

# ffmpeg（文字起こしの前処理に必要）
# winget install Gyan.FFmpeg
```

> **⚠ `uv sync` は実行しないこと**
> pyproject.toml の依存が不足する場合は `uv pip install` で個別追加すること。

### 環境確認

```bash
uv run python transcribe.py --check
```

## 使い方

```bash
# 単発処理（出力は入力ファイルと同じディレクトリに .md で保存）
uv run python transcribe.py recording.m4a

# 出力先を指定（festaセッションフォルダへ直接）
uv run python transcribe.py recording.m4a \
  --out ~/workspace-ai/nomuraya-strategy/festa/sessions/

# 複数ファイルを一括処理
uv run python transcribe.py *.m4a --out ./output/

# モデル・精度を変更（デフォルト: large-v3 / float32）
uv run python transcribe.py recording.m4a --model medium --compute-type float16

# 対象確認のみ（処理しない）
uv run python transcribe.py *.m4a --dry-run

# 既存出力ファイルを上書き
uv run python transcribe.py recording.m4a --force

# 話者ラベルを実名に置換（文字起こし時）
uv run python transcribe.py recording.m4a --speakers 2 \
  --rename-speakers SPEAKER_00=shima SPEAKER_01=chisuzu

# 既存.mdファイルの話者ラベルを一括置換
uv run python transcribe.py session.md \
  --rename-speakers SPEAKER_00=shima SPEAKER_01=chisuzu
```

### オプション一覧

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--out` | 入力と同じ | 出力ディレクトリ |
| `--model` | `large-v3` | Whisper モデル名 |
| `--device` | `auto` | `auto` / `cuda` / `cpu` |
| `--compute-type` | `float32` | `float32` / `float16` / `int8` |
| `--language` | `ja` | 言語コード |
| `--check` | — | 環境確認のみ |
| `--dry-run` | — | 対象確認のみ |
| `--force` | — | 既存ファイルも上書き |
| `--rename-speakers` | — | 話者ラベル置換（`OLD=NEW`形式、複数可） |

### 出力形式

Markdown 形式で保存される。ファイル名は `YYYYMMDD-{元のファイル名}.md`。

```markdown
# 20260413-booth-strategy

録音ファイル: `booth-strategy.m4a`
文字起こし日時: 2026-04-13 14:30

---

[0.000] えーと、今日はブース設営について話したいんですが
[3.200] まず敷き布の使い方から考えたいと思います
...
```

## モデルの選び方

| モデル | VRAM目安 | 速度 | 精度 | 用途 |
|--------|---------|------|------|------|
| `tiny` | 1GB | 最速 | 低 | 動作確認 |
| `base` | 1GB | 速い | 中 | 短いメモ |
| `medium` | 5GB | 普通 | 高 | 日常用途 |
| `large-v3` | 10GB | 遅い | 最高 | 本番用途 |

Mac CPU では `large-v3` でも動くが、1時間音声で30〜60分程度かかる。
Windows CUDA（RTX 3060以上）なら `large-v3 float16` で5〜10分。

## プロジェクトでの使い方

各プロジェクトからは絶対パスで呼び出す。

```bash
# festa（ブース戦略の壁打ち録音）
python ~/workspace-ai/nomuraya-operation/toolkit-audio/transcribe.py \
  "$USERPROFILE/Documents/Sound recordings/recording.m4a" \
  --out ~/workspace-ai/nomuraya-strategy/festa/sessions/

# arano-bot（Zoom録音の一括処理）
python ~/workspace-ai/nomuraya-operation/toolkit-audio/transcribe.py \
  *.mp4 \
  --out ~/workspace-ai/nomuraya-job-fde/arano-bot/data/transcripts/raw/
```

---

## ロードマップ

### 現在（v0.1 - nomuraya-operation）

- [x] Windows CUDA / Mac CPU 自動判定
- [x] m4a / mp4 / wav 等の主要フォーマット対応
- [x] Markdown 出力（セッションファイル直接生成）
- [x] 既存ファイルのスキップ / --force 上書き
- [ ] 話者分離オプション（`--diarize`）
- [ ] 後処理: LLM によるサマリー生成（`--summarize`）
- [ ] 後処理: タスク抽出・Issue 自動起票（`--extract-tasks`）

### スピンアウト検討条件（→ 独立オーガニゼーション `nomuraya-tools`）

以下のいずれかに該当する場合、`nomuraya-tools/toolkit-audio` として独立リポジトリ化する:

- **利用プロジェクトが3つ以上**になった時（現在: festa, arano-bot）
- **話者分離・LLM後処理**など nomuraya-operation の責務を超える機能が増えた時
- **他オーガニゼーションのメンバーが使いたい**ケースが発生した時

スピンアウト後は `nomuraya-operation/toolkit-audio` を archiveし、
各プロジェクトの呼び出しパスを更新する（移行コストは低い）。

### 将来の機能候補

- AssemblyAI バックエンド対応（クラウドASR、話者分離精度が高い）
- リアルタイム文字起こし（録音しながら同時変換）
- Discord 通話録音との統合（Craig Bot 出力の自動処理）
- YouTube / Zoom 録画の直接取り込み
