r"""
音声・動画ファイルの文字起こしツール（Windows/Mac対応）

対応フォーマット: m4a, mp4, mp3, wav, flac, ogg, webm
エンジン: faster-whisper（ローカル）
デバイス: Windows CUDA GPU / Mac CPU（自動判定）

使い方:
    # 単発処理
    uv run python transcribe.py recording.m4a

    # 出力先を指定
    uv run python transcribe.py recording.m4a --out ~/workspace-ai/nomuraya-strategy/festa/sessions/

    # モデル・精度を指定
    uv run python transcribe.py recording.m4a --model large-v3 --compute-type float16

    # 複数ファイルを一括処理
    uv run python transcribe.py *.m4a --out ./output/

    # 環境確認
    uv run python transcribe.py --check

    # 対象確認のみ（処理しない）
    uv run python transcribe.py *.mp4 --dry-run
"""
import argparse
import os
import platform
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path


SUPPORTED_EXTENSIONS = {".m4a", ".mp4", ".mp3", ".wav", ".flac", ".ogg", ".webm"}


# ──────────────────────────────────────────────
# Windows CUDA ランタイム補助
# ──────────────────────────────────────────────

def configure_windows_cuda_runtime() -> list[Path]:
    """Windows で ctranslate2 が見つけにくい CUDA DLL を PATH へ追加する。"""
    if os.name != "nt":
        return []

    home = Path.home()
    candidates: list[Path] = []

    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidates.append(Path(cuda_path) / "bin")

    candidates.extend(
        sorted((home / "AppData" / "Local" / "Programs" / "Python").glob(
            "Python*/Lib/site-packages/torch/lib"
        ))
    )
    candidates.append(
        home / "AppData" / "Local" / "Programs" / "Ollama" / "lib" / "ollama" / "cuda_v12"
    )

    added: list[Path] = []
    for dll_dir in candidates:
        if not dll_dir.exists() or dll_dir in added:
            continue
        try:
            os.add_dll_directory(str(dll_dir))
        except (AttributeError, FileNotFoundError):
            pass
        os.environ["PATH"] = f"{dll_dir}{os.pathsep}{os.environ.get('PATH', '')}"
        added.append(dll_dir)

    return added


# ──────────────────────────────────────────────
# デバイス判定
# ──────────────────────────────────────────────

def detect_cuda() -> bool:
    configure_windows_cuda_runtime()
    try:
        import ctranslate2
        types = ctranslate2.get_supported_compute_types("cuda")
        return bool(types)
    except Exception:
        return False


def resolve_device(device_arg: str) -> str:
    if device_arg == "cpu":
        return "cpu"
    if device_arg == "cuda":
        if not detect_cuda():
            raise RuntimeError("CUDA が利用できません。--device cpu を指定してください。")
        return "cuda"
    # auto
    return "cuda" if detect_cuda() else "cpu"


def resolve_compute_type(compute_type: str, device: str) -> str:
    if device == "cpu" and compute_type == "float16":
        print("警告: CPU では float16 を使えないため float32 に切り替えます")
        return "float32"
    return compute_type


# ──────────────────────────────────────────────
# 環境チェック
# ──────────────────────────────────────────────

def check_environment(device_arg: str, model_name: str, compute_type: str):
    dll_dirs = configure_windows_cuda_runtime()

    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"ffmpeg: {shutil.which('ffmpeg') or '未検出 ⚠'}")

    try:
        import faster_whisper
        print(f"faster-whisper: {getattr(faster_whisper, '__version__', 'installed')}")
    except ImportError:
        print("faster-whisper: 未インストール ⚠  →  uv pip install faster-whisper")

    try:
        import ctranslate2
        print(f"ctranslate2: {ctranslate2.__version__}")
    except ImportError:
        print("ctranslate2: 未インストール ⚠")

    cuda_ok = detect_cuda()
    print(f"CUDA: {'利用可能' if cuda_ok else '利用不可（CPU実行になります）'}")

    if dll_dirs:
        print("追加 DLL パス:")
        for d in dll_dirs:
            print(f"  {d}")

    device = resolve_device(device_arg)
    ct = resolve_compute_type(compute_type, device)
    print(f"実行設定: device={device}, model={model_name}, compute_type={ct}")
    print("環境チェック完了")


# ──────────────────────────────────────────────
# モデルロード・文字起こし
# ──────────────────────────────────────────────

def load_model(model_name: str, device: str, compute_type: str):
    configure_windows_cuda_runtime()
    from faster_whisper import WhisperModel
    print(f"モデルロード中... (model={model_name}, device={device}, compute_type={compute_type})")
    return WhisperModel(model_name, device=device, compute_type=compute_type)


def transcribe_file(audio: Path, model, language: str) -> list[str]:
    print(f"  文字起こし中: {audio.name}")
    t0 = time.time()

    segments, _ = model.transcribe(
        str(audio),
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    lines = []
    for seg in segments:
        lines.append(f"[{seg.start:.3f}] {seg.text.strip()}")

    elapsed = time.time() - t0
    print(f"  完了: {elapsed:.0f}秒, {len(lines)} セグメント")
    return lines


# ──────────────────────────────────────────────
# 出力ファイル名の決定
# ──────────────────────────────────────────────

def resolve_output_path(audio: Path, out_dir: Path | None) -> Path:
    date_prefix = datetime.now().strftime("%Y%m%d")
    stem = audio.stem
    # 既に日付プレフィックスがあればそのまま
    if not stem[:8].isdigit():
        stem = f"{date_prefix}-{stem}"
    out = (out_dir or audio.parent) / f"{stem}.md"
    return out


def write_markdown(lines: list[str], audio: Path, out_path: Path):
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    header = f"# {out_path.stem}\n\n録音ファイル: `{audio.name}`  \n文字起こし日時: {date_str}\n\n---\n\n"
    body = "\n".join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + body + "\n", encoding="utf-8")
    print(f"  保存: {out_path}")


# ──────────────────────────────────────────────
# エントリーポイント
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="音声・動画ファイルを文字起こしする（Windows/Mac対応）")
    parser.add_argument("files", nargs="*", help="処理する音声/動画ファイル")
    parser.add_argument("--out", type=Path, default=None, help="出力ディレクトリ（省略時は入力ファイルと同じ場所）")
    parser.add_argument("--model", default="large-v3", help="Whisper モデル名（デフォルト: large-v3）")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--compute-type", default="float32", choices=["float32", "float16", "int8"])
    parser.add_argument("--language", default="ja", help="言語コード（デフォルト: ja）")
    parser.add_argument("--check", action="store_true", help="環境確認のみ")
    parser.add_argument("--dry-run", action="store_true", help="対象確認のみ（処理しない）")
    parser.add_argument("--force", action="store_true", help="既存の出力ファイルも上書き")
    args = parser.parse_args()

    if args.check:
        check_environment(args.device, args.model, args.compute_type)
        return

    if not args.files:
        parser.print_help()
        sys.exit(1)

    # ファイル収集
    targets: list[Path] = []
    for pattern in args.files:
        p = Path(pattern)
        if p.exists():
            targets.append(p)
        else:
            # glob展開（シェルが展開しない場合の保険）
            matched = list(Path(".").glob(pattern))
            targets.extend(matched)

    targets = [
        p for p in targets
        if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.stat().st_size > 0
    ]

    if not targets:
        print("処理対象ファイルが見つかりません。")
        print(f"対応拡張子: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        sys.exit(1)

    # 既存スキップ判定
    if not args.force:
        filtered = []
        for audio in targets:
            out = resolve_output_path(audio, args.out)
            if out.exists():
                print(f"スキップ（既存）: {out.name}  →  --force で上書き可")
            else:
                filtered.append(audio)
        targets = filtered

    print(f"処理対象: {len(targets)} ファイル")

    if args.dry_run:
        for p in targets:
            print(f"  {p} ({p.stat().st_size // 1024}KB)")
        return

    if not targets:
        return

    device = resolve_device(args.device)
    compute_type = resolve_compute_type(args.compute_type, device)
    model = load_model(args.model, device, compute_type)

    failures = 0
    for i, audio in enumerate(targets, 1):
        print(f"\n[{i}/{len(targets)}] {audio.name}")
        try:
            lines = transcribe_file(audio, model, args.language)
            out_path = resolve_output_path(audio, args.out)
            write_markdown(lines, audio, out_path)
        except Exception as e:
            print(f"  エラー: {e}")
            failures += 1

    print(f"\n完了: {len(targets) - failures}/{len(targets)} ファイル処理")

    # Windows CUDA では正常終了でも os._exit が必要な場合がある
    if os.name == "nt" and device == "cuda":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if failures == 0 else 1)

    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
