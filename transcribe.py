r"""
音声・動画ファイルの文字起こしツール（Windows/Mac対応）

対応フォーマット: m4a, mp4, mp3, wav, flac, ogg, webm
エンジン: faster-whisper（ローカル）+ pyannote.audio（話者分離）
デバイス: Windows CUDA GPU / Mac CPU（自動判定）

使い方:
    # 1人（タイムスタンプのみ）
    uv run python transcribe.py recording.m4a

    # 2人以上（話者分離あり）
    uv run python transcribe.py recording.m4a --speakers 2

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
# Windows CUDA DLL を最優先でPATHに追加（import前に実行）
# ──────────────────────────────────────────────

def _early_cuda_setup():
    if os.name != "nt":
        return
    home = Path.home()
    venv_site = Path(__file__).parent / ".venv" / "Lib" / "site-packages"
    dll_dirs = [
        # venv 内 torch/lib（cuDNN DLL が含まれる）
        venv_site / "torch" / "lib",
        # venv 内 ctranslate2
        venv_site / "ctranslate2",
        # Python313 システム torch
        home / "AppData" / "Local" / "Programs" / "Python" / "Python313" / "Lib" / "site-packages" / "torch" / "lib",
        # arano-bot venv ctranslate2
        home / "workspace-ai" / "nomuraya-job-fde" / "arano-bot" / ".venv-whisperx"
        / "Lib" / "site-packages" / "ctranslate2",
        home / "AppData" / "Local" / "Programs" / "Ollama" / "lib" / "ollama" / "cuda_v12",
    ]
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        dll_dirs.insert(0, Path(cuda_path) / "bin")

    for d in dll_dirs:
        if not d.exists():
            continue
        try:
            os.add_dll_directory(str(d))
        except (AttributeError, OSError):
            pass
        os.environ["PATH"] = f"{d}{os.pathsep}{os.environ.get('PATH', '')}"

_early_cuda_setup()


# ──────────────────────────────────────────────
# HF トークン解決
# ──────────────────────────────────────────────

def resolve_hf_token() -> str | None:
    """環境変数 → HF キャッシュの順でトークンを探す。"""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    for cache in [
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / "AppData" / "Roaming" / "huggingface" / "token",
    ]:
        if cache.exists():
            t = cache.read_text().strip()
            if t:
                return t
    return None


# ──────────────────────────────────────────────
# Windows CUDA ランタイム補助
# ──────────────────────────────────────────────

def configure_windows_cuda_runtime() -> list[Path]:
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
    # arano-bot venv の ctranslate2 に同梱された cuDNN DLL
    candidates.append(
        home / "workspace-ai" / "nomuraya-job-fde" / "arano-bot" / ".venv-whisperx"
        / "Lib" / "site-packages" / "ctranslate2"
    )
    # toolkit-audio 自身の venv 内 ctranslate2
    candidates.append(
        Path(__file__).parent / ".venv" / "Lib" / "site-packages" / "ctranslate2"
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
        return bool(ctranslate2.get_supported_compute_types("cuda"))
    except Exception:
        return False


def resolve_device(device_arg: str) -> str:
    if device_arg == "cpu":
        return "cpu"
    if device_arg == "cuda":
        if not detect_cuda():
            raise RuntimeError("CUDA が利用できません。--device cpu を指定してください。")
        return "cuda"
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
    configure_windows_cuda_runtime()
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"ffmpeg: {shutil.which('ffmpeg') or '未検出 ⚠'}")
    try:
        import faster_whisper
        print(f"faster-whisper: {getattr(faster_whisper, '__version__', 'installed')}")
    except ImportError:
        print("faster-whisper: 未インストール ⚠  →  uv pip install faster-whisper")
    try:
        import pyannote.audio
        print(f"pyannote.audio: {getattr(pyannote.audio, '__version__', 'installed')}")
    except ImportError:
        print("pyannote.audio: 未インストール（話者分離不可）→  uv pip install pyannote.audio")
    cuda_ok = detect_cuda()
    print(f"CUDA: {'利用可能' if cuda_ok else '利用不可（CPU実行）'}")
    hf_token = resolve_hf_token()
    print(f"HF token: {'あり' if hf_token else 'なし（話者分離不可）'}")
    device = resolve_device(device_arg)
    ct = resolve_compute_type(compute_type, device)
    print(f"実行設定: device={device}, model={model_name}, compute_type={ct}")
    print("環境チェック完了")


# ──────────────────────────────────────────────
# モデルロード・文字起こし
# ──────────────────────────────────────────────

def load_whisper(model_name: str, device: str, compute_type: str):
    configure_windows_cuda_runtime()
    from faster_whisper import WhisperModel
    print(f"Whisper モデルロード中... (model={model_name}, device={device}, compute_type={compute_type})")
    return WhisperModel(model_name, device=device, compute_type=compute_type)


def transcribe_only(audio: Path, model, language: str) -> list[dict]:
    """話者分離なし: [{start, end, text}]"""
    segments, _ = model.transcribe(
        str(audio),
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]


def to_wav(audio: Path) -> Path:
    """m4a等をwavに変換してffmpegのDLL問題を回避する。既にwavならそのまま返す。"""
    if audio.suffix.lower() == ".wav":
        return audio
    wav = audio.with_suffix(".wav")
    if not wav.exists():
        import subprocess
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio), "-ar", "16000", "-ac", "1", str(wav)],
            check=True, capture_output=True,
        )
    return wav


def transcribe_with_diarization(audio: Path, model, language: str, num_speakers: int, hf_token: str) -> list[dict]:
    """話者分離あり: [{start, end, speaker, text}]"""
    import torch
    from pyannote.audio import Pipeline

    wav = to_wav(audio)

    # Windows: torch/lib の DLL を明示的にロードしてから pyannote を使う
    if os.name == "nt":
        import ctypes
        torch_lib = Path(torch.__file__).parent / "lib"
        for dll in torch_lib.glob("cudnn*.dll"):
            try:
                ctypes.CDLL(str(dll))
            except OSError:
                pass

    print(f"  pyannote パイプラインロード中（CPU）...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )

    # torchcodec 非対応環境向け: torchaudio で事前ロードして辞書形式で渡す
    print(f"  音声ロード中...")
    import torchaudio
    waveform, sample_rate = torchaudio.load(str(wav))
    audio_input = {"waveform": waveform, "sample_rate": sample_rate}

    print(f"  話者分離中 (speakers={num_speakers})...")
    result = pipeline(audio_input, num_speakers=num_speakers)
    diarization = result.speaker_diarization

    # 話者ターンを収集
    turns = [(turn.start, turn.end, speaker) for turn, _, speaker in diarization.itertracks(yield_label=True)]

    # Whisper セグメントを取得
    print(f"  文字起こし中...")
    segments, _ = model.transcribe(
        str(audio),
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    segments = list(segments)

    # セグメントを話者ターンに割り当て（中点で最も重なるターンを選択）
    def find_speaker(start: float, end: float) -> str:
        mid = (start + end) / 2
        best, best_len = "UNKNOWN", 0.0
        for t_start, t_end, spk in turns:
            overlap = max(0.0, min(end, t_end) - max(start, t_start))
            if overlap > best_len or (best_len == 0 and t_start <= mid <= t_end):
                best, best_len = spk, overlap
        return best

    result = []
    for seg in segments:
        speaker = find_speaker(seg.start, seg.end)
        result.append({"start": seg.start, "end": seg.end, "speaker": speaker, "text": seg.text.strip()})
    return result


# ──────────────────────────────────────────────
# 出力
# ──────────────────────────────────────────────

def resolve_output_path(audio: Path, out_dir: Path | None) -> Path:
    date_prefix = datetime.now().strftime("%Y%m%d")
    stem = audio.stem
    if not stem[:8].isdigit():
        stem = f"{date_prefix}-{stem}"
    return (out_dir or audio.parent) / f"{stem}.md"


def write_markdown(segments: list[dict], audio: Path, out_path: Path):
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    has_speaker = "speaker" in (segments[0] if segments else {})

    lines = [
        f"# {out_path.stem}",
        "",
        f"録音ファイル: `{audio.name}`  ",
        f"文字起こし日時: {date_str}",
        "",
        "---",
        "",
    ]

    for seg in segments:
        ts = f"[{seg['start']:.3f}]"
        if has_speaker:
            lines.append(f"{ts} **{seg['speaker']}**: {seg['text']}")
        else:
            lines.append(f"{ts} {seg['text']}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  保存: {out_path}")


# ──────────────────────────────────────────────
# エントリーポイント
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="音声・動画ファイルを文字起こしする（Windows/Mac対応）")
    parser.add_argument("files", nargs="*", help="処理する音声/動画ファイル")
    parser.add_argument("--out", type=Path, default=None, help="出力ディレクトリ")
    parser.add_argument("--model", default="large-v3", help="Whisper モデル名（デフォルト: large-v3）")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--compute-type", default="float32", choices=["float32", "float16", "int8"])
    parser.add_argument("--language", default="ja", help="言語コード（デフォルト: ja）")
    parser.add_argument("--speakers", type=int, default=None, help="話者数（省略時は話者分離なし）")
    parser.add_argument("--check", action="store_true", help="環境確認のみ")
    parser.add_argument("--dry-run", action="store_true", help="対象確認のみ")
    parser.add_argument("--force", action="store_true", help="既存ファイルも上書き")
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
            targets.extend(Path(".").glob(pattern))

    targets = [p for p in targets if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.stat().st_size > 0]

    if not targets:
        print("処理対象ファイルが見つかりません。")
        print(f"対応拡張子: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        sys.exit(1)

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
    if args.speakers:
        print(f"話者分離: あり（{args.speakers}人）")
    else:
        print("話者分離: なし（1人モード）")

    if args.dry_run:
        for p in targets:
            print(f"  {p} ({p.stat().st_size // 1024}KB)")
        return

    if not targets:
        return

    # HFトークン（話者分離時のみ必須）
    hf_token = None
    if args.speakers:
        hf_token = resolve_hf_token()
        if not hf_token:
            print("エラー: 話者分離には HF トークンが必要です。")
            print("  huggingface-cli login  または  HF_TOKEN 環境変数を設定してください。")
            sys.exit(1)

    device = resolve_device(args.device)
    compute_type = resolve_compute_type(args.compute_type, device)
    model = load_whisper(args.model, device, compute_type)

    failures = 0
    for i, audio in enumerate(targets, 1):
        print(f"\n[{i}/{len(targets)}] {audio.name}")
        t0 = time.time()
        try:
            if args.speakers:
                segments = transcribe_with_diarization(audio, model, args.language, args.speakers, hf_token)
            else:
                segments = transcribe_only(audio, model, args.language)

            elapsed = time.time() - t0
            print(f"  完了: {elapsed:.0f}秒, {len(segments)} セグメント")

            out_path = resolve_output_path(audio, args.out)
            write_markdown(segments, audio, out_path)
        except Exception as e:
            print(f"  エラー: {e}")
            failures += 1

    print(f"\n完了: {len(targets) - failures}/{len(targets)} ファイル処理")

    if os.name == "nt" and device == "cuda":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if failures == 0 else 1)

    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
