"""Download VieNeu-TTS v3 Turbo INT8 model from Hugging Face into ``model/``.

Run once on the server (or any machine) before starting the web server.
The engine will then read directly from ``model/onnx_int8`` and
``model/codec`` without any further HF calls.

Usage:
    python scripts/download_model.py              # download both repos
    python scripts/download_model.py --backbone-only
    python scripts/download_model.py --codec-only
    python scripts/download_model.py --check      # verify existing files
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

# ── HF repo / subfolder constants (must match onnx_runtime_lite.py) ──────────
_V3_REPO = "pnnbao-ump/VieNeu-TTS-v3-Turbo"
_CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX"
_ONNX_SUBFOLDER = "onnx_int8"

_GRAPH_FILES = [
    "vieneu_prefill.onnx", "vieneu_decode_step.onnx",
    "vieneu_acoustic_cached.onnx", "vieneu_backbone_shared.data",
    "vieneu_v3_heads.npz", "config.json", "tokenizer.json",
]
_CODEC_FILES = [
    "moss_audio_tokenizer_decode_step.onnx", "moss_audio_tokenizer_decode_shared.data",
    "codec_browser_onnx_meta.json",
]

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "model"
ONNX_DIR = MODEL_DIR / "onnx_int8"
CODEC_DIR = MODEL_DIR / "codec"


def _fetch_file(repo: str, filename: str, subfolder: str, dest_dir: Path) -> Tuple[bool, str]:
    """Download a single file from HF into dest_dir. Returns (ok, message)."""
    from huggingface_hub import hf_hub_download

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename

    # Skip if already present and non-empty
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return True, f"skip (exists): {dest_path.name}"

    try:
        cached_path = hf_hub_download(
            repo, filename, repo_type="model", subfolder=subfolder or None
        )
        import shutil
        shutil.copy2(cached_path, dest_path)
        return True, f"downloaded: {dest_path.name}"
    except Exception as e:
        return False, f"FAILED: {filename} -> {e}"


def download_backbone() -> bool:
    """Download ONNX backbone into model/onnx_int8/."""
    print(f"\n[backbone] {_V3_REPO}/{_ONNX_SUBFOLDER}")
    ok = True
    for fn in _GRAPH_FILES:
        success, msg = _fetch_file(_V3_REPO, fn, _ONNX_SUBFOLDER, ONNX_DIR)
        print(f"  {msg}")
        if not success:
            ok = False
    return ok


def download_codec() -> bool:
    """Download MOSS codec into model/codec/."""
    print(f"\n[codec] {_CODEC_REPO}")
    ok = True
    for fn in _CODEC_FILES:
        success, msg = _fetch_file(_CODEC_REPO, fn, None, CODEC_DIR)
        print(f"  {msg}")
        if not success:
            ok = False
    return ok


def check_model() -> bool:
    """Verify all required files exist and are non-empty."""
    all_ok = True
    print("\n[check]")
    for d, files, label in [(ONNX_DIR, _GRAPH_FILES, "onnx_int8"), (CODEC_DIR, _CODEC_FILES, "codec")]:
        print(f"  {label}/")
        for fn in files:
            p = d / fn
            if p.exists() and p.stat().st_size > 0:
                size_mb = p.stat().st_size / 1_048_576
                print(f"    OK  {fn} ({size_mb:.1f} MB)")
            else:
                print(f"    MISSING  {fn}")
                all_ok = False
    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Download VieNeu INT8 model to model/")
    parser.add_argument("--backbone-only", action="store_true")
    parser.add_argument("--codec-only", action="store_true")
    parser.add_argument("--check", action="store_true", help="Only verify, don't download")
    args = parser.parse_args()

    if args.check:
        sys.exit(0 if check_model() else 1)

    do_backbone = not args.codec_only
    do_codec = not args.backbone_only

    ok = True
    if do_backbone:
        ok = download_backbone() and ok
    if do_codec:
        ok = download_codec() and ok

    print("\n" + "=" * 50)
    if ok:
        check_model()
        print("\nModel ready in model/. Engine will use it directly.")
        return 0
    else:
        print("\nSome files failed. Re-run the script to retry.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
