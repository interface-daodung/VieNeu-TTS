"""VieNeu-TTS v3 Turbo — minimal CPU streaming API (FastAPI).

Đã cắt gọn từ https://github.com/pnnbao97/VieNeu-TTS (Apache-2.0) — xem NOTICE.
Chỉ giữ đường TTS giọng preset:
    uv run python -m apps.web_stream        # http://127.0.0.1:8001
    GET /stream?text=...&voice_id=...  -> audio/wav 48kHz

Deploy on server:
    bash scripts/deploy.sh
    # or manually:
    uv run python scripts/download_model.py
    VIENEU_HOST=0.0.0.0 uv run python -m apps.web_stream
"""
import os
import sys
import time
import io
import hmac
import hashlib
import secrets
import itertools
import subprocess
import threading
import wave
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from vieneu import Vieneu

SAMPLE_RATE = 48_000
app = FastAPI()
vieneu = None

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "model"
ONNX_DIR = MODEL_DIR / "onnx_int8"
CODEC_DIR = MODEL_DIR / "codec"

# Required files for INT8 backbone and codec
_REQUIRED_ONNX = [
    "vieneu_prefill.onnx", "vieneu_decode_step.onnx",
    "vieneu_acoustic_cached.onnx", "vieneu_backbone_shared.data",
    "vieneu_v3_heads.npz", "config.json", "tokenizer.json",
]
_REQUIRED_CODEC = [
    "moss_audio_tokenizer_decode_step.onnx", "moss_audio_tokenizer_decode_shared.data",
    "codec_browser_onnx_meta.json",
]

# ── Token auth ───────────────────────────────────────────────────────────────
# Set VIENEU_TOKEN in .env to enable. If unset, server is open (local dev).
_TOKEN: Optional[str] = os.environ.get("VIENEU_TOKEN", "").strip() or None


def _verify_token(token: Optional[str]) -> None:
    """Raise 401 if token auth is enabled and token is missing/wrong."""
    if _TOKEN is None:
        return
    if token is None:
        raise HTTPException(status_code=401, detail="Missing token")
    # Constant-time compare to avoid timing attacks
    if not hmac.compare_digest(token, _TOKEN):
        raise HTTPException(status_code=401, detail="Invalid token")


# Single-user semantics: chỉ phục vụ kết nối mới nhất. Request /stream mới sẽ
# chiếm token, generator của request cũ tự dừng ở chunk kế tiếp.
_stream_lock = threading.Lock()
_stream_counter = itertools.count(1)
_current_stream_id = 0


def _claim_stream() -> int:
    global _current_stream_id
    with _stream_lock:
        my_id = next(_stream_counter)
        _current_stream_id = my_id
        return my_id


def _is_current(my_id: int) -> bool:
    with _stream_lock:
        return my_id == _current_stream_id


def _files_present(directory: Path, files: list) -> bool:
    """Check if all required files exist and are non-empty."""
    if not directory.is_dir():
        return False
    return all((directory / f).is_file() and (directory / f).stat().st_size > 0 for f in files)


def _ensure_model() -> bool:
    """Ensure INT8 model is present in model/. Download if missing.

    Returns True if model is ready, False if download failed.
    """
    if _files_present(ONNX_DIR, _REQUIRED_ONNX) and _files_present(CODEC_DIR, _REQUIRED_CODEC):
        print("[model] Found local INT8 model in model/")
        return True

    print("[model] Local INT8 model not found or incomplete. Downloading from Hugging Face...")
    script = ROOT_DIR / "scripts" / "download_model.py"
    if not script.is_file():
        print(f"[model] ERROR: {script} not found. Cannot download model.")
        return False

    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(ROOT_DIR),
            capture_output=False,
        )
        if result.returncode != 0:
            print("[model] ERROR: download failed. Check network connection.")
            return False
        print("[model] Download complete.")
        return True
    except Exception as e:
        print(f"[model] ERROR during download: {e}")
        return False


def _local_dir(env_name: str, *sub: str) -> Optional[str]:
    """Env override, else bundled ``model/<sub>`` if present, else None."""
    v = os.environ.get(env_name)
    if v:
        return v
    p = MODEL_DIR.joinpath(*sub)
    return str(p) if p.is_dir() else None


def load_model():
    global vieneu

    # Pre-flight: ensure model is available before loading
    if not _ensure_model():
        print("[model] FATAL: Model not available. Cannot start server.")
        sys.exit(1)

    # Hardcode INT8 — khong cho env override
    print(f"Loading VieNeu-TTS v3 Turbo (INT8, CPU)...")
    vieneu = Vieneu(
        backend="onnx",
        precision="int8",
        onnx_dir=_local_dir("VIENEU_ONNX_DIR", "onnx_int8"),
        codec_dir=_local_dir("VIENEU_CODEC_DIR", "codec"),
    )
    print(f"Ready. intra_op threads: {getattr(vieneu.engine, 'ort_intra_op_threads', '?')}")


load_model()


@app.get("/health")
async def health(token: Optional[str] = Query(None)):
    """Health check — returns model status."""
    _verify_token(token)
    return {
        "status": "ok",
        "model": "v3-turbo-int8",
        "precision": "int8",
        "sample_rate": SAMPLE_RATE,
        "voices": len(vieneu._preset_voices) if vieneu else 0,
    }


def _pcm16(audio_f32: np.ndarray) -> bytes:
    return (np.asarray(audio_f32) * 32767).clip(-32768, 32767).astype(np.int16).tobytes()


@app.get("/stream")
async def stream(
    text: str,
    voice_id: Optional[str] = None,
    token: Optional[str] = Query(None),
):
    """Stream 48 kHz WAV. Request mới hơn sẽ ngắt stream này (single-user)."""
    _verify_token(token)
    my_id = _claim_stream()

    def gen():
        if not _is_current(my_id):
            return
        h = io.BytesIO()
        with wave.open(h, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
            w.setnframes(1_000_000_000)
        yield h.getvalue()

        t0 = time.perf_counter()
        first_at = None
        n_chunks = 0
        emitted = 0
        for chunk in vieneu.infer_stream(text, voice=voice_id or None):
            if not _is_current(my_id):
                print("superseded by newer request, stopping stream")
                break
            if chunk is None or len(chunk) == 0:
                continue
            if first_at is None:
                first_at = time.perf_counter() - t0
                print(f"TTFA: {first_at*1000:.0f} ms")
            n_chunks += 1
            emitted += len(chunk)
            yield _pcm16(chunk)
        if first_at is not None:
            gen_time = time.perf_counter() - t0
            audio_s = emitted / SAMPLE_RATE
            rtf = gen_time / audio_s if audio_s else 0
            print(f"{n_chunks} chunks | audio {audio_s:.2f}s | gen {gen_time:.2f}s "
                  f"| RTF {rtf:.3f}" if rtf else "")

    return StreamingResponse(gen(), media_type="audio/wav")


class StreamReq(BaseModel):
    text: str
    voice_id: Optional[str] = None
    token: Optional[str] = None


@app.post("/stream")
async def stream_post(req: StreamReq):
    _verify_token(req.token)
    return await stream(req.text, req.voice_id, req.token)


def main():
    host = os.environ.get("VIENEU_HOST", "127.0.0.1")
    port = int(os.environ.get("VIENEU_PORT", "8001"))
    print(f"VieNeu v3 Turbo streaming: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
