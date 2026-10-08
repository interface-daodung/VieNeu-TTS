"""VieNeu-TTS v3 Turbo — minimal CPU streaming API (FastAPI).

Chỉ giữ đường TTS giọng preset:
    uv run python -m apps.web_stream        # http://127.0.0.1:8001
    GET /stream?text=...&voice_id=...  -> audio/wav 48kHz
"""
import os
import time
import io
import itertools
import threading
import wave
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from vieneu import Vieneu

SAMPLE_RATE = 48_000
app = FastAPI()
vieneu = None

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "model"

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


def _local_dir(env_name: str, *sub: str) -> Optional[str]:
    """Env override, else bundled ``model/<sub>`` if present, else None (HF cache)."""
    v = os.environ.get(env_name)
    if v:
        return v
    p = MODEL_DIR.joinpath(*sub)
    return str(p) if p.is_dir() else None


def load_model():
    global vieneu
    print(f"Loading VieNeu-TTS v3 Turbo ({os.environ.get('VIENEU_PRECISION', 'fp32')}, CPU)...")
    vieneu = Vieneu(
        backend="onnx",
        precision=os.environ.get("VIENEU_PRECISION", "fp32"),
        onnx_dir=_local_dir("VIENEU_ONNX_DIR", "onnx_int8"),
        codec_dir=_local_dir("VIENEU_CODEC_DIR", "codec"),
    )
    print(f"Ready. intra_op threads: {getattr(vieneu.engine, 'ort_intra_op_threads', '?')}")


load_model()


def _pcm16(audio_f32: np.ndarray) -> bytes:
    return (np.asarray(audio_f32) * 32767).clip(-32768, 32767).astype(np.int16).tobytes()


@app.get("/stream")
async def stream(text: str, voice_id: Optional[str] = None):
    """Stream 48 kHz WAV. Request mới hơn sẽ ngắt stream này (single-user)."""
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


@app.post("/stream")
async def stream_post(req: StreamReq):
    return await stream(req.text, req.voice_id)


def main():
    print("Mo http://localhost:8001 de test VieNeu v3 Turbo streaming (CPU)")
    uvicorn.run(app, host="127.0.0.1", port=8001)


if __name__ == "__main__":
    main()
