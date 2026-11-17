"""VieNeu-TTS v3 Turbo — minimal CPU streaming API (FastAPI).

Đã cắt gọn từ https://github.com/pnnbao97/VieNeu-TTS (Apache-2.0) — xem NOTICE.
Hỗ trợ:
- GET /stream?text=...&voice_id=...&sample_rate=16000&camera_mode=true
- POST /stream  (JSON: text, voice_id, sample_rate, camera_mode)
- GET /voices   (Liệt kê danh sách giọng đọc có sẵn)
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
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from vieneu import Vieneu

SAMPLE_RATE_DEFAULT = 48_000
SAMPLE_RATE_CAMERA = 16_000

app = FastAPI(
    title="VieNeu-TTS v3 Turbo API",
    description="Hệ thống API sinh giọng nói tiếng Việt thời gian thực (Real-time Audio Streaming) tối ưu cho CPU. Hỗ trợ cờ chuẩn Camera 16kHz.",
    version="3.7.0",
)
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


def _resample_48k_to_16k(audio_f32: np.ndarray) -> np.ndarray:
    """
    Hạ mẫu chuẩn 3:1 từ 48,000 Hz về 16,000 Hz.
    Ưu tiên scipy.signal.resample_poly nếu có; fallback sang bộ lọc FIR chống răng cưa (pure numpy).
    """
    try:
        import scipy.signal
        return scipy.signal.resample_poly(audio_f32, 1, 3).astype(np.float32)
    except Exception:
        # 21-tap FIR anti-aliasing filter (cutoff ~0.33 pi)
        taps = np.array([
            -0.001, -0.004, 0.007, 0.033, 0.038, -0.019, -0.109,
            -0.103, 0.125, 0.449, 0.600, 0.449, 0.125, -0.103,
            -0.109, -0.019, 0.038, 0.033, 0.007, -0.004, -0.001
        ], dtype=np.float32)
        taps /= np.sum(taps)
        return np.convolve(audio_f32, taps, mode="same")[::3].astype(np.float32)


def _pcm16(audio_f32: np.ndarray) -> bytes:
    return (np.asarray(audio_f32) * 32767).clip(-32768, 32767).astype(np.int16).tobytes()


@app.get("/stream")
async def stream(
    text: str,
    voice_id: Optional[str] = None,
    sample_rate: Optional[int] = Query(None, description="Tần số mẫu đầu ra: 48000 (mặc định) hoặc 16000 (chuẩn Camera)."),
    camera_mode: bool = Query(False, description="Cờ chuyên dụng cho Camera Imou/Dahua. Khi bật True, tự động xuất chuẩn 16,000 Hz 16-bit Mono."),
):
    """
    Stream âm thanh WAV thời gian thực theo chunk.
    Hỗ trợ xuất trực tiếp 16,000 Hz (16kHz) cho Loa Camera khi bật `camera_mode=True` hoặc `sample_rate=16000`.
    """
    my_id = _claim_stream()
    target_sr = SAMPLE_RATE_CAMERA if (camera_mode or sample_rate == 16_000) else SAMPLE_RATE_DEFAULT

    def gen():
        if not _is_current(my_id):
            return
        h = io.BytesIO()
        with wave.open(h, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(target_sr)
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

            # Hạ mẫu trực tiếp từ 48kHz về 16kHz cho Camera
            if target_sr == SAMPLE_RATE_CAMERA:
                chunk = _resample_48k_to_16k(chunk)

            if first_at is None:
                first_at = time.perf_counter() - t0
                print(f"TTFA: {first_at*1000:.0f} ms (Target SR: {target_sr} Hz)")

            n_chunks += 1
            emitted += len(chunk)
            yield _pcm16(chunk)

        if first_at is not None:
            gen_time = time.perf_counter() - t0
            audio_s = emitted / target_sr
            rtf = gen_time / audio_s if audio_s else 0
            print(f"{n_chunks} chunks | audio {audio_s:.2f}s | gen {gen_time:.2f}s "
                  f"| SR {target_sr}Hz | RTF {rtf:.3f}" if rtf else "")

    return StreamingResponse(gen(), media_type="audio/wav")


class StreamReq(BaseModel):
    text: str = Field(..., description="Văn bản tiếng Việt cần đọc")
    voice_id: Optional[str] = Field("Minh Quân", description="Tên giọng đọc preset (ví dụ: 'Minh Quân', 'Mai Anh'...)")
    sample_rate: Optional[int] = Field(None, description="Tùy chọn tần số mẫu: 16000 hoặc 48000")
    camera_mode: Optional[bool] = Field(False, description="Cờ bật chế độ chuẩn 16,000 Hz cho Loa Camera (tránh phải convert)")


@app.post("/stream")
async def stream_post(req: StreamReq):
    """
    POST endpoint nhận yêu cầu tổng hợp giọng nói qua JSON body.
    Hỗ trợ cờ camera_mode=True hoặc sample_rate=16000 để phát trực tiếp ra loa Camera.
    """
    return await stream(
        text=req.text,
        voice_id=req.voice_id,
        sample_rate=req.sample_rate,
        camera_mode=req.camera_mode or False,
    )


@app.get("/voices")
def get_voices():
    """Liệt kê toàn bộ các giọng đọc preset có sẵn trong hệ thống."""
    voices = []
    if vieneu:
        for label, vid in vieneu.list_preset_voices():
            voice_info = vieneu.get_preset_voice(vid)
            voices.append({
                "id": vid,
                "label": label,
                "gender": voice_info.get("gender"),
                "description": voice_info.get("description"),
            })
    return {
        "default_voice": getattr(vieneu, "_default_voice", "Minh Quân"),
        "total": len(voices),
        "voices": voices,
    }


def main():
    print("Mở http://localhost:8001 để test VieNeu v3 Turbo streaming (CPU)")
    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()
