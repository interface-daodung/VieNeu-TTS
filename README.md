# VieNeu-TTS Minimal

> **Fork rút gọn** từ [pnnbao97/VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS) (bản open-source v3 Turbo).
> Bản này chỉ giữ **1 việc duy nhất**: streaming TTS giọng Việt sẵn qua HTTP — bỏ voice cloning,
> Gradio UI, fine-tune, Docker, tests và mọi backend GPU. Nhẹ hơn, đơn giản hơn, chạy offline hoàn toàn.

## Chạy

```bash
uv sync
make run        # hoặc: uv run python -m apps.web_stream
# mở http://localhost:8001/docs để thử
```

Model (~200 MB) nằm sẵn trong `model/` (`onnx_int8/` + `codec/`), app ưu tiên dùng local,
không cần mạng hay HuggingFace cache.

## API

| Endpoint | Mô tả |
|---|---|
| `GET /stream?text=...&voice_id=...` | Stream WAV 48 kHz mono |
| `POST /stream` `{text, voice_id}` | Như trên, dùng khi text dài |

```bash
curl "http://localhost:8001/stream?text=Xin%20ch%C3%A0o" --output out.wav
```

`voice_id` là tên giọng preset (bỏ trống = giọng mặc định).

> Server dùng local 1 người: gọi `/stream` mới sẽ tự ngắt stream cũ,
> chỉ phục vụ kết nối mới nhất.

## Cấu trúc

```
apps/web_stream.py      # FastAPI server (/stream)
src/vieneu/             # backend v3 Turbo ONNX/CPU (preset voice + infer_stream)
src/vieneu_utils/       # làm sạch text, chia chunk, phonemize (sea-g2p)
model/                  # model local (git-ignored, ~200 MB)
```

## Biến môi trường (`.env`)

| Biến | Mặc định | Mô tả |
|---|---|---|
| `VIENEU_PRECISION` | `fp32` | `int8` (nhẹ ~3x) hoặc `fp32` |
| `VIENEU_ONNX_DIR` | `model/onnx_int8` | Đường dẫn backbone local |
| `VIENEU_CODEC_DIR` | `model/codec` | Đường dẫn codec local |

## Dọn dẹp

```bash
make cls   # xóa HF cache thừa + __pycache__ + outputs/
```

## Ghi nhận

Model và giọng đọc thuộc về tác giả gốc **Phạm Nguyễn Ngọc Bảo** (xem repo upstream).
Code trong fork này giữ nguyên giấy phép **Apache-2.0** (`LICENSE`).
