# VieNeu-TTS — Bản cá nhân gọn nhẹ

> Tool đọc tiếng Việt offline chạy trên máy cá nhân (CPU, 1 người dùng).
> Fork rút gọn từ [pnnbao97/VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS) (VieNeu-TTS v3 Turbo, Apache-2.0).
> Bản này **không phải demo, không phải API server production** — chỉ là công cụ local:
> mở lên, gửi text, nghe/stream file WAV 48 kHz, xong tắt.

## 1. Bản này là gì / không phải gì

**Giữ lại (đủ dùng cá nhân):**

- TTS giọng preset tiếng Việt, streaming WAV 48 kHz mono qua HTTP local (`GET/POST /stream`)
- Chạy CPU bằng ONNX Runtime, ưu tiên model trong `model/` nên **offline hoàn toàn**
- 23 giọng preset sẵn (mặc định: `Minh Quân`), hỗ trợ cue cảm xúc `[cười]`, `[thở dài]`, `[hắng giọng]`
- Server bind `127.0.0.1:8001`, cơ chế **1 người dùng**: request `/stream` mới tự ngắt stream cũ

**Đã cắt so với repo gốc (để nhẹ):**

- Voice cloning (`ref_audio`, `add_voice`, denoise), Gradio Web UI, Docker
- Fine-tune LoRA, backend GPU (torch/transformers/lmdeploy), v3 Nano, v2 legacy
- Tests, docs benchmark, examples OpenAI-API, watermark, PDF

Nếu bạn cần các tính năng trên, dùng repo gốc của tác giả.

## 2. Yêu cầu

- Python 3.12, [uv](https://docs.astral.sh/uv/) (Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`)
- CPU bất kỳ (không cần GPU), RAM trống ~2 GB khi chạy
- Model local `model/onnx_int8/` + `model/codec/` (~200 MB, đã git-ignored — copy vào 1 lần, xem mục 4)

## 3. Cài và chạy

```bash
git clone https://github.com/interface-daodung/VieNeu-TTS.git
cd VieNeu-TTS

cp .env.example .env   # Windows: copy .env.example .env (không bắt buộc)
uv sync

make run               # hoặc: uv run python -m apps.web_stream
# mở http://localhost:8001/docs để thử trực tiếp
```

Lần đầu tải model mất ~10–20 s (đọc ONNX từ `model/`), các lần sau nghe ngay.

## 4. Model local

```
model/
  onnx_int8/   # backbone int8: vieneu_prefill.onnx, vieneu_decode_step.onnx, ...
  codec/       # MOSS codec decode_step + meta
```

- App ưu tiên `model/` nếu thư mục tồn tại, không cần mạng hay HuggingFace cache.
- Nếu thiếu `model/`, engine tự tải từ `pnnbao-ump/VieNeu-TTS-v3-Turbo` và `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX` về cache (cần mạng lần đầu).
- Thư mục `model/` đã nằm trong `.gitignore` nên không bị push lên GitHub.

## 5. Dùng hàng ngày

Server chỉ nghe trên máy bạn: `http://127.0.0.1:8001`. **Đừng public port này.**

### 5.1. Thử nhanh trên trình duyệt

Mở `http://localhost:8001/docs` → `GET /stream` → Try it out → nhập `text`, để trống `voice_id` nếu dùng giọng mặc định.

### 5.2. Tải file WAV bằng curl

```bash
# Giọng mặc định
curl "http://localhost:8001/stream?text=Xin%20ch%C3%A0o%20c%E1%BA%A3%20nh%C3%A0" --output out.wav

# Chọn giọng + text dài thì dùng POST
curl -X POST http://localhost:8001/stream \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Hôm nay trời đẹp. [cười] Mình đọc thử một đoạn dài.\", \"voice_id\": \"Mai Anh\"}" \
  --output out.wav
```

### 5.3. Dùng trong Python

```python
from vieneu import Vieneu

tts = Vieneu(backend="onnx")  # đọc model/ local, CPU
names = [vid for _, vid in tts.list_preset_voices()]
print(names)

with open("out.pcm", "wb") as f:
    for chunk in tts.infer_stream("Xin chào, đây là bản cá nhân chạy offline.", voice="Mai Anh"):
        f.write((chunk * 32767).clip(-32768, 32767).astype("<i2").tobytes())
# file trên là PCM 48 kHz mono — bọc header WAV hoặc dùng soundfile để lưu .wav
```

### 5.4. Giọng đọc

23 preset trong `src/vieneu/assets/voices_v3_turbo.json` (mặc định `Minh Quân`):

> Minh Đức, Phạm Tuyên, Thái Sơn, Xuân Vĩnh, Thanh Bình, Trúc Ly, Ngọc Linh,
> Đoan Trang, Mai Anh, Thục Đoan, Minh Triết, Thùy Dung, Quang Sơn, Ngọc Trân,
> Mỹ Duyên, Quỳnh Anh, Đức Trí, Kim Thanh, Ngọc Huyền, Adam, Mạnh Dũng, Minh Quân, Anh Khôi

`voice_id` chính là tên trên (phân biệt hoa/thường, gõ sai sẽ báo lỗi kèm danh sách đúng).
Để trống `voice_id` = dùng giọng mặc định.

### 5.5. Lưu ý single-user

- Mỗi request `/stream` mới chiếm quyền phát, stream cũ tự dừng ở chunk kế tiếp.
- Muốn nghe song song nhiều máy: đây không phải tool dành cho việc đó.

## 6. Cấu hình (`.env`)

| Biến | Mặc định | Nghĩa |
|---|---|---|
| `VIENEU_PRECISION` | `fp32` | `int8` nhẹ/nhanh hơn (~3x), `fp32` chất lượng tối đa |
| `VIENEU_ONNX_DIR` | `model/onnx_int8` | Đường dẫn backbone local; bỏ trống = dùng HF cache |
| `VIENEU_CODEC_DIR` | `model/codec` | Đường dẫn codec local; bỏ trống = dùng HF cache |

Xem [.env.example](.env.example). `PYTHONUTF8=1` đã bật sẵn trong `Makefile` để không lỗi tiếng Việt trên Windows.

## 7. Cấu trúc

```
apps/web_stream.py      # server local duy nhất (GET/POST /stream, 127.0.0.1:8001)
src/vieneu/             # backend v3 Turbo ONNX/CPU: preset voice + infer_stream
src/vieneu_utils/       # làm sạch text, chia chunk, phonemize (sea-g2p)
model/                  # model local, git-ignored (~200 MB)
.env.example            # mẫu cấu hình cá nhân
NOTICE                  # ghi nhận bản quyền upstream + fork
```

## 8. Dọn dẹp

```bash
make cls   # xóa HF cache thừa + __pycache__ + outputs/
```

## 9. Ghi nhận & giấy phép

- Toàn bộ repo (code + model weights upstream) theo **Apache-2.0** — xem [LICENSE](LICENSE); model v3 Turbo cũng được công bố `apache-2.0` trên [Hugging Face](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo).
- Model, giọng đọc và kiến trúc v3 Turbo thuộc về tác giả gốc **Phạm Nguyễn Ngọc Bảo** — xem repo upstream và file [NOTICE](NOTICE) (ghi chú các file đã sửa theo mục 4(b) Apache-2.0).
- Phần cắt gọn/server local trong fork này © 2026 Đào Mạnh Dũng, phát hành lại dưới **Apache-2.0** (xem [LICENSE](LICENSE)).
- `model/` không nằm trong repo (git-ignored) — bạn tự tải từ nguồn gốc, tuân theo license của nguồn đó.
- Dùng cá nhân trên máy của bạn. Không dùng giọng preset để giả mạo người thật.
