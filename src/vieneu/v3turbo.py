"""VieNeu-TTS v3 Turbo — minimal preset-voice backend (ONNX/CPU).

Đã cắt gọn từ https://github.com/pnnbao97/VieNeu-TTS (Apache-2.0) — xem NOTICE.
Giữ duy nhất đường TTS giọng sẵn + stream:
    tts = Vieneu(backend="onnx")
    wav = next(tts.infer_stream("Xin chào", voice="Xuân Vĩnh"))
"""
import logging
from pathlib import Path
from typing import Any, Generator, List, Optional, Tuple, Union

import numpy as np

from .base import BaseVieneuTTS
from ._v3_turbo_engine.rep_history import DEFAULT_REP_WINDOW
from vieneu_utils.phonemize_text import (
    phonemize_text_with_emotions,
    normalize_to_chunks_v3_with_gaps,
)
from vieneu_utils.core_utils import (
    max_expected_frames, gaps_to_silence, pause_pad_samples,
)


def _cap_frames(sampling: dict, cap: int) -> dict:
    out = dict(sampling)
    out["max_new_frames"] = min(out.get("max_new_frames", 300), cap)
    return out

logger = logging.getLogger("Vieneu.V3Turbo")


class V3TurboVieNeuTTS(BaseVieneuTTS):
    """VieNeu-TTS v3 Turbo, ONNX/CPU only, preset voices only."""

    def __init__(
        self,
        backbone_repo: str = "pnnbao-ump/VieNeu-TTS-v3-Turbo",
        backend: str = "onnx",
        onnx_repo: Optional[str] = None,
        onnx_dir: Optional[str] = None,
        codec_dir: Optional[str] = None,
        precision: str = "fp32",
        onnx_subfolder: Optional[str] = None,
        threads: int = 0,
        **kwargs: Any,
    ):
        super().__init__()
        self.sample_rate = 48_000

        if backend != "onnx":
            raise ValueError(f"Minimal server only supports backend='onnx', got {backend!r}.")
        if onnx_subfolder is None:
            onnx_subfolder = {"int8": "onnx_int8", "fp32": "onnx_update"}.get(str(precision).lower(), "onnx_update")

        from ._v3_turbo_engine.onnx_runtime_lite import OnnxV3LiteEngine
        logger.info(f"Loading VieNeu-TTS v3 Turbo (ONNX/CPU) from: {backbone_repo}/{onnx_subfolder} ...")
        self.engine = OnnxV3LiteEngine(
            checkpoint_path=backbone_repo,
            onnx_repo=onnx_repo,
            onnx_dir=onnx_dir,
            codec_dir=codec_dir,
            onnx_subfolder=onnx_subfolder,
            threads=threads,
        )
        self.backend = "onnx"
        logger.info("VieNeu-TTS v3 Turbo ready (backend=onnx)")

        self.default_style = "tu_nhien"
        self._preset_voices: dict = {}
        self._default_voice: Optional[str] = None
        self.backbone_repo = backbone_repo
        self._load_v3_voices()

    # ── Preset voices (speaker embedding + reference codes) ─────────────────────
    def _load_v3_voices(self) -> None:
        """Load the built-in voices from assets/voices_v3_turbo.json."""
        import json
        path = Path(__file__).parent / "assets" / "voices_v3_turbo.json"
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        for name, v in data.get("presets", {}).items():
            emb = v.get("speaker_emb")
            codes = v.get("codes")
            self._preset_voices[name] = {
                "description": v.get("description", ""),
                "gender": v.get("gender", ""),
                "style": v.get("style", self.default_style),
                "speaker_emb": np.asarray(emb, dtype=np.float32) if emb is not None else None,
                "codes": np.asarray(codes, dtype=np.int64) if codes is not None else None,
            }
        self._default_voice = data.get("default_voice")
        logger.info(f"Loaded {len(self._preset_voices)} preset voices (default: {self._default_voice})")

    def list_preset_voices(self) -> List[tuple]:
        """Return ``[(label, voice_id), ...]`` for the built-in voices."""
        return [(f"{n} — {v['description']}" if v["description"] else n, n)
                for n, v in self._preset_voices.items()]

    def get_preset_voice(self, voice_name: Optional[str] = None) -> dict:
        name = voice_name or self._default_voice
        if name not in self._preset_voices:
            raise ValueError(f"Voice '{name}' not found. Available: {list(self._preset_voices)}")
        return self._preset_voices[name]

    def _resolve_ref(self, voice) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Resolve a preset voice name/dict to ``(speaker_emb, ref_codes)``."""
        preset = None
        if isinstance(voice, str):
            preset = self._preset_voices.get(voice)
            if preset is None:
                raise ValueError(f"Voice '{voice}' not found. Available: {list(self._preset_voices)}")
        elif isinstance(voice, dict):
            preset = voice
        elif self._default_voice:
            preset = self._preset_voices[self._default_voice]
        if preset is None:
            raise ValueError("Provide a preset `voice` name.")
        return preset["speaker_emb"], preset.get("codes")

    def infer_stream(
        self,
        text: str,
        voice: Optional[Union[str, dict]] = None,
        temperature: float = 0.8,
        top_k: int = 25,
        top_p: float = 0.95,
        max_new_frames: int = 300,
        repetition_penalty: float = 1.2,
        repetition_window: int = DEFAULT_REP_WINDOW,
        max_chars: int = 256,
        apply_watermark: bool = True,
        **kwargs: Any,
    ) -> Generator[np.ndarray, None, None]:
        speaker_emb, ref_codes = self._resolve_ref(voice)
        chunks, gaps = normalize_to_chunks_v3_with_gaps(text, max_chars=max_chars)
        pauses = gaps_to_silence(gaps)
        stream_fn = getattr(self.engine, "infer_stream", None)
        sr = self.sample_rate
        last_out: Optional[np.ndarray] = None
        for ci, chunk in enumerate(chunks):
            ph = phonemize_text_with_emotions(chunk)
            chunk_frames_cap = min(max_new_frames, max_expected_frames(ph))
            gen_kwargs = dict(
                phonemes=ph, speaker_emb=speaker_emb, ref_codes=ref_codes,
                temperature=temperature, top_k=top_k, top_p=top_p,
                max_new_frames=chunk_frames_cap, repetition_penalty=repetition_penalty,
                repetition_window=repetition_window,
            )
            subs = stream_fn(**gen_kwargs) if stream_fn is not None else (self.engine.infer(**gen_kwargs),)
            first = True
            for sub in subs:
                if sub is None or len(sub) == 0:
                    continue
                if first and last_out is not None:
                    pad = pause_pad_samples(last_out, sub, sr, pauses[ci - 1])
                    if pad > 0:
                        yield np.zeros(pad, dtype=np.float32)
                first = False
                last_out = sub
                yield self._apply_watermark(sub) if apply_watermark else sub

    def close(self) -> None:
        self.engine = None
