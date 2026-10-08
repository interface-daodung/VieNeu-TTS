from typing import Any
import numpy as np
import logging

logger = logging.getLogger("Vieneu")


class BaseVieneuTTS:
    """Minimal base: sample rate + optional watermark + lifecycle."""

    def __init__(self):
        self.sample_rate = 48_000
        self.watermarker = None
        self._init_watermarker()

    def _init_watermarker(self) -> None:
        """Initialize optional audio watermarker (Resemble Perth)."""
        try:
            import perth
        except ImportError:
            self.watermarker = None
            return
        try:
            wm_cls = getattr(perth, "PerthImplicitWatermarker", None)
            if wm_cls is None:
                raise AttributeError("installed 'perth' package is not resemble-perth")
            self.watermarker = wm_cls()
            logger.info("Audio watermarking initialized (Perth)")
        except Exception as e:
            self.watermarker = None
            logger.warning("Watermarker init failed — output will NOT be watermarked: %s", e)

    def _apply_watermark(self, wav: np.ndarray) -> np.ndarray:
        if self.watermarker:
            return self.watermarker.apply_watermark(wav, sample_rate=self.sample_rate)
        return wav

    def close(self) -> None:
        pass

    def __enter__(self) -> 'BaseVieneuTTS':
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
