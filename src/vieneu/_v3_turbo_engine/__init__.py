"""Minimal engine package: torch-free ONNX streaming engine only.

Đã cắt gọn từ https://github.com/pnnbao97/VieNeu-TTS (Apache-2.0) — xem NOTICE.
"""

__all__ = ["OnnxV3LiteEngine"]


def __getattr__(name):  # PEP 562 lazy attribute import
    if name == "OnnxV3LiteEngine":
        from .onnx_runtime_lite import OnnxV3LiteEngine
        return OnnxV3LiteEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
