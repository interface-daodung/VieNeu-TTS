"""Minimal engine package: torch-free ONNX streaming engine only."""

__all__ = ["OnnxV3LiteEngine"]


def __getattr__(name):  # PEP 562 lazy attribute import
    if name == "OnnxV3LiteEngine":
        from .onnx_runtime_lite import OnnxV3LiteEngine
        return OnnxV3LiteEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
