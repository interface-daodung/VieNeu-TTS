"""Minimal factory — đã cắt gọn từ https://github.com/pnnbao97/VieNeu-TTS (Apache-2.0), xem NOTICE."""
def Vieneu(mode="v3turbo", **kwargs):
    """Minimal factory: only the v3 Turbo ONNX/CPU backend (preset-voice TTS)."""
    if mode not in ("v3turbo", "onnx"):
        raise ValueError(f"Unsupported mode {mode!r}: minimal server only supports 'v3turbo'.")
    from .v3turbo import V3TurboVieNeuTTS
    return V3TurboVieNeuTTS(**kwargs)
