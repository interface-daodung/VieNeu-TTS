"""Text cleaning + phonemization cho VieNeu-TTS (minimal).

Đã cắt gọn từ https://github.com/pnnbao97/VieNeu-TTS (Apache-2.0) — xem NOTICE.
Giữ: normalize (sea-g2p, luôn punc_norm), emotion cue, chia chunk v3 theo câu.
Bỏ: phonemize batch/dict, chunk legacy v1/v2.
"""

import functools
import logging
import re
from typing import Optional
from sea_g2p import SEAPipeline, Normalizer, punc_norm

logger = logging.getLogger("Vieneu.Phonemizer")

RE_NEWLINE_SPLIT = re.compile(r"[\r\n]+")

# ---------------------------------------------------------------------------
# Inline non-verbal cues (emotion tokens)
# ---------------------------------------------------------------------------
_EMOTION_TAG_TO_K = {
    "chuckle": 1, "cười": 1, "cuoi": 1,
    "sigh": 2, "thở dài": 2, "tho dai": 2,
    "clear throat": 3, "hắng giọng": 3, "hang giong": 3,
}
_EMOTION_SPLIT_RE = re.compile(r"(\[[^\]]+\]|<\|emotion_\d+\|>)")
_ATTACHING_PUNCT = set(".,!?;:…)]}\"'’”")


def _emotion_tag_token(tag: str) -> Optional[str]:
    t = tag.strip()
    if t.startswith("<|"):
        return t
    inner = t[1:-1].strip().lower()
    k = _EMOTION_TAG_TO_K.get(inner)
    return f"<|emotion_{k}|>" if k is not None else None


class PuncNormalizer:
    """``sea_g2p.Normalizer`` với punc_norm mặc định True."""

    def __init__(self, lang: str = "vi") -> None:
        self._n = Normalizer(lang=lang)

    def normalize(self, text, punc_norm: bool = True):
        return self._n.normalize(text, punc_norm=punc_norm)

    def normalize_batch(self, texts, punc_norm: bool = True):
        return self._n.normalize_batch(texts, punc_norm=punc_norm)


_pipeline: SEAPipeline = None
_normalizer: PuncNormalizer = None

def _get_pipeline() -> SEAPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = SEAPipeline(lang="vi")
    return _pipeline

def _get_normalizer() -> PuncNormalizer:
    global _normalizer
    if _normalizer is None:
        _normalizer = PuncNormalizer()
    return _normalizer


@functools.lru_cache(maxsize=1024)
def _phonemize_cached(text: str, punc_norm: bool = True) -> str:
    return _get_pipeline().run(text, punc_norm=punc_norm)


def phonemize_text(text: str) -> str:
    """Normalize + phonemize một chuỗi."""
    return _phonemize_cached(text)


def phonemize_text_with_emotions(text: str) -> str:
    """Phonemize, giữ cue ``[cười]/[thở dài]/[hắng giọng]`` thành ``<|emotion_k|>``."""
    if "[" not in text and "<|emotion_" not in text:
        return _phonemize_cached(text)
    out = ""
    for i, part in enumerate(_EMOTION_SPLIT_RE.split(text)):
        token = _emotion_tag_token(part) if i % 2 == 1 else None
        if token is not None:
            out = (out + " " + token) if out else token
            continue
        ph = _phonemize_cached(part, punc_norm=False) if part and part.strip() else ""
        if not ph:
            continue
        if not out:
            out = ph
        elif ph[0] in _ATTACHING_PUNCT:
            out += ph
        else:
            out += " " + ph
    return punc_norm(out)


def _normalize_sentence_keep_cues(sentence: str, normalizer) -> str:
    if "[" not in sentence and "<|emotion_" not in sentence:
        return normalizer.normalize(sentence, punc_norm=False)
    parts = []
    for i, part in enumerate(_EMOTION_SPLIT_RE.split(sentence)):
        if i % 2 == 1:
            tok = _emotion_tag_token(part)
            parts.append(tok if tok is not None else part)
        elif part.strip():
            parts.append(normalizer.normalize(part, punc_norm=False))
    return " ".join(p for p in parts if p)


def _normalized_sentences_by_para(
    text: str,
    keep_cues: bool = False,
) -> list[list[str]]:
    """Text thô -> list ĐOẠN, mỗi đoạn là list CÂU đã normalize (tách câu trước, normalize sau)."""
    from vieneu_utils.core_utils import split_into_sentences

    paragraphs = [p for p in RE_NEWLINE_SPLIT.split(text) if p.strip()]
    if not paragraphs:
        return []

    normalizer = _get_normalizer()
    out: list[list[str]] = []
    for para in paragraphs:
        sentences = split_into_sentences(para)
        if not sentences:
            continue
        if keep_cues:
            out.append([_normalize_sentence_keep_cues(s, normalizer) for s in sentences])
        else:
            out.append(normalizer.normalize_batch(sentences, punc_norm=False))
    return out


_EMOTION_TOKEN_RE = re.compile(r"<\|emotion_\d+\|>")


def _effective_len(chunk: str) -> int:
    return len(_EMOTION_TOKEN_RE.sub("", chunk).strip())


def _merge_short_chunks(
    chunks: list[str], gaps: list[str], min_chars: int
) -> tuple[list[str], list[str]]:
    """Gộp chunk ngắn hơn ``min_chars`` vào chunk kề (chống ảo giác AR)."""
    while len(chunks) > 1:
        short = [i for i, c in enumerate(chunks) if _effective_len(c) < min_chars]
        if not short:
            break
        i = min(short, key=lambda k: _effective_len(chunks[k]))
        sides = []
        if i < len(chunks) - 1:
            sides.append(((gaps[i] != "para", -len(chunks[i + 1])), "R"))
        if i > 0:
            sides.append(((gaps[i - 1] != "para", -len(chunks[i - 1])), "L"))
        side = max(sides, key=lambda t: t[0])[1]
        if side == "R":
            chunks[i] = chunks[i] + " " + chunks.pop(i + 1)
            gaps.pop(i)
        else:
            chunks[i - 1] = chunks[i - 1] + " " + chunks.pop(i)
            gaps.pop(i - 1)
    return chunks, gaps


def normalize_to_chunks_v3_with_gaps(
    text: str, max_chars: int = 256, min_chunk_chars: int = 20
) -> tuple[list[str], list[str]]:
    """Chia text thành chunk (kèm loại ranh giới ``para/sentence/minor`` để nghỉ đúng nhịp)."""
    from vieneu_utils.core_utils import pack_sentences_into_chunks, _classify_gap

    if not text:
        return [], []

    keep_cues = "[" in text or "<|emotion_" in text
    chunks: list[str] = []
    gaps: list[str] = []
    for sentences in _normalized_sentences_by_para(text, keep_cues=keep_cues):
        para_chunks = pack_sentences_into_chunks(sentences, max_chars=max_chars)
        if not para_chunks:
            continue
        if chunks:
            gaps.append("para")
        for j, ch in enumerate(para_chunks):
            if j > 0:
                gaps.append("sentence")
            chunks.append(ch)

    chunks = [punc_norm(c) for c in chunks]
    gaps = [g if g == "para" else _classify_gap(chunks[i]) for i, g in enumerate(gaps)]
    return _merge_short_chunks(chunks, gaps, min_chunk_chars)
