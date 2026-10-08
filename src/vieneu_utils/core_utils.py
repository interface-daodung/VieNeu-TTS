"""Text-chunking + audio-gap helpers cho đường v3 streaming (minimal).

Đã cắt gọn từ https://github.com/pnnbao97/VieNeu-TTS (Apache-2.0) — xem NOTICE.
Giữ: tách câu (nhận biết ngoặc/trích dẫn), đóng gói chunk, phân loại ranh giới,
khoảng nghỉ, đo im lặng mép, trần frame theo phoneme.
Bỏ: split v1/v2 legacy, PhoneChunk, join/trim/fade, babble guard, env_bool.
"""
import re
from typing import List, Tuple, Optional

import numpy as np

# ─── Regex ───────────────────────────────────────────────────────────────────

RE_MINOR_PUNCT = re.compile(r'(?<=[\,\;\:\-\–\—])\s+')

RE_TOKEN_KEEP_EN = re.compile(r'<en>.*?</en>|\S+', re.IGNORECASE | re.DOTALL)


def _tokenize_keep_en(s: str) -> List[str]:
    return RE_TOKEN_KEEP_EN.findall(s)

# ─── Tách câu nhận biết ngoặc/trích dẫn ──────────────────────────────────────
_OPEN_TO_CLOSE = {
    '(': ')', '[': ']', '{': '}',
    '“': '”', '‘': '’', '«': '»', '‹': '›', '「': '」', '『': '』',
}
_OPENERS = frozenset(_OPEN_TO_CLOSE)
_CLOSERS = frozenset(_OPEN_TO_CLOSE.values())
_SYMMETRIC_QUOTE = '"'
_SENT_END_CHARS = frozenset('.!?…')
_TRAILING_CLOSE = _CLOSERS | frozenset('"\'’”')

# ─── Khoảng nghỉ theo loại ranh giới ─────────────────────────────────────────
V3_GAP_SILENCE = {"para": 0.70, "sentence": 0.50, "minor": 0.30}

# ─── Trần frame theo phoneme ─────────────────────────────────────────────────
MAX_FRAMES_PER_PHONE = 2.0
_FRAME_CAP_SLACK = 24
_FRAME_MARKUP_RE = re.compile(r"<\|emotion_\d+\|>|</?en>")
SINGLE_WORD_MAX_FRAMES = 13
SYLLABLE_CAP_PER_EXTRA = 5
SYLLABLE_CAP_MAX_SYL = 4
_SINGLE_WORD_MAX_PHONES = 24

_IPA_VOWELS = set("aeiouyæɐɑɒɔəɘɛɜɤɯɵøœʉʊʌɪɨɚɝᵻᵿ")


def is_cue_only(phonemes: str) -> bool:
    ph = phonemes or ""
    return "<|emotion_" in ph and not any(ch.isalpha() for ch in _FRAME_MARKUP_RE.sub("", ph))


def syllable_count(phonemes: str) -> int:
    """Số tiếng ước lượng từ chuỗi phoneme (tiếng Việt: mỗi từ một tiếng)."""
    stripped = _FRAME_MARKUP_RE.sub("", phonemes or "")
    total = 0
    for tok in stripped.split():
        groups, in_v, consonant_seen = 0, False, True
        for ch in tok:
            if ch in _IPA_VOWELS:
                if not in_v and consonant_seen:
                    groups += 1
                in_v, consonant_seen = True, False
            elif ch in "ːˈˌ" or ch.isdigit():
                if ch in "ˈˌ" and groups > 0:
                    in_v, consonant_seen = False, True
                else:
                    in_v = False
            else:
                in_v, consonant_seen = False, True
        if any(ch.isalpha() for ch in tok):
            total += max(1, groups)
    return total


def max_expected_frames(phonemes: str) -> int:
    """Số frame TỐI ĐA hợp lý cho chunk (chặn chunk ngắn 'nói thêm')."""
    stripped = _FRAME_MARKUP_RE.sub("", phonemes or "")
    eff_len = len(stripped)
    cap = _FRAME_CAP_SLACK + int(np.ceil(MAX_FRAMES_PER_PHONE * eff_len))
    if is_cue_only(phonemes):
        return min(cap, SINGLE_WORD_MAX_FRAMES)
    if "<|emotion_" not in (phonemes or ""):
        syl = max(1, syllable_count(phonemes))
        if syl <= SYLLABLE_CAP_MAX_SYL and eff_len <= _SINGLE_WORD_MAX_PHONES * syl:
            cap = min(cap, SINGLE_WORD_MAX_FRAMES + SYLLABLE_CAP_PER_EXTRA * (syl - 1))
    return cap


def gaps_to_silence(gaps: List[str]) -> List[float]:
    return [V3_GAP_SILENCE.get(g, V3_GAP_SILENCE["sentence"]) for g in gaps]


def edge_silence(wav: np.ndarray, sr: int, thresh_db: float = -45.0, win_s: float = 0.01) -> Tuple[int, int]:
    """``(lead, tail)``: số mẫu im lặng ở đầu/cuối ``wav``."""
    n_samp = int(wav.size)
    win = max(1, int(win_s * sr))
    n_win = n_samp // win
    if n_win == 0:
        return n_samp, 0
    env = np.abs(wav[: n_win * win]).reshape(n_win, win).mean(1)
    above = np.flatnonzero(env > 10 ** (thresh_db / 20))
    if not above.size:
        return n_samp, 0
    return int(above[0]) * win, n_samp - (int(above[-1]) + 1) * win


def pause_pad_samples(prev_wav: np.ndarray, next_wav: np.ndarray, sr: int, pause_s: float) -> int:
    """Số mẫu zeros cần chèn để khoảng nghỉ thật đạt ``pause_s`` (streaming)."""
    lead_prev, tail = edge_silence(prev_wav, sr)
    if lead_prev == prev_wav.size:
        tail = prev_wav.size
    lead, _ = edge_silence(next_wav, sr)
    return max(0, int(pause_s * sr) - tail - lead)


def _scan_sentences(text: str, quote_aware: bool = True) -> Tuple[List[str], bool]:
    sentences: List[str] = []
    n = len(text)
    start = i = 0
    depth = 0
    in_quote = False

    while i < n:
        ch = text[i]
        if quote_aware and ch == _SYMMETRIC_QUOTE:
            in_quote = not in_quote
        elif quote_aware and ch in _OPENERS:
            depth += 1
        elif quote_aware and ch in _CLOSERS:
            if depth:
                depth -= 1
        elif ch in _SENT_END_CHARS and depth == 0 and not in_quote:
            j = i + 1
            while j < n and text[j] in _SENT_END_CHARS:
                j += 1
            while j < n and text[j] in _TRAILING_CLOSE:
                j += 1
            if j >= n or text[j].isspace():
                sentences.append(text[start:j])
                start = i = j
                continue
            i = j
            continue
        i += 1

    if start < n:
        sentences.append(text[start:])

    return [s.strip() for s in sentences if s.strip()], (depth == 0 and not in_quote)


def split_into_sentences(text: str) -> List[str]:
    """Tách text THÔ thành câu, không cắt trong ngoặc/trích dẫn."""
    if not text:
        return []
    sentences, balanced = _scan_sentences(text, quote_aware=True)
    if not balanced:
        sentences, _ = _scan_sentences(text, quote_aware=False)
    return sentences


# ─── Cắt mảnh dài: ưu tiên từ nối ────────────────────────────────────────────
_CONN_WORDS = frozenset(
    "và nhưng hoặc song rồi nên vì nếu khi để do bởi".split()
)
_CONN_PAIRS = frozenset([
    ("sau", "khi"), ("trước", "khi"), ("trong", "khi"), ("mỗi", "khi"),
    ("đến", "khi"), ("tới", "khi"), ("cho", "nên"), ("cho", "đến"),
    ("bởi", "vì"), ("nếu", "như"), ("tuy", "nhiên"), ("thế", "nhưng"),
    ("vì", "vậy"), ("vì", "thế"), ("do", "đó"), ("sau", "đó"),
])
_CONN_STRIP = "\"'“”‘’()[]«»…"


def _conn_key(token: str) -> str:
    return token.strip(_CONN_STRIP).lower()


def _natural_cut(words: List[str], start: int, end: int, min_left: int) -> Optional[int]:
    left = sum(len(w) for w in words[start:end]) + (end - start - 1)
    for j in range(end - 1, start, -1):
        left -= len(words[j]) + 1
        if left < min_left:
            return None
        key, prev = _conn_key(words[j]), _conn_key(words[j - 1])
        if (prev, key) in _CONN_PAIRS or prev in _CONN_WORDS:
            continue
        nxt = _conn_key(words[j + 1]) if j + 1 < len(words) else ""
        if key in _CONN_WORDS or (key, nxt) in _CONN_PAIRS:
            return j
    return None


CHUNK_TAIL_SLACK = 15


def _tail_slack(max_chars: int) -> int:
    return min(CHUNK_TAIL_SLACK, max_chars // 8)


def _fits(cur_len: int, add_len: int, max_chars: int) -> bool:
    total = cur_len + 1 + add_len if cur_len else add_len
    slack = _tail_slack(max_chars)
    return total <= max_chars or (add_len <= slack and total <= max_chars + slack)


def _split_long_part(part: str, max_chars: int) -> List[str]:
    words = _tokenize_keep_en(part)
    min_left = max_chars // 2
    pieces: List[str] = []
    start = 0
    while start < len(words):
        end, length = start, 0
        while end < len(words):
            add = length + 1 + len(words[end]) if end > start else len(words[end])
            if end > start and add > max_chars:
                break
            length, end = add, end + 1
        if end < len(words):
            rest = sum(len(w) for w in words[end:]) + (len(words) - end - 1)
            if _fits(length, rest, max_chars):
                end = len(words)
                pieces.append(" ".join(words[start:end]))
                break
            cut = _natural_cut(words, start, end, min_left)
            if cut is not None:
                end = cut
            else:
                while end > start + 1 and (
                    _conn_key(words[end - 1]), _conn_key(words[end])
                ) in _CONN_PAIRS:
                    end -= 1
        pieces.append(" ".join(words[start:end]))
        start = end
    return pieces


def pack_sentences_into_chunks(sentences: List[str], max_chars: int = 256) -> List[str]:
    """Đóng gói các CÂU thành chunk ~<= ``max_chars`` (greedy, giữ thứ tự)."""
    final_chunks: List[str] = []
    buffer = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > max_chars:
            if buffer:
                final_chunks.append(buffer)
                buffer = ""

            sub_parts = RE_MINOR_PUNCT.split(sentence)
            for part in sub_parts:
                part = part.strip()
                if not part:
                    continue
                if _fits(len(buffer), len(part), max_chars):
                    buffer = (buffer + ' ' + part) if buffer else part
                else:
                    if buffer:
                        final_chunks.append(buffer)
                    buffer = part
                    if len(buffer) > max_chars:
                        pieces = _split_long_part(buffer, max_chars)
                        final_chunks.extend(pieces[:-1])
                        buffer = pieces[-1] if pieces else ""
        else:
            if buffer and not _fits(len(buffer), len(sentence), max_chars):
                final_chunks.append(buffer)
                buffer = sentence
            else:
                buffer = (buffer + ' ' + sentence) if buffer else sentence

    if buffer:
        final_chunks.append(buffer)

    return [c.strip() for c in final_chunks if c.strip()]


def _classify_gap(chunk: str) -> str:
    c = chunk.rstrip()
    return "sentence" if c and c[-1] in ".!?" else "minor"
