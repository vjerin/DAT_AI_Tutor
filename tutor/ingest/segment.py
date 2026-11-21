"""Splits timed words into transcript segments (slide-bounded or fixed windows)."""
from __future__ import annotations

from dataclasses import dataclass

from .asr import Word
from .slides import SlideInterval

SENT_END = (".", "?", "!")


@dataclass
class Segment:
    t_start_ms: int
    t_end_ms: int
    text: str
    slide_no: int | None = None


def _join(words: list[Word]) -> str:
    return " ".join(w.text for w in words).replace(" ,", ",").replace(" .", ".")


def _split_long(words: list[Word], max_ms: int, min_ms: int) -> list[list[Word]]:
    """Split a run of words into chunks <= max_ms, preferring sentence ends."""
    chunks: list[list[Word]] = []
    cur: list[Word] = []
    for w in words:
        cur.append(w)
        span = cur[-1].end_ms - cur[0].start_ms
        at_sentence_end = w.text.endswith(SENT_END)
        if (span >= max_ms * 0.6 and at_sentence_end) or span >= max_ms:
            chunks.append(cur)
            cur = []
    if cur:
        if chunks and (cur[-1].end_ms - cur[0].start_ms) < min_ms:
            chunks[-1].extend(cur)
        else:
            chunks.append(cur)
    return chunks


def segment_by_slides(words: list[Word], slides: list[SlideInterval], max_ms: int = 90000,
                      min_ms: int = 6000) -> list[Segment]:
    if not slides:
        return segment_fixed(words, 30000)
    out: list[Segment] = []
    buckets: dict[int, list[Word]] = {s.slide_no: [] for s in slides}
    for w in words:
        mid = (w.start_ms + w.end_ms) // 2
        slide = next((s for s in slides if s.t_start_ms <= mid < s.t_end_ms), slides[-1] if mid >= slides[-1].t_start_ms else slides[0])
        buckets[slide.slide_no].append(w)
    for s in slides:
        ws = buckets[s.slide_no]
        if not ws:
            continue
        for chunk in _split_long(ws, max_ms, min_ms):
            out.append(Segment(t_start_ms=chunk[0].start_ms, t_end_ms=chunk[-1].end_ms,
                               text=_join(chunk), slide_no=s.slide_no))
    return out


def segment_fixed(words: list[Word], window_ms: int = 30000) -> list[Segment]:
    out: list[Segment] = []
    cur: list[Word] = []
    for w in words:
        cur.append(w)
        span = cur[-1].end_ms - cur[0].start_ms
        if span >= window_ms or (span >= window_ms * 0.8 and w.text.endswith(SENT_END)):
            out.append(Segment(cur[0].start_ms, cur[-1].end_ms, _join(cur)))
            cur = []
    if cur:
        out.append(Segment(cur[0].start_ms, cur[-1].end_ms, _join(cur)))
    return out
