"""Slide-change detection on sampled frames, repeat detection and PDF pages."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


def dhash(path: str | Path, size: int = 8) -> int:
    import numpy as np
    img = Image.open(path).convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    px = np.asarray(img).reshape(-1).tolist()
    bits = 0
    for row in range(size):
        for col in range(size):
            left = px[row * (size + 1) + col]
            right = px[row * (size + 1) + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def thumb(path: str | Path, size: tuple[int, int] = (160, 90)):
    import numpy as np
    return np.asarray(Image.open(path).convert("L").resize(size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def frame_diff(a, b) -> float:
    """Fraction of thumbnail pixels that changed noticeably (robust to compression noise)."""
    import numpy as np
    return float(np.mean(np.abs(a - b) > 0.12))


@dataclass
class SlideInterval:
    slide_no: int
    t_start_ms: int
    t_end_ms: int
    frame: Path          # representative frame (middle of the interval)


def detect_slides(frames: list[tuple[int, Path]], duration_ms: int, threshold: float = 0.01,
                  min_slide_ms: int = 2000) -> list[SlideInterval]:
    """``threshold``: fraction of changed thumbnail pixels that starts a new slide."""
    if not frames:
        return []
    thumbs = [thumb(p) for _, p in frames]
    starts = [0]
    for i in range(1, len(frames)):
        if frame_diff(thumbs[i], thumbs[i - 1]) > threshold:
            starts.append(i)
    # build raw intervals
    raw: list[tuple[int, int, int]] = []   # (start_idx, end_idx_exclusive, start_ms)
    for j, s in enumerate(starts):
        e = starts[j + 1] if j + 1 < len(starts) else len(frames)
        raw.append((s, e, frames[s][0]))
    # merge flickers shorter than min_slide_ms into previous interval
    merged: list[list[int]] = []
    for s, e, t0 in raw:
        t1 = frames[e][0] if e < len(frames) else duration_ms
        if merged and (t1 - t0) < min_slide_ms:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    out = []
    for n, (s, e) in enumerate(merged, start=1):
        t0 = frames[s][0]
        t1 = frames[e][0] if e < len(frames) else duration_ms
        mid = frames[(s + e - 1) // 2][1]
        out.append(SlideInterval(slide_no=n, t_start_ms=t0, t_end_ms=max(t1, t0 + 1), frame=mid))
    return out


def dedupe_intervals(intervals: list[SlideInterval], threshold: int = 4) -> dict[int, int]:
    """Map slide_no -> canonical slide_no for slides that reappear (e.g. instructor flips back)."""
    canon: dict[int, int] = {}
    seen: list[tuple[int, int]] = []
    for iv in intervals:
        h = dhash(iv.frame)
        match = next((n for n, hh in seen if hamming(h, hh) <= threshold), None)
        if match is None:
            seen.append((iv.slide_no, h))
            canon[iv.slide_no] = iv.slide_no
        else:
            canon[iv.slide_no] = match
    return canon


def pdf_pages(pdf_path: str | Path, out_dir: str | Path, dpi: int = 110) -> list[tuple[int, str, Path]]:
    """Return [(page_no, text, image_path)] for a slide/notes PDF (needs PyMuPDF)."""
    try:
        import fitz  # PyMuPDF
    except ImportError as e:  # pragma: no cover - optional dependency
        raise RuntimeError("PyMuPDF is required for PDF ingestion: pip install pymupdf") from e
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc, start=1):
        img = out_dir / f"page_{i:03d}.png"
        page.get_pixmap(dpi=dpi).save(str(img))
        pages.append((i, page.get_text("text"), img))
    return pages
