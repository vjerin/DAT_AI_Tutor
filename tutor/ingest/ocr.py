"""OCR backends for slide and code frames."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


class OCR:
    def read(self, image: str | Path) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class NoOCR(OCR):
    def read(self, image):
        return ""


class TesseractOCR(OCR):
    """Uses the ``tesseract`` CLI (apt install tesseract-ocr / brew install tesseract)."""

    def __init__(self, lang: str = "eng", psm: int = 4):
        if not shutil.which("tesseract"):
            raise RuntimeError("tesseract binary not found")
        self.lang, self.psm = lang, psm

    def _run(self, image, psm: int) -> str:
        res = subprocess.run(["tesseract", str(image), "stdout", "-l", self.lang, "--psm", str(psm)],
                             capture_output=True, text=True)
        return clean_ocr(res.stdout)

    def read(self, image):
        """Layout mode (psm 4) keeps table rows together; a sparse pass (psm 11)
        recovers isolated words such as white-on-dark slide titles."""
        main = self._run(image, self.psm)
        sparse = self._run(image, 11)
        have = set(re.findall(r"[a-z0-9]+", main.lower()))
        top, tail = [], []
        for i, line in enumerate(sparse.splitlines()):
            toks = re.findall(r"[a-z0-9]+", line.lower())
            if toks and not set(toks) <= have and sum(len(t) for t in toks) >= 3:
                (top if i < 2 else tail).append(line.strip(" _|"))
                have.update(toks)
        return "\n".join([*top, main, *tail]).strip()


class PaddleOCRBackend(OCR):  # pragma: no cover - heavy optional dependency
    """PaddleOCR (pip install paddleocr paddlepaddle). Better on slides and code."""

    def __init__(self, lang: str = "en"):
        from paddleocr import PaddleOCR
        self.engine = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)

    def read(self, image):
        result = self.engine.ocr(str(image), cls=True) or []
        lines = []
        for page in result:
            for box, (txt, conf) in page or []:
                if conf >= 0.5:
                    lines.append((box[0][1], box[0][0], txt))
        lines.sort()
        return clean_ocr("\n".join(t for _, _, t in lines))


def clean_ocr(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = line.rstrip()
        if not line.strip():
            continue
        # drop lines that are mostly noise characters
        alnum = sum(ch.isalnum() for ch in line)
        if alnum < max(2, 0.4 * len(line.strip())):
            continue
        lines.append(line)
    return "\n".join(lines)


CODE_HINTS = re.compile(
    r"(\bdef\b|\bimport\b|\breturn\b|\bclass\b|\bfor\b .* \bin\b|=\s*\w+\(|\w+\.\w+\(|\(\)|;\s*$|^\s{2,}\S|\{|\}|==|\+=|->)",
    re.M,
)


def looks_like_code(text: str) -> bool:
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    hits = sum(1 for l in lines if CODE_HINTS.search(l))
    return hits >= max(2, 0.4 * len(lines))


def tidy_code(text: str) -> str:
    """Undo common OCR spacing errors in code: ``loss. backward ()`` -> ``loss.backward()``."""
    text = re.sub(r"(\w)\.\s+(\w)", r"\1.\2", text)
    text = re.sub(r"(\w)\s+\(\s*\)", r"\1()", text)
    text = re.sub(r"(\.\w+)\s+\(", r"\1(", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"(\w) _(\w)", r"\1_\2", text)
    return text


def make_ocr(name: str) -> OCR:
    if name == "none":
        return NoOCR()
    if name == "paddle":
        return PaddleOCRBackend()
    if name == "tesseract":
        return TesseractOCR()
    raise ValueError(f"unknown OCR backend {name!r}")
