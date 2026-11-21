"""Ingests a course manifest into versioned, timestamped evidence units.

Manifest format: see examples/ocw_manifest.example.yaml. Paths are relative to
the manifest file.
"""
from __future__ import annotations

import logging
import shutil
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Optional

import yaml

from ..config import Config
from ..schema import EvidenceUnit, Modality
from ..store import EvidenceStore, Workspace
from ..text import normalize, sha256_text
from .asr import Word, load_transcript
from .captions import Captioner, NoCaptioner
from .concepts import tag_concepts
from .media import extract_audio, probe_duration_ms, sample_frames
from .ocr import OCR, NoOCR, looks_like_code, tidy_code
from .segment import segment_by_slides, segment_fixed
from .slides import SlideInterval, dedupe_intervals, detect_slides, pdf_pages

log = logging.getLogger(__name__)


def file_hash(path: Optional[Path]) -> str:
    if not path or not path.exists():
        return ""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Ingestor:
    def __init__(self, cfg: Config, ws: Workspace, store: EvidenceStore, ocr: OCR | None = None,
                 captioner: Captioner | None = None, scan: Callable[[str], float] | None = None,
                 asr=None):
        self.cfg, self.ws, self.store = cfg, ws, store
        self.ocr = ocr or NoOCR()
        self.captioner = captioner or NoCaptioner()
        self.scan = scan
        self.asr = asr

    def ingest_manifest(self, manifest_path: str | Path, force: bool = False) -> list[dict]:
        manifest_path = Path(manifest_path)
        m = yaml.safe_load(manifest_path.read_text())
        base = manifest_path.parent
        report = []
        for lec in m["lectures"]:
            report.append(self.ingest_lecture(m, lec, base, force=force))
        return report

    def ingest_lecture(self, course_meta: dict, lec: dict, base: Path, force: bool = False) -> dict:
        course = course_meta["course"]
        source_id = f"{course}/{lec['id']}"
        p = lambda key: (base / lec[key]) if lec.get(key) else None  # noqa: E731
        video, transcript, pdf = p("video"), p("transcript"), p("slides_pdf")
        content_hash = sha256_text(file_hash(video), file_hash(transcript), file_hash(pdf))
        src, is_new = self.store.register_source({
            "source_id": source_id, "course": course, "title": lec.get("title", lec["id"]),
            "media_uri": str(video.resolve()) if video else None,
            "slides_uri": str(pdf.resolve()) if pdf else None,
            "transcript_uri": str(transcript.resolve()) if transcript else None,
            "url": lec.get("url"), "licence": lec.get("licence", course_meta.get("licence")),
            "content_hash": content_hash,
        })
        if not is_new and not force:
            return {"source_id": source_id, "version": src.version, "status": "unchanged"}

        frames_dir = self.ws.path("frames", course, lec["id"], f"v{src.version}")
        frames_dir.mkdir(parents=True, exist_ok=True)
        words = self._words(video, transcript, frames_dir)
        slides: list[SlideInterval] = []
        duration = words[-1].end_ms if words else 0
        if video and video.exists():
            duration = probe_duration_ms(video)
            slides = self._slides(video, frames_dir, duration)

        units = self._build_units(src.source_id, src.version, course, src.title, words, slides, pdf, frames_dir)
        concept_lists = tag_concepts([u.search_text() for u in units])
        for u, cs in zip(units, concept_lists):
            u.concepts = cs
            if self.scan and self.cfg.get("ingest.scan_injection", True):
                if self.scan(u.search_text()) >= self.cfg.get("guard.threshold", 0.5):
                    u.flags.append("suspected_injection")
        self.store.add_units(units)
        counts: dict[str, int] = {}
        for u in units:
            counts[u.modality.value] = counts.get(u.modality.value, 0) + 1
        log.info("ingested %s v%s: %s", source_id, src.version, counts)
        return {"source_id": source_id, "version": src.version, "status": "ingested", "units": counts,
                "slides": len(slides), "flagged": sum("suspected_injection" in u.flags for u in units)}

    def _words(self, video: Optional[Path], transcript: Optional[Path], work: Path) -> list[Word]:
        if transcript and transcript.exists():
            return load_transcript(transcript)
        if self.asr and video and video.exists():
            wav = extract_audio(video, work / "audio.wav")
            return self.asr.transcribe(wav)
        log.warning("no transcript and no ASR configured; speech will not be indexed")
        return []

    def _slides(self, video: Path, frames_dir: Path, duration: int) -> list[SlideInterval]:
        tmp = frames_dir / "_samples"
        frames = sample_frames(video, tmp, fps=self.cfg.get("ingest.frame_fps", 1.0))
        slides = detect_slides(frames, duration, threshold=self.cfg.get("ingest.slide_change_threshold", 0.01),
                               min_slide_ms=self.cfg.get("ingest.min_slide_ms", 2000))
        kept = []
        for s in slides:
            dst = frames_dir / f"slide_{s.slide_no:03d}.jpg"
            shutil.copy(s.frame, dst)
            s.frame = dst
            kept.append(s)
        shutil.rmtree(tmp, ignore_errors=True)
        return kept

    def _build_units(self, source_id, version, course, title, words, slides, pdf, frames_dir) -> list[EvidenceUnit]:
        cfg = self.cfg
        prefix = f"{source_id}/v{version}"
        units: list[EvidenceUnit] = []

        # visual units: one per slide interval (repeat slides reuse OCR of the canonical one)
        canon = dedupe_intervals(slides) if slides else {}
        ocr_cache: dict[int, tuple[str, str]] = {}
        for s in slides:
            c = canon.get(s.slide_no, s.slide_no)
            if c not in ocr_cache:
                text = self.ocr.read(s.frame)
                caption = self.captioner.caption(s.frame, text)
                ocr_cache[c] = (text, caption)
            text, caption = ocr_cache[c]
            if not text and not caption:
                continue
            modality = Modality.CODE if looks_like_code(text) else Modality.SLIDE
            if modality == Modality.CODE:
                text = tidy_code(text)
            units.append(EvidenceUnit(
                evidence_id=f"{prefix}/slide{s.slide_no:03d}", source_id=source_id, source_version=version,
                course=course, lecture_title=title, modality=modality, text=text, caption=caption or None,
                t_start_ms=s.t_start_ms, t_end_ms=s.t_end_ms, slide_no=s.slide_no, frame_uri=str(s.frame),
                content_hash=sha256_text(text, caption or ""),
            ))

        # PDF pages, aligned to the slide interval with the most similar OCR text
        if pdf and pdf.exists():
            for page_no, text, img in pdf_pages(pdf, frames_dir / "pdf"):
                t0 = t1 = None
                best = 0.0
                for u in units:
                    if u.modality == Modality.PDF_PAGE:
                        continue
                    r = SequenceMatcher(None, normalize(text)[:600], normalize(u.text)[:600]).ratio()
                    if r > best:
                        best, t0, t1 = r, u.t_start_ms, u.t_end_ms
                if best < 0.35:
                    t0 = t1 = None
                units.append(EvidenceUnit(
                    evidence_id=f"{prefix}/page{page_no:03d}", source_id=source_id, source_version=version,
                    course=course, lecture_title=title, modality=Modality.PDF_PAGE, text=text.strip(),
                    t_start_ms=t0, t_end_ms=t1, slide_no=page_no, frame_uri=str(img),
                    content_hash=sha256_text(text),
                ))

        # transcript units
        if cfg.get("ingest.segment_policy", "slide") == "slide" and slides:
            segs = segment_by_slides(words, slides, cfg.get("ingest.max_segment_ms", 90000),
                                     cfg.get("ingest.min_segment_ms", 6000))
        else:
            segs = segment_fixed(words, cfg.get("ingest.fixed_window_ms", 30000))
        for k, seg in enumerate(segs, start=1):
            units.append(EvidenceUnit(
                evidence_id=f"{prefix}/seg{k:04d}", source_id=source_id, source_version=version,
                course=course, lecture_title=title, modality=Modality.TRANSCRIPT, text=seg.text,
                t_start_ms=seg.t_start_ms, t_end_ms=seg.t_end_ms, slide_no=seg.slide_no,
                content_hash=sha256_text(seg.text),
            ))
        units.sort(key=lambda u: (u.t_start_ms if u.t_start_ms is not None else 10**12, u.evidence_id))
        return units
