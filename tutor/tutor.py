"""Tutor: ties ingestion, retrieval, generation, verification and memory together."""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from .cache import AnswerCache
from .config import Config
from .generate.generator import make_generator
from .generate.llm import make_llm
from .index.manager import IndexManager
from .ingest.captions import make_captioner
from .ingest.ocr import make_ocr
from .ingest.pipeline import Ingestor
from .memory.service import MemoryService
from .retrieve.classify import make_classifier
from .retrieve.pipeline import Retriever
from .retrieve.rerank import LexicalReranker
from .retrieve.rewrite import make_rewriter
from .schema import Citation, EvidenceUnit, RetrievedItem, Status, TutorResponse
from .store import EvidenceStore, Workspace
from .text import ms_to_clock
from .verify.abstain import coverage_features, extract_features, make_abstainer
from .verify.guard import make_guard, scrub_output
from .verify.support import make_verifier

log = logging.getLogger(__name__)


class Tutor:
    def __init__(self, cfg: Config, ws: Workspace):
        self.cfg, self.ws = cfg, ws
        self.store = EvidenceStore(ws.db_path)
        self.llm = make_llm(cfg)
        self.guard = make_guard(cfg)
        self.cache = AnswerCache()
        self.store.on_new_version(self.cache.invalidate_source)
        self.memory = MemoryService(cfg.section("memory"), str(ws.path("memory.db")), llm=self.llm)
        self.memory.on_delete(self.cache.invalidate_memory)
        self.classifier = make_classifier(cfg)
        self.rewriter = make_rewriter(cfg, self.llm)
        self.verifier = make_verifier(cfg, self.llm)
        self.abstainer = make_abstainer(cfg)
        self.index = IndexManager(cfg, ws.path("indexes", cfg.get("index_name", "default")))
        self.retriever: Optional[Retriever] = None
        self.generator = None
        try:
            self.index.load()
            self._build_runtime()
        except FileNotFoundError:
            log.info("no index yet; ingest and index first")

    @classmethod
    def open(cls, workspace: str | Path, config: str | Path | None = None, overrides: list[str] | None = None) -> "Tutor":
        return cls(Config.load(config, overrides), Workspace(workspace))

    def _build_runtime(self) -> None:
        self.retriever = Retriever(self.cfg, self.store, self.index)
        scorer = LexicalReranker(self.index.bm25.idf if self.index.bm25 else None).score
        self.generator = make_generator(self.cfg, self.llm, scorer)

    def ingest(self, manifest: str | Path, force: bool = False, reindex: bool = True) -> list[dict]:
        ing = Ingestor(self.cfg, self.ws, self.store, ocr=make_ocr(self.cfg.get("ingest.ocr", "tesseract")),
                       captioner=make_captioner(self.cfg.get("ingest.captioner", "none"), self.llm),
                       scan=self.guard.score, asr=self._asr())
        report = ing.ingest_manifest(manifest, force=force)
        for r in report:
            if r["status"] != "unchanged":
                self.cache.invalidate_source(r["source_id"])
        if reindex:
            self.reindex()
        return report

    def _asr(self):
        if self.cfg.get("ingest.asr") == "faster_whisper":
            from .ingest.asr import FasterWhisperASR
            return FasterWhisperASR(self.cfg.get("ingest.asr_model", "large-v3"))
        return None

    def reindex(self) -> dict:
        info = self.index.build(self.store.all_units())
        self._build_runtime()
        self.cache.clear()
        return info

    def ask(self, question: str, learner_id: str = "anon", session_id: str = "default",
            use_memory: bool = True, use_cache: bool = True, allowed: Optional[set[str]] = None) -> TutorResponse:
        if self.retriever is None:
            raise RuntimeError("no index: run ingest first")
        c = self.cfg
        timings: dict[str, float] = {}
        t = time.perf_counter()
        mem_on = use_memory and c.get("memory.enabled", True)
        history = self.memory.history(learner_id, session_id) if mem_on else []
        rewritten = self.rewriter.rewrite(question, history)
        qclass = self.classifier.predict(rewritten)
        timings["rewrite_classify"] = (time.perf_counter() - t) * 1000

        items, rinfo = self.retriever.retrieve(rewritten, qclass, allowed=allowed)
        timings.update(rinfo.pop("timings_ms"))
        t = time.perf_counter()
        concepts = self._concepts(items)
        profile_items = self.memory.profile(learner_id, concepts) if mem_on else []
        profile = self.memory.profile_lines(profile_items)
        timings["memory_read"] = (time.perf_counter() - t) * 1000

        key = self.cache.key(rewritten, [p.item_id for p in profile_items], c.get("name", "default"))
        cached = self.cache.get(key) if (use_cache and c.get("cache.enabled", True) and allowed is None) else None
        if cached:
            cached.cached = True
            cached.query = question
            self._after_turn(learner_id, session_id, question, cached, concepts, mem_on)
            return cached

        t = time.perf_counter()
        gen = self.generator.generate(rewritten, items, profile)
        timings["generate"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        by_id = {it.unit.evidence_id: it.unit for it in items}
        evidence_texts = [u.search_text() for u in by_id.values()]
        segments = []
        dropped = []
        thr = c.get("verifier.support_threshold", 0.6)
        for s in gen.segments:
            s.text = scrub_output(s.text, evidence_texts)
            s.citations = [x for x in s.citations if x in by_id]
            if not s.citations:
                dropped.append({"text": s.text, "reason": "no valid citation"})
                continue
            s.support_score = self.verifier.score(s.text, [by_id[x].search_text() for x in s.citations])
            s.supported = s.support_score >= thr
            if not s.supported and c.get("verifier.drop_unsupported", True):
                dropped.append({"text": s.text, "reason": f"unsupported ({s.support_score:.2f})"})
                continue
            segments.append(s)
        feats = extract_features(items, segments, gen.status, qclass)
        cited_texts = [by_id[x].search_text() for s in segments for x in s.citations] or evidence_texts[:3]
        bm25 = self.index.bm25
        feats.update(coverage_features(rewritten, cited_texts, " ".join(s.text for s in segments),
                                       bm25.idf if bm25 else None, bm25.postings if bm25 else None))
        if dropped and gen.segments:
            feats["frac_supported"] = len(segments) / len(gen.segments)
        decision, conf = self.abstainer.decide(feats)
        status = decision
        if gen.status == Status.ABSTAINED or not segments:
            status = Status.ABSTAINED
        elif gen.status == Status.QUALIFIED and status == Status.ANSWERED:
            status = Status.QUALIFIED
        timings["verify"] = (time.perf_counter() - t) * 1000

        resp = TutorResponse(query=question, rewritten_query=rewritten, query_class=qclass, status=status,
                             confidence=round(conf, 4), learner_profile_used=profile, timings_ms={})
        if status == Status.ABSTAINED:
            resp.abstain_reason = gen.abstain_reason or (
                "The course material does not clearly answer this." if segments or gen.segments else
                "This does not appear to be covered in the course material.")
            resp.suggestions = [self.citation(it.unit) for it in items[:2] if (it.rerank_score or 0) >= 0.3]
        else:
            resp.segments = segments
            for s in segments:
                for eid in s.citations:
                    resp.citations[eid] = self.citation(by_id[eid])
        resp.trace = {"retrieval": rinfo, "features": {k: round(v, 4) for k, v in feats.items()},
                      "generator_status": gen.status.value, "dropped_segments": dropped,
                      "evidence": [it.unit.evidence_id for it in items], "concepts": concepts,
                      "raw": (gen.raw or "")[:2000]}
        resp.timings_ms = {k: round(v, 2) for k, v in timings.items()}
        resp.timings_ms["total"] = round(sum(timings.values()), 2)
        if c.get("cache.enabled", True) and allowed is None:
            self.cache.put(key, resp, [p.item_id for p in profile_items])
        self._after_turn(learner_id, session_id, question, resp, concepts, mem_on)
        return resp

    def _after_turn(self, learner_id, session_id, question, resp: TutorResponse, concepts, mem_on) -> None:
        if not mem_on:
            return
        t = time.perf_counter()
        turn_id = uuid.uuid4().hex[:10]
        self.memory.record_turn(learner_id, session_id, "user", question)
        self.memory.record_turn(learner_id, session_id, "assistant", resp.answer_text or (resp.abstain_reason or ""))
        resp.trace["memory_ops"] = self.memory.observe(learner_id, session_id, question, concepts, turn_id,
                                                       list(resp.citations))
        resp.timings_ms["memory_write"] = round((time.perf_counter() - t) * 1000, 2)

    @staticmethod
    def _concepts(items: list[RetrievedItem], n: int = 3) -> list[str]:
        out: list[str] = []
        for it in items[:n]:
            for cpt in it.unit.concepts:
                if cpt not in out:
                    out.append(cpt)
        return out[:6]

    def citation(self, u: EvidenceUnit) -> Citation:
        src = self.store.get_source(u.source_id, u.source_version)
        sec = (u.t_start_ms or 0) // 1000
        link = None
        if src and src.url:
            link = _with_time(src.url, sec) if u.t_start_ms is not None else src.url
        elif src and src.media_uri:
            link = f"/media/{u.source_id}/v{u.source_version}#t={sec}" if u.t_start_ms is not None else None
        when = f"{ms_to_clock(u.t_start_ms)}-{ms_to_clock(u.t_end_ms)}" if u.t_start_ms is not None else ""
        bits = [u.lecture_title, f"v{u.source_version}"]
        if when:
            bits.append(when)
        if u.slide_no is not None and u.is_visual:
            bits.append(f"slide {u.slide_no}" if u.modality.value != "pdf_page" else f"page {u.slide_no}")
        superseded = not u.is_current
        if superseded:
            bits.append("superseded")
        return Citation(evidence_id=u.evidence_id, source_id=u.source_id, version=u.source_version,
                        lecture_title=u.lecture_title, modality=u.modality, t_start_ms=u.t_start_ms,
                        t_end_ms=u.t_end_ms, slide_no=u.slide_no,
                        frame_uri=f"/frames/{u.evidence_id}" if u.frame_uri else None,
                        link=link, label=", ".join(bits), superseded=superseded, excerpt=u.text[:300])


def _with_time(url: str, sec: int) -> str:
    p = urlparse(url)
    if "youtube.com" in p.netloc or "youtu.be" in p.netloc:
        q = parse_qs(p.query)
        q["t"] = [f"{sec}s"]
        return urlunparse(p._replace(query=urlencode(q, doseq=True)))
    return f"{url}#t={sec}"
