"""Answer generation with per-sentence citations (extractive baseline and LLM)."""
from __future__ import annotations

import logging
from pathlib import Path

from ..schema import AnswerSegment, RetrievedItem, Status
from ..text import split_lines, split_sentences
from .llm import LLM, parse_json
from .prompts import build_messages

log = logging.getLogger(__name__)


class GenerationResult:
    def __init__(self, status: Status, segments: list[AnswerSegment], abstain_reason: str | None = None,
                 raw: str | None = None):
        self.status, self.segments, self.abstain_reason, self.raw = status, segments, abstain_reason, raw


class ExtractiveGenerator:
    def __init__(self, scorer, max_sentences: int = 3, min_score: float = 0.25):
        self.scorer = scorer                     # callable(query, text) -> 0..1
        self.max_sentences, self.min_score = max_sentences, min_score

    def _pieces(self, u) -> list[tuple[str, str, str]]:
        """Returns (shown text, scoring context, raw text) for each sentence or slide line.

        Sentences are scored with the two previous sentences as context; slide
        lines get the slide title and table header prepended.
        """
        out = []
        if u.is_visual:
            lines = split_lines(u.text)
            title = lines[0] if lines else ""
            header = ""
            for i, line in enumerate(lines[1:], start=1):
                is_header = (i + 1 < len(lines) and any(ch.isdigit() for ch in lines[i + 1])
                             and not any(ch.isdigit() for ch in line))
                if is_header:
                    header = line
                    continue
                ctx = " ".join(x for x in (title, header, line) if x)
                shown = f"{title}: " + (f"{header} — {line}" if header and any(ch.isdigit() for ch in line) else line)
                out.append((shown, ctx, line))
            if u.caption:
                out += [(s, f"{title} {s}", s) for s in split_sentences(u.caption)]
            if not out and title:
                out.append((title, title, title))
        else:
            sents = split_sentences(u.text)
            for i, s in enumerate(sents):
                out.append((s, " ".join(sents[max(0, i - 2): i + 1]), s))
        return out

    def generate(self, question: str, items: list[RetrievedItem], profile: list[str]) -> GenerationResult:
        cands = []
        for rank, it in enumerate(items):
            u = it.unit
            for p, ctx, raw in self._pieces(u):
                if len(p) < 4:
                    continue
                own = self.scorer(question, raw)
                s = max(own, 0.85 * self.scorer(question, ctx)) * (1.0 - 0.04 * rank)
                cands.append((s, rank, p, u.evidence_id, own))
        cands.sort(key=lambda x: -x[0])
        chosen: list[tuple[float, int, str, str]] = []
        for s, rank, p, eid, own in cands:
            if s < self.min_score:
                break
            if chosen and own == 0.0:       # supporting sentences must share at least one question term
                continue
            if any(p.lower() in c[2].lower() or c[2].lower() in p.lower() for c in chosen):
                continue
            chosen.append((s, rank, p, eid))
            if len(chosen) >= self.max_sentences:
                break
        if not chosen:
            return GenerationResult(Status.ABSTAINED, [], "No passage in the course material matches the question.")
        # keep the strongest sentence first, others only if reasonably close to it
        best = chosen[0][0]
        chosen = [c for c in chosen if c[0] >= 0.6 * best]
        segs = [AnswerSegment(text=p + "." if p[-1:].isalnum() else p, citations=[eid])
                for _, _, p, eid in chosen]
        return GenerationResult(Status.ANSWERED, segs)


class LLMGenerator:
    def __init__(self, llm: LLM, vision: bool = False):
        self.llm, self.vision = llm, vision

    def generate(self, question: str, items: list[RetrievedItem], profile: list[str]) -> GenerationResult:
        units = [it.unit for it in items]
        images = [Path(u.frame_uri) for u in units if self.vision and u.is_visual and u.frame_uri
                  and Path(u.frame_uri).exists()][:4]
        raw = self.llm.complete(build_messages(question, units, profile), json_mode=True, images=images or None)
        try:
            data = parse_json(raw)
        except (ValueError, Exception) as e:  # noqa: BLE001
            log.warning("could not parse generator output: %s", e)
            return GenerationResult(Status.ABSTAINED, [], "Generator returned malformed output.", raw)
        valid = {u.evidence_id for u in units}
        segs = []
        for s in data.get("segments", []) or []:
            text = str(s.get("text", "")).strip()
            if not text:
                continue
            cites = [c for c in s.get("citations", []) or [] if c in valid]
            segs.append(AnswerSegment(text=text, citations=cites))
        try:
            status = Status(data.get("status", "answered"))
        except ValueError:
            status = Status.ANSWERED
        if not segs and status != Status.ABSTAINED:
            status = Status.ABSTAINED
        return GenerationResult(status, segs, data.get("abstain_reason"), raw)


def make_generator(cfg, llm: LLM | None, scorer):
    if cfg.get("generator.type", "extractive") == "llm":
        if llm is None:
            raise ValueError("generator.type=llm requires an llm section")
        return LLMGenerator(llm, vision=cfg.get("llm.vision", False))
    return ExtractiveGenerator(scorer, cfg.get("generator.max_sentences", 3), cfg.get("generator.min_sentence_score", 0.25))
