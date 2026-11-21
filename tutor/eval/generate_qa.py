"""Draft candidate questions for the gold set from evidence units.

Candidates must be checked by hand before they go into a test split.
"""
from __future__ import annotations

from pathlib import Path

from ..generate.llm import parse_json
from ..schema import EvidenceUnit, Modality
from ..store import EvidenceStore
from ..text import content_terms, numbers
from .dataset import EvalItem

QA_PROMPT = """Write {n} question(s) a student might ask that are answered ONLY by this lecture evidence.
Each answer must be short (a phrase, value or one sentence) and copied/paraphrased from the evidence.
{visual_hint}
Return JSON: {{"items": [{{"question": "...", "answer": "...", "class": "conceptual|exact_term|procedural|visual|temporal"}}]}}

Evidence ({modality}, {lecture}):
{text}"""


def generate_candidates(llm, unit: EvidenceUnit, n: int = 2) -> list[EvalItem]:
    visual = unit.modality != Modality.TRANSCRIPT
    hint = ("The question must be about something visible on the slide (a value, label, diagram or code line) "
            "and should say 'on the slide' or 'in the code shown'.") if visual else ""
    raw = llm.complete([{"role": "user", "content": QA_PROMPT.format(
        n=n, visual_hint=hint, modality=unit.modality.value, lecture=unit.lecture_title, text=unit.search_text()[:2500])}],
        json_mode=True, images=[Path(unit.frame_uri)] if visual and unit.frame_uri else None)
    try:
        data = parse_json(raw)
    except Exception:  # noqa: BLE001
        return []
    out = []
    for k, it in enumerate(data.get("items", [])):
        out.append(EvalItem(qid=f"gen-{unit.evidence_id.replace('/', '-')}-{k}", split="candidate",
                            cls=it.get("class", "visual" if visual else "conceptual"), question=it["question"],
                            gold_answer=it.get("answer"), gold_evidence=[unit.evidence_id], gold_source=unit.source_id,
                            gold_window_ms=[unit.t_start_ms, unit.t_end_ms] if unit.t_start_ms is not None else None))
    return out


def is_visual_only(store: EvidenceStore, unit: EvidenceUnit, answer: str, threshold: float = 0.6) -> bool:
    """True if the answer cannot be found in speech overlapping the slide's time span."""
    speech = " ".join(u.text for u in store.neighbours(unit, Modality.TRANSCRIPT))
    if not speech:
        return True
    a_nums = numbers(answer)
    if a_nums and a_nums <= numbers(speech):
        return False
    terms = content_terms(answer)
    if not terms:
        return True
    return len(terms & content_terms(speech)) / len(terms) < threshold
