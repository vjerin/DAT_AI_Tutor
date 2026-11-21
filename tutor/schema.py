"""Data models shared across the package."""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Modality(str, Enum):
    TRANSCRIPT = "transcript"
    SLIDE = "slide"        # frame whose content is mostly slide text / diagram
    CODE = "code"          # frame showing code
    PDF_PAGE = "pdf_page"  # page of a slide deck / notes PDF


VISUAL_MODALITIES = {Modality.SLIDE, Modality.CODE, Modality.PDF_PAGE}


class Source(BaseModel):
    """One lecture (or document) at one content version."""
    source_id: str                     # e.g. "gd101/lec01"
    version: int                       # 1, 2, ...
    course: str
    title: str
    media_uri: Optional[str] = None    # local path or URL of the video
    slides_uri: Optional[str] = None
    transcript_uri: Optional[str] = None
    url: Optional[str] = None          # public page (e.g. OCW / YouTube) for deep links
    licence: Optional[str] = None
    content_hash: str
    ingested_at: float
    is_current: bool = True
    superseded_by: Optional[int] = None

    @property
    def key(self) -> str:
        return f"{self.source_id}/v{self.version}"


class EvidenceUnit(BaseModel):
    evidence_id: str                   # "gd101/lec01/v2/seg0003"
    source_id: str
    source_version: int
    course: str
    lecture_title: str
    modality: Modality
    text: str                          # transcript words / OCR text / caption
    t_start_ms: Optional[int] = None
    t_end_ms: Optional[int] = None
    slide_no: Optional[int] = None
    frame_uri: Optional[str] = None    # image file for visual units
    caption: Optional[str] = None      # VLM description (visual units)
    concepts: list[str] = Field(default_factory=list)
    content_hash: str = ""
    is_current: bool = True
    flags: list[str] = Field(default_factory=list)   # e.g. ["suspected_injection"]

    @property
    def is_visual(self) -> bool:
        return self.modality in VISUAL_MODALITIES

    def search_text(self) -> str:
        """Text used by the sparse/dense indexes."""
        parts = [self.text]
        if self.caption:
            parts.append(self.caption)
        return "\n".join(p for p in parts if p)


class RetrievedItem(BaseModel):
    unit: EvidenceUnit
    fused_score: float = 0.0
    rerank_score: Optional[float] = None
    channel_ranks: dict[str, int] = Field(default_factory=dict)
    channel_scores: dict[str, float] = Field(default_factory=dict)

    @property
    def score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.fused_score


class Citation(BaseModel):
    evidence_id: str
    source_id: str
    version: int
    lecture_title: str
    modality: Modality
    t_start_ms: Optional[int] = None
    t_end_ms: Optional[int] = None
    slide_no: Optional[int] = None
    frame_uri: Optional[str] = None
    link: Optional[str] = None
    label: str = ""
    superseded: bool = False
    excerpt: str = ""


class AnswerSegment(BaseModel):
    text: str
    citations: list[str] = Field(default_factory=list)
    supported: Optional[bool] = None
    support_score: Optional[float] = None


class Status(str, Enum):
    ANSWERED = "answered"
    QUALIFIED = "qualified"
    ABSTAINED = "abstained"


class TutorResponse(BaseModel):
    query: str
    rewritten_query: str
    query_class: str
    status: Status
    segments: list[AnswerSegment] = Field(default_factory=list)
    citations: dict[str, Citation] = Field(default_factory=dict)
    abstain_reason: Optional[str] = None
    suggestions: list[Citation] = Field(default_factory=list)  # nearest moments when abstaining
    confidence: float = 0.0
    learner_profile_used: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    trace: dict[str, Any] = Field(default_factory=dict)
    cached: bool = False

    @property
    def answer_text(self) -> str:
        return " ".join(s.text for s in self.segments)


class MemoryType(str, Enum):
    GOAL = "goal"
    MASTERY = "mastery"
    MISCONCEPTION = "misconception"
    PREFERENCE = "preference"


class MemoryItem(BaseModel):
    item_id: str
    learner_id: str
    type: MemoryType
    key: str                      # concept name / goal text / preference name
    value: str                    # human-readable statement
    confidence: float = 0.5
    observations: int = 1
    mastery_p: Optional[float] = None
    provenance: list[dict[str, Any]] = Field(default_factory=list)
    created_at: float
    updated_at: float
    expires_at: Optional[float] = None
    status: str = "active"        # active | pending | expired


class MemoryOp(BaseModel):
    op: str                       # ADD | UPDATE | DELETE | OBSERVE | NOOP
    type: Optional[MemoryType] = None
    key: str = ""
    value: str = ""
    confidence: float = 0.5
    explicit: bool = False        # learner stated it directly
    correct: Optional[bool] = None  # for mastery observations
    item_id: Optional[str] = None
    source_turn: Optional[str] = None
