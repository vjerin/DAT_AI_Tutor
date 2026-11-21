"""HTTP API and static client.

Run with: python -m tutor.cli serve --workspace workspace
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from ..tutor import Tutor

CLIENT = Path(__file__).resolve().parent.parent / "client" / "index.html"


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=4000, pattern=r"\S")
    learner_id: str = "demo-learner"
    session_id: str = "default"
    use_memory: bool = True


class CorrectIn(BaseModel):
    value: str


class IngestIn(BaseModel):
    manifest: str
    force: bool = False


def create_app(workspace: str, config: Optional[str] = None) -> FastAPI:
    tutor = Tutor.open(workspace, config)
    app = FastAPI(title="Grounded multimodal AI tutor", version="0.1.0")
    app.state.tutor = tutor

    @app.get("/", response_class=HTMLResponse)
    def index():
        return CLIENT.read_text()

    @app.post("/api/ask")
    def ask(body: AskIn):
        r = tutor.ask(body.question, body.learner_id, body.session_id, use_memory=body.use_memory)
        return r.model_dump(mode="json")

    @app.get("/api/sources")
    def sources():
        return [s.model_dump() for s in tutor.store.list_sources()]

    @app.get("/api/evidence/{evidence_id:path}")
    def evidence(evidence_id: str):
        u = tutor.store.get(evidence_id)
        if not u:
            raise HTTPException(404, "unknown evidence id")
        return u.model_dump(mode="json")

    @app.get("/frames/{evidence_id:path}")
    def frame(evidence_id: str):
        u = tutor.store.get(evidence_id)
        if not u or not u.frame_uri or not Path(u.frame_uri).exists():
            raise HTTPException(404, "no frame")
        return FileResponse(u.frame_uri)

    @app.get("/media/{course}/{lecture}/v{version}")
    def media(course: str, lecture: str, version: int):
        s = tutor.store.get_source(f"{course}/{lecture}", version)
        if not s or not s.media_uri or not Path(s.media_uri).exists():
            raise HTTPException(404, "no media")
        return FileResponse(s.media_uri, media_type="video/mp4")

    @app.get("/api/memory/{learner_id}")
    def memory(learner_id: str):
        return [m.model_dump(mode="json") for m in tutor.memory.list(learner_id)]

    @app.get("/api/memory/{learner_id}/audit")
    def audit(learner_id: str):
        return tutor.memory.durable.audit_log(learner_id)

    @app.patch("/api/memory/item/{item_id}")
    def correct(item_id: str, body: CorrectIn):
        try:
            return tutor.memory.correct(item_id, body.value).model_dump(mode="json")
        except KeyError:
            raise HTTPException(404, "unknown memory item")

    @app.delete("/api/memory/item/{item_id}")
    def delete(item_id: str):
        if not tutor.memory.delete(item_id):
            raise HTTPException(404, "unknown memory item")
        return {"deleted": item_id}

    @app.delete("/api/memory/{learner_id}")
    def delete_all(learner_id: str):
        return {"deleted": tutor.memory.delete_learner(learner_id)}

    @app.post("/api/ingest")
    def ingest(body: IngestIn):
        return tutor.ingest(body.manifest, force=body.force)

    @app.get("/api/health")
    def health():
        return {"ok": True, "evidence": tutor.store.count(), "config": tutor.cfg.get("name")}

    return app


def _default_app() -> FastAPI:
    return create_app(os.environ.get("TUTOR_WORKSPACE", "workspace"), os.environ.get("TUTOR_CONFIG"))


app = None
if os.environ.get("TUTOR_WORKSPACE"):
    app = _default_app()
