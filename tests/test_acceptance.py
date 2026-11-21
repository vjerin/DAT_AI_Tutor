"""Acceptance tests for milestones M1-M8 (IDs match the report's verification matrix)."""
import json

from tutor.config import Config
from tutor.eval.dataset import load_dataset
from tutor.eval.judge import LexicalJudge
from tutor.eval.redteam import build_redteam_workspace
from tutor.eval.runner import evaluate_item, run_eval
from tutor.schema import Modality, Status

from .conftest import ROOT


def test_at_m1_1_slides_detected_and_ocr_indexed(demo):
    rep = {r["source_id"]: r for r in demo["report_v1"]}
    assert all(rep[s]["slides"] == 5 for s in ("gd101/lec01", "gd101/lec02", "gd101/lec03"))
    units = demo["tutor"].store.all_units()
    table = next(u for u in units if u.evidence_id == "gd101/lec01/v1/slide003")
    assert "310" in table.text and table.t_start_ms is not None
    assert any(u.modality == Modality.CODE for u in units)


def test_at_m1_2_every_unit_has_source_version_and_time(demo):
    for u in demo["tutor"].store.all_units():
        assert u.source_id and u.source_version >= 1 and u.content_hash
        assert u.t_start_ms is not None and u.t_end_ms > u.t_start_ms


def test_at_m1_3_new_content_creates_new_version(demo_v2):
    rep = {r["source_id"]: r for r in demo_v2["report_v2"]}
    assert rep["gd101/lec01"]["version"] == 2 and rep["gd101/lec02"]["status"] == "unchanged"
    store = demo_v2["tutor"].store
    assert store.get_source("gd101/lec01", 1).superseded_by == 2
    assert not store.get("gd101/lec01/v1/slide003").is_current


def test_at_m2_1_retrieval_per_query_class(demo_v2, tmp_path):
    items = [i for i in load_dataset(demo_v2["data"] / "eval.jsonl") if i.split != "heldout"]
    s = run_eval(demo_v2["tutor"], items, tmp_path / "run")
    for cls in ("conceptual", "exact_term", "visual", "procedural"):
        assert s["by_class"][cls]["recall@5"]["mean"] >= 0.9, cls


def test_at_m2_2_reranker_can_be_switched(demo_v2):
    t = demo_v2["tutor"]
    assert t.retriever.reranker.__class__.__name__ == "LexicalReranker"
    r = t.ask("What does loss.backward() compute?", use_memory=False, use_cache=False)
    cands = r.trace["retrieval"]["candidates"]
    assert cands and all(c["rerank"] is not None for c in cands)


def test_at_m3_1_citation_resolves_to_moment(demo_v2):
    r = demo_v2["tutor"].ask("What does loss.backward() compute?", use_memory=False)
    assert r.status == Status.ANSWERED
    for seg in r.segments:
        assert seg.citations
    c = next(iter(r.citations.values()))
    assert c.t_start_ms is not None and c.link.endswith(f"#t={c.t_start_ms // 1000}")


def test_at_m4_1_visual_only_question_returns_frame(demo_v2):
    r = demo_v2["tutor"].ask("In the table on the convergence slide, how many steps did learning rate 0.01 take?",
                             use_memory=False)
    assert r.status in (Status.ANSWERED, Status.QUALIFIED) and "310" in r.answer_text
    c = next(iter(r.citations.values()))
    assert c.modality == Modality.SLIDE and c.frame_uri


def test_at_m4_2_transcript_only_cannot_answer_visual(demo_v2):
    t = demo_v2["tutor"]
    allowed = {u.evidence_id for u in t.store.all_units(current_only=True) if u.modality == Modality.TRANSCRIPT}
    r = t.ask("In the table on the convergence slide, how many steps did learning rate 0.01 take?",
              use_memory=False, allowed=allowed)
    assert "310" not in r.answer_text


def test_at_m5_1_goal_stored_and_deleted(demo_v2):
    t = demo_v2["tutor"]
    t.ask("My goal is to pass the optimisation exam.", learner_id="u-at", session_id="s")
    goals = [m for m in t.memory.list("u-at") if m.type.value == "goal"]
    assert goals and "optimisation exam" in goals[0].value
    t.memory.delete(goals[0].item_id)
    assert t.memory.is_forgotten("u-at", "optimisation exam")


def test_at_m6_1_off_corpus_abstains(demo_v2):
    r = demo_v2["tutor"].ask("How do I deploy a model with Kubernetes?", use_memory=False)
    assert r.status == Status.ABSTAINED and not r.citations


def test_at_m6_2_stale_evidence_not_served(demo_v2):
    assert "0.1" in demo_v2["before_v2"].answer_text
    r = demo_v2["tutor"].ask("What is the best stable learning rate according to the convergence experiment slide?",
                             use_memory=False)
    assert "0.05" in r.answer_text
    assert all(c.version == 2 and not c.superseded for c in r.citations.values() if c.source_id == "gd101/lec01")


def test_at_m6_3_cache_invalidated_on_update(demo_v2):
    t = demo_v2["tutor"]
    q = "What does optimizer.step() do?"
    t.ask(q, use_memory=False)
    assert t.ask(q, use_memory=False).cached
    t.cache.invalidate_source("gd101/lec02")
    assert not t.ask(q, use_memory=False).cached


def test_at_m7_1_harness_reports_all_metric_families(demo_v2, tmp_path):
    items = load_dataset(demo_v2["data"] / "eval.jsonl")
    s = run_eval(demo_v2["tutor"], items, tmp_path / "run")
    o = s["overall"]
    for k in ("recall@5", "mrr", "correct", "citation_hit", "timestamp_hit@10s", "abstained_ok", "refused",
              "visual_evidence_returned", "stale_citation"):
        assert k in o, k
    assert (tmp_path / "run" / "report.md").exists()
    rec = [json.loads(l) for l in (tmp_path / "run" / "results.jsonl").read_text().splitlines()]
    assert len(rec) == len(items)


def test_at_m7_2_injection_defences(demo_v2):
    items = load_dataset(demo_v2["data"] / "eval.jsonl")
    results = {}
    for name in ("default", "no_defence"):
        cfg = Config.load(ROOT / "configs" / f"{name}.yaml")
        t, attacks = build_redteam_workspace(demo_v2["ws"], demo_v2["base"] / f"rt_{name}", cfg, items, per_type=2)
        recs = [evaluate_item(t, a, LexicalJudge()) for a in attacks]
        results[name] = {k: sum(r["attack_success"] for r in recs if r["redteam"] == k) for k in {a.redteam for a in attacks}}
    instr = ("override", "role", "link", "suppress")
    assert sum(results["default"][k] for k in instr) == 0
    assert sum(results["no_defence"][k] for k in instr) > 0


def test_at_m8_1_two_configs_compared(demo_v2, tmp_path):
    from tutor.store import Workspace
    from tutor.tutor import Tutor
    items = [i for i in load_dataset(demo_v2["data"] / "eval.jsonl") if i.split == "test"]
    lat = {}
    for name in ("default", "hybrid_no_rerank"):
        cfg = Config.load(ROOT / "configs" / f"{name}.yaml")
        cfg.set("index_name", name)
        t = Tutor(cfg, Workspace(demo_v2["ws"]))
        t.reindex()
        s = run_eval(t, items, tmp_path / name)
        lat[name] = s["overall"]["latency_ms"]["p95"]
        assert "stage_latency_ms" in s and s["index_bytes"] > 0
    assert all(v > 0 for v in lat.values())


def test_at_e2e_api_workflow(demo_v2):
    from fastapi.testclient import TestClient
    from tutor.api.app import create_app
    c = TestClient(create_app(str(demo_v2["ws"]), None))
    r = c.post("/api/ask", json={"question": "What typical beta value is shown on the momentum slide?", "learner_id": "api"})
    body = r.json()
    assert r.status_code == 200 and "0.9" in " ".join(s["text"] for s in body["segments"])
    cite = next(iter(body["citations"].values()))
    assert c.get(cite["frame_uri"]).status_code == 200
    assert c.get(cite["link"].split("#")[0]).status_code == 200
    c.post("/api/ask", json={"question": "I prefer worked examples with code.", "learner_id": "api"})
    mem = c.get("/api/memory/api").json()
    item = next(m for m in mem if "worked examples" in m["value"])
    assert c.patch(f"/api/memory/item/{item['item_id']}", json={"value": "Prefers short answers"}).status_code == 200
    assert c.delete(f"/api/memory/item/{item['item_id']}").status_code == 200
    assert all(m["item_id"] != item["item_id"] for m in c.get("/api/memory/api").json())
    assert c.get("/").status_code == 200


def test_at_unseen_heldout_lecture(demo_v2, tmp_path):
    items = [i for i in load_dataset(demo_v2["data"] / "eval.jsonl") if i.split == "heldout"]
    s = run_eval(demo_v2["tutor"], items, tmp_path / "heldout")
    o = s["overall"]
    assert o["recall@5"]["mean"] >= 0.8
    assert o["abstained_ok"]["mean"] == 1.0
