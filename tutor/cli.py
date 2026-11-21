"""Command line interface. Run `python -m tutor.cli --help` for the list of commands."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _tutor(a):
    from .tutor import Tutor
    return Tutor.open(a.workspace, a.config, a.set)


def cmd_ingest(a):
    t = _tutor(a)
    for r in t.ingest(a.manifest, force=a.force):
        print(json.dumps(r))
    print(json.dumps({"evidence": t.store.count()}))


def cmd_index(a):
    print(json.dumps(_tutor(a).reindex()))


def cmd_ask(a):
    t = _tutor(a)
    r = t.ask(" ".join(a.question), learner_id=a.learner, session_id=a.session)
    if a.json:
        print(r.model_dump_json(indent=2))
        return
    print(f"[{r.status.value} | class={r.query_class} | conf={r.confidence:.2f} | {r.timings_ms.get('total')} ms]")
    if r.abstain_reason:
        print("  " + r.abstain_reason)
        for s in r.suggestions:
            print(f"  nearest: {s.label}  {s.link or ''}")
    for s in r.segments:
        labels = "; ".join(r.citations[c].label for c in s.citations)
        print(f"- {s.text}  [{labels}]")
    for c in r.citations.values():
        print(f"  {c.evidence_id} -> {c.link}")


def cmd_serve(a):
    import uvicorn
    from .api.app import create_app
    uvicorn.run(create_app(a.workspace, a.config), host=a.host, port=a.port)


def cmd_eval(a):
    from .eval.dataset import load_dataset
    from .eval.runner import run_eval
    t = _tutor(a)
    items = load_dataset(a.dataset, split=a.split)
    s = run_eval(t, items, a.out, config_path=a.config)
    print(Path(a.out, "report.md").read_text())
    return s


def cmd_sweep(a):
    """Run the same dataset under several configs (M8). Each config gets its own index."""
    from .config import Config
    from .eval.dataset import load_dataset
    from .eval.runner import run_eval
    from .store import Workspace
    from .tutor import Tutor
    items = load_dataset(a.dataset, split=a.split)
    rows = []
    for cfg_path in a.configs:
        cfg = Config.load(cfg_path, a.set)
        name = cfg.get("name")
        cfg.set("index_name", name)
        if a.manifests:      # configs that change ingestion (e.g. chunking) get a fresh workspace
            t = Tutor(cfg, Workspace(Path(a.out) / name / "workspace"))
            for m in a.manifests:
                t.ingest(m)
        else:
            t = Tutor(cfg, Workspace(a.workspace))
            t.reindex()
        s = run_eval(t, items, Path(a.out) / name, config_path=cfg_path)
        o = s["overall"]
        g = lambda k: o.get(k, {}).get("mean", float("nan"))  # noqa: E731
        rows.append((name, g("recall@5"), g("correct"), g("citation_hit"), g("timestamp_hit@10s"), g("abstained_ok"),
                     g("refused"), o["latency_ms"]["p50"], o["latency_ms"]["p95"]))
    lines = ["| config | recall@5 | correct | citation_hit | ts_hit@10s | abstain_ok | refused | p50 ms | p95 ms |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r[0]} | " + " | ".join(f"{x:.3f}" for x in r[1:7]) + f" | {r[7]} | {r[8]} |")
    table = "\n".join(lines)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    Path(a.out, "sweep.md").write_text("# Configuration sweep\n\n" + table + "\n")
    print(table)


def cmd_compare(a):
    from .eval.report import compare_runs
    print(compare_runs(a.run_a, a.run_b))


def cmd_redteam(a):
    from .config import Config
    from .eval.dataset import load_dataset
    from .eval.redteam import build_redteam_workspace
    from .eval.runner import run_eval
    items = load_dataset(a.dataset)
    cfg = Config.load(a.config, a.set)
    t, attacks = build_redteam_workspace(a.workspace, a.redteam_workspace, cfg, items, per_type=a.per_type)
    run_eval(t, attacks, a.out, config_path=a.config)
    print(Path(a.out, "report.md").read_text())


def cmd_memtest(a):
    from .config import Config
    from .eval.memory_tests import scenario_results
    res = scenario_results(Config.load(a.config, a.set))
    for r in res:
        print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['scenario']}  {r['detail']}")
    passed = sum(r["passed"] for r in res)
    print(f"\n{passed}/{len(res)} memory scenarios passed")
    if passed != len(res):
        sys.exit(1)


def cmd_train_classifier(a):
    from .eval.dataset import load_dataset
    from .train.classifier import train_classifier
    r = train_classifier(load_dataset(a.dataset), a.out)
    print(r.pop("report"))
    print(json.dumps(r, indent=2))


def cmd_calibrate(a):
    from .eval.dataset import load_dataset
    from .train.calibrator import fit_calibrator
    t = _tutor(a)
    print(json.dumps(fit_calibrator(t, load_dataset(a.dataset, split=a.split), a.out), indent=2))


def cmd_mine(a):
    from .eval.dataset import load_dataset
    from .train.mine import mine_triplets, write_jsonl
    t = _tutor(a)
    rows = mine_triplets(t.store, load_dataset(a.dataset, split=a.split))
    write_jsonl(rows, a.out)
    print(f"wrote {len(rows)} triplets to {a.out}")


def cmd_demo(a):
    """Generate the synthetic demo course, ingest v1, evaluate, ingest v2, evaluate again."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_demo_course import build
    from .eval.dataset import load_dataset
    from .eval.runner import run_eval
    from .tutor import Tutor
    data = Path(a.data)
    build(data)
    t = Tutor.open(a.workspace, a.config or str(ROOT / "configs" / "default.yaml"), a.set)
    print(json.dumps(t.ingest(data / "manifest_v1.yaml"), indent=1))
    print(json.dumps(t.ingest(data / "manifest_v2.yaml"), indent=1))
    items = load_dataset(data / "eval.jsonl")
    run_eval(t, items, Path(a.workspace) / "runs" / "demo")
    print(Path(a.workspace, "runs", "demo", "report.md").read_text())


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="tutor")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, ws=True):
        if ws:
            p.add_argument("--workspace", default="workspace")
        p.add_argument("--config", default=None)
        p.add_argument("--set", action="append", default=[], help="override, e.g. --set retrieval.reranker=none")
        return p

    p = common(sub.add_parser("ingest")); p.add_argument("--manifest", required=True); p.add_argument("--force", action="store_true"); p.set_defaults(f=cmd_ingest)
    p = common(sub.add_parser("index")); p.set_defaults(f=cmd_index)
    p = common(sub.add_parser("ask")); p.add_argument("question", nargs="+"); p.add_argument("--learner", default="cli")
    p.add_argument("--session", default="cli"); p.add_argument("--json", action="store_true"); p.set_defaults(f=cmd_ask)
    p = common(sub.add_parser("serve")); p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8080); p.set_defaults(f=cmd_serve)
    p = common(sub.add_parser("eval")); p.add_argument("--dataset", required=True); p.add_argument("--split", default=None)
    p.add_argument("--out", default="runs/latest"); p.set_defaults(f=cmd_eval)
    p = common(sub.add_parser("sweep")); p.add_argument("--dataset", required=True); p.add_argument("--split", default=None)
    p.add_argument("--configs", nargs="+", required=True); p.add_argument("--out", default="runs/sweep")
    p.add_argument("--manifests", nargs="*", default=None, help="re-ingest these manifests per config (for ingest-level settings)")
    p.set_defaults(f=cmd_sweep)
    p = sub.add_parser("compare"); p.add_argument("run_a"); p.add_argument("run_b"); p.set_defaults(f=cmd_compare)
    p = common(sub.add_parser("redteam")); p.add_argument("--dataset", required=True); p.add_argument("--out", default="runs/redteam")
    p.add_argument("--redteam-workspace", default="workspace_redteam"); p.add_argument("--per-type", type=int, default=3); p.set_defaults(f=cmd_redteam)
    p = common(sub.add_parser("memtest"), ws=False); p.set_defaults(f=cmd_memtest)
    p = sub.add_parser("train-classifier"); p.add_argument("--dataset", required=True); p.add_argument("--out", required=True); p.set_defaults(f=cmd_train_classifier)
    p = common(sub.add_parser("calibrate")); p.add_argument("--dataset", required=True); p.add_argument("--split", default="dev")
    p.add_argument("--out", required=True); p.set_defaults(f=cmd_calibrate)
    p = common(sub.add_parser("mine")); p.add_argument("--dataset", required=True); p.add_argument("--split", default=None)
    p.add_argument("--out", required=True); p.set_defaults(f=cmd_mine)
    p = common(sub.add_parser("demo")); p.add_argument("--data", default="data/demo_course"); p.set_defaults(f=cmd_demo)
    a = ap.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
