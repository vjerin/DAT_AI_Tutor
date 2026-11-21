import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="session")
def demo(tmp_path_factory):
    """Demo course with lecture 1 at v1 and v2 ingested, plus lecture 2."""
    from make_demo_course import build
    from tutor.tutor import Tutor
    base = tmp_path_factory.mktemp("demo")
    data = build(base / "data")
    tutor = Tutor.open(base / "ws", ROOT / "configs" / "default.yaml")
    r1 = tutor.ingest(data / "manifest_v1.yaml")
    return {"tutor": tutor, "data": data, "ws": base / "ws", "base": base, "report_v1": r1}


@pytest.fixture(scope="session")
def demo_v2(demo):
    t = demo["tutor"]
    before = t.ask("What is the best stable learning rate according to the convergence experiment slide?",
                   use_memory=False)
    report = t.ingest(demo["data"] / "manifest_v2.yaml")
    return {**demo, "before_v2": before, "report_v2": report}
