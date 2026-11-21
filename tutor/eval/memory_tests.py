"""Scripted learner scenarios for the memory service."""
from __future__ import annotations

import tempfile
from pathlib import Path

from ..config import Config
from ..schema import MemoryType
from ..memory.service import MemoryService


class FakeClock:
    def __init__(self, t: float = 1_700_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance_days(self, d: float) -> None:
        self.t += d * 86400


def _svc(cfg: Config, tmp: Path, clock: FakeClock) -> MemoryService:
    return MemoryService(cfg.section("memory"), str(tmp / "mem.db"), clock=clock)


def scenario_results(cfg: Config | None = None) -> list[dict]:
    cfg = cfg or Config()
    results = []

    def check(name, ok, detail=""):
        results.append({"scenario": name, "passed": bool(ok), "detail": detail})

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        clock = FakeClock()
        m = _svc(cfg, tmp, clock)
        L, S = "persona-asha", "s1"
        concepts = ["learning rate", "gradient descent"]

        # 1. explicit goal is stored and shows up in the profile
        m.record_turn(L, S, "user", "My goal is to pass the mid-term on optimisation.")
        m.observe(L, S, "My goal is to pass the mid-term on optimisation.", concepts, "t1", ["e1"])
        goals = [i for i in m.list(L) if i.type == MemoryType.GOAL]
        check("goal_stored", len(goals) == 1 and goals[0].status == "active", goals[0].value if goals else "none")
        check("goal_in_profile", any("mid-term" in p for p in m.profile_lines(m.profile(L, concepts))))

        # 2. implicit / inferred goal is NOT stored (goals require an explicit statement)
        m.observe(L, S, "Exams are coming up soon I guess.", concepts, "t2", [])
        check("no_inferred_goal", len([i for i in m.list(L) if i.type == MemoryType.GOAL]) == 1)

        # 3. misconception needs two observations
        m.observe(L, S, "I thought a bigger learning rate always converges faster.", concepts, "t3", ["e2"])
        mc = [i for i in m.list(L) if i.type == MemoryType.MISCONCEPTION]
        check("misconception_pending_after_one", len(mc) == 1 and mc[0].status == "pending")
        check("pending_not_in_profile", not any("May believe" in p for p in m.profile_lines(m.profile(L, concepts))))
        m.observe(L, S, "I thought a large learning rate is always better.", concepts, "t4", ["e3"])
        mc = [i for i in m.list(L) if i.type == MemoryType.MISCONCEPTION]
        check("misconception_active_after_two", len(mc) == 1 and mc[0].status == "active" and mc[0].observations == 2)

        # 4. sensitive content is never stored
        m.observe(L, S, "I prefer short answers because of my anxiety diagnosis.", concepts, "t5", [])
        check("sensitive_rejected", not m.durable.raw_contains("anxiety"))

        # 5. preferences
        m.observe(L, S, "I prefer worked examples with code.", concepts, "t6", [])
        prefs = [i for i in m.list(L) if i.type == MemoryType.PREFERENCE]
        check("preference_stored", any("worked examples" in p.value for p in prefs))

        # 6. mastery via BKT moves in the right direction
        m.observe(L, S, "I'm confused about this.", concepts, "t7", [])
        low = next((i.mastery_p for i in m.list(L) if i.type == MemoryType.MASTERY and i.key == "learning rate"), None)
        m.observe(L, S, "Got it, that makes sense now.", concepts, "t8", [])
        high = next((i.mastery_p for i in m.list(L) if i.type == MemoryType.MASTERY and i.key == "learning rate"), None)
        check("mastery_updates", low is not None and high is not None and high > low, f"{low} -> {high}")

        # 7. correction replaces the value and scrubs the old one from session state
        g = [i for i in m.list(L) if i.type == MemoryType.GOAL][0]
        m.correct(g.item_id, "Goal: score above 80 in the final exam")
        check("correction_applied", m.durable.get(g.item_id).value == "Goal: score above 80 in the final exam")
        check("old_value_scrubbed", not m.ephemeral.contains(L, "mid-term on optimisation"))

        # 8. spoken correction updates the existing goal instead of adding a second one
        m.observe(L, S, "Actually my goal is to understand backpropagation.", concepts, "t9", [])
        goals = [i for i in m.list(L) if i.type == MemoryType.GOAL]
        check("spoken_correction", len(goals) == 1 and "backpropagation" in goals[0].value)

        # 9. deletion leaves no trace in durable store, audit content or session text
        p = [i for i in m.list(L) if i.type == MemoryType.PREFERENCE][0]
        m.record_turn(L, S, "user", "Remember: worked examples with code please")
        deleted_ids = []
        m.on_delete(deleted_ids.append)
        m.delete(p.item_id)
        check("deleted_unrecoverable", m.is_forgotten(L, "worked examples"), "durable + audit + session")
        check("delete_notifies_cache", p.item_id in deleted_ids)

        # 10. spoken deletion request
        m.observe(L, S, "Please forget my goal.", concepts, "t10", [])
        check("spoken_delete", not [i for i in m.list(L) if i.type == MemoryType.GOAL])

        # 11. expiry: misconceptions expire if not seen again
        clock.advance_days(cfg.get("memory.ttl_days.misconception", 45) + 1)
        check("misconception_expires", not [i for i in m.list(L) if i.type == MemoryType.MISCONCEPTION])

        # 12. ephemeral session state expires
        clock.advance_days(2)
        check("session_expires", m.history(L, S) == [])

        # 13. forget everything
        m.observe(L, S, "My goal is to learn convex optimisation.", concepts, "t11", [])
        m.delete_learner(L)
        check("delete_learner", m.list(L) == [] and not m.durable.raw_contains("convex"))
    return results
