from tutor.eval.memory_tests import scenario_results


def test_memory_scenarios_all_pass():
    failed = [r for r in scenario_results() if not r["passed"]]
    assert not failed, failed
