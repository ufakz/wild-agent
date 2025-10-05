"""Unit tests for paired explore→harvest iteration counting."""

from src.agents.graph import route_by_phase
from src.agents.harvester import cycle_update


def _state(**kwargs) -> dict:
    base = {
        "phase": "plan",
        "iteration": 0,
        "max_iterations": 10,
        "target_count": 100,
        "samples": [],
        "fresh_explore": False,
    }
    base.update(kwargs)
    return base


def test_audit_allowed_at_cycle_limit():
    state = _state(phase="audit", iteration=10)
    assert route_by_phase(state) == "auditor"


def test_plan_allowed_at_cycle_limit():
    state = _state(phase="plan", iteration=10)
    assert route_by_phase(state) == "planner"


def test_explore_blocked_at_cycle_limit():
    state = _state(phase="explore", iteration=10)
    assert route_by_phase(state) == "end"


def test_harvest_blocked_at_limit_when_fresh_explore():
    state = _state(phase="harvest", iteration=10, fresh_explore=True)
    assert route_by_phase(state) == "end"


def test_harvest_allowed_at_limit_when_draining_urls():
    state = _state(
        phase="harvest",
        iteration=10,
        fresh_explore=False,
        pending_urls=["https://example.com/page2"],
    )
    assert route_by_phase(state) == "harvester"


def test_harvest_allowed_below_limit():
    state = _state(phase="harvest", iteration=5, fresh_explore=True)
    assert route_by_phase(state) == "harvester"


def test_ends_when_target_count_reached():
    state = _state(phase="harvest", iteration=0, samples=[{}] * 100, target_count=100)
    assert route_by_phase(state) == "end"


def test_cycle_update_increments_and_clears_fresh_explore():
    state = _state(iteration=3, fresh_explore=True)
    assert cycle_update(state) == {"iteration": 4, "fresh_explore": False}


def test_cycle_update_noop_without_fresh_explore():
    state = _state(iteration=3, fresh_explore=False)
    assert cycle_update(state) == {}
