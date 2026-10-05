"""Stage-conditioned credit (Temporal GRPO, arXiv:2608.13026)."""

import pytest

from prime_rl.orchestrator.algo.temporal_grpo import StageRecord, parse_stages, stage_advantages


def spans(values):
    """Collapse per-token advantages into (value, run_length) pairs."""
    out = []
    for v in values:
        if out and out[-1][0] == pytest.approx(v):
            out[-1][1] += 1
        else:
            out.append([v, 1])
    return [(round(v, 4), n) for v, n in out]


def test_parse_nodes_and_prerequisites():
    # Sampled assistant nodes at trace.nodes positions 1, 3, 5, 7 (prompts/tool results in between).
    turns = [(1, 10), (3, 20), (5, 30), (7, 40)]
    raw = [{"name": "a", "node": 3}, {"name": "b", "node": 1}, {"name": "c", "node": None}, {"name": "d", "node": 7}]
    # "b" cannot end before its prerequisite; "d" is unreached because "c" never completed.
    assert parse_stages(raw, turns, num_nodes=9) == (["a", "b", "c", "d"], [30, 30])


def test_parse_rejects_bad_turn():
    assert parse_stages([{"name": "a", "node": 9}], [(1, 5)], num_nodes=3) is None
    assert parse_stages(None, [(1, 5)], num_nodes=3) is None


def test_parse_token_offsets():
    raw = [{"name": "a", "token": 12}, {"name": "b", "token": 999}]
    assert parse_stages(raw, [(1, 50)], num_nodes=2) == (["a", "b"], [12, 50])


def test_late_failure_keeps_early_credit():
    # Three intermediate stages (near, grasp, move) + final (place = task reward); 10 tokens per stage.
    records = [
        StageRecord([10, 20, 30], 40, 1.0),  # full success
        StageRecord([10, 20], 40, 0.0),  # fails during "move"
        StageRecord([], 40, 0.0),  # fails during "near"
        StageRecord([10, 20, 30], 40, 0.0),  # fails only at the final stage
    ]
    adv = stage_advantages(records, 3)
    third = round(1 / 3, 4)
    assert spans(adv[0]) == [(0.25, 10), (0.0, 10), (third, 10), (0.5, 10)]
    assert spans(adv[1]) == [(0.25, 10), (0.0, 10), (round(-2 / 3, 4), 20)]
    assert spans(adv[2]) == [(-0.75, 40)]
    # Vanilla GRPO would give this rollout -0.25 on every token.
    assert spans(adv[3]) == [(0.25, 10), (0.0, 10), (third, 10), (-0.5, 10)]


def test_all_fail_group_still_has_signal():
    # Final reward is 0 everywhere, so plain GRPO sees nothing; stage 1 still separates them.
    records = [StageRecord([5], 10, 0.0), StageRecord([], 10, 0.0)]
    adv = stage_advantages(records, 1)
    assert spans(adv[0]) == [(0.5, 5), (0.0, 5)]
    assert spans(adv[1]) == [(-0.5, 10)]


def test_no_intermediate_stages_is_grpo():
    records = [StageRecord([], 4, 1.0), StageRecord([], 4, 0.0), StageRecord([], 4, 0.0)]
    adv = stage_advantages(records, 0)
    assert spans(adv[0]) == [(round(2 / 3, 4), 4)]
    assert spans(adv[1]) == [(round(-1 / 3, 4), 4)]


def test_trajectory_blend():
    records = [StageRecord([5], 10, 0.0), StageRecord([], 10, 0.0)]
    adv = stage_advantages(records, 1, trajectory_weight=0.5)
    assert spans(adv[0]) == [(0.25, 5), (0.0, 5)]
