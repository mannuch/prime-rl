"""Temporal GRPO with continuous stage rewards: TD credit over stages."""

import itertools

import pytest

from prime_rl.orchestrator.algo import temporal_grpo as binary
from prime_rl.orchestrator.algo.temporal_grpo import StageRecord
from prime_rl.orchestrator.algo.temporal_grpo_continuous import stage_advantages


def spans(values):
    """Collapse per-token advantages into (value, run_length) pairs."""
    out = []
    for v in values:
        if out and out[-1][0] == pytest.approx(v):
            out[-1][1] += 1
        else:
            out.append([v, 1])
    return [(round(v, 4), n) for v, n in out]


def stage_values(record, tokens):
    """One advantage per entered stage: the value at each stage's first token."""
    return [tokens[start] for start in (0, *record.boundaries) if start < len(tokens)]


# Three intermediate stages (near, grasp, move) + final (place = task reward); 10 tokens per stage.
LATE_FAILURE = [
    StageRecord([10, 20, 30], 40, 1.0),  # full success
    StageRecord([10, 20], 40, 0.0),  # fails during "move"
    StageRecord([], 40, 0.0),  # fails during "near"
    StageRecord([10, 20, 30], 40, 0.0),  # fails only at the final stage
]


def test_binary_stages_reproduce_temporal_grpo():
    # Every group of 4 over the binary outcomes of 2 intermediate stages + a task reward
    # (all-fail, all-pass, single-entrant and mixed groups included): per-stage
    # normalization cancels the (1 + V[k+1]) factor, leaving the paper's rule.
    states = [([], 0.0), ([10], 0.0), ([10, 20], 0.0), ([10, 20], 1.0)]
    for group in itertools.product(states, repeat=4):
        records = [StageRecord(list(boundaries), 30, reward) for boundaries, reward in group]
        expected = binary.stage_advantages(records, 2, std_normalize=True)
        for actual, paper in zip(stage_advantages(records, 2), expected, strict=True):
            assert actual == pytest.approx(paper, abs=1e-4)


def test_late_failure_keeps_early_credit():
    adv = stage_advantages(LATE_FAILURE, 3)
    assert spans(adv[0]) == [(0.5773, 10), (0.0, 10), (0.7071, 10), (1.0, 10)]
    assert spans(adv[1]) == [(0.5773, 10), (0.0, 10), (-1.4142, 20)]
    assert spans(adv[2]) == [(-1.732, 40)]
    # Vanilla GRPO would give this rollout -0.25 on every token.
    assert spans(adv[3]) == [(0.5773, 10), (0.0, 10), (0.7071, 10), (-1.0, 10)]


def test_all_fail_group_still_has_signal():
    # Nobody earns task reward, so the downstream group value is 0; the binary
    # stage reward keeps stage 1 separating them.
    records = [StageRecord([5], 10, 0.0), StageRecord([], 10, 0.0)]
    adv = stage_advantages(records, 1)
    assert spans(adv[0]) == [(1.0, 5), (0.0, 5)]
    assert spans(adv[1]) == [(-1.0, 10)]


def test_continuous_stage_rewards_conserve_credit():
    # Stage 1 pays a quality score; the task reward is paid where each rollout ended.
    records = [
        StageRecord([5], 10, 1.0, [0.7]),
        StageRecord([5], 10, 0.0, [0.3]),
        StageRecord([], 10, 0.0, [0.1]),
    ]
    adv = stage_advantages(records, 1, std_normalize=False)
    assert stage_values(records[0], adv[0]) == pytest.approx([0.5, 0.5])
    assert stage_values(records[1], adv[1]) == pytest.approx([0.1, -0.5])
    assert stage_values(records[2], adv[2]) == pytest.approx([-0.6])
    # Unnormalized, a rollout's stage advantages sum to its return minus the group mean (0.7).
    for record, tokens, total in zip(records, adv, [1.7, 0.3, 0.1], strict=True):
        assert sum(stage_values(record, tokens)) == pytest.approx(total - 0.7)


def test_stage_lambda_one_is_return_to_go():
    adv = stage_advantages(LATE_FAILURE, 3, std_normalize=False, stage_lambda=1.0)
    assert spans(adv[0]) == [(1.75, 10), (1.0, 20), (0.5, 10)]
    # Its return-to-go from "grasp" on equals the group's, so those stages are neutral.
    assert spans(adv[3]) == [(0.75, 10), (0.0, 20), (-0.5, 10)]
