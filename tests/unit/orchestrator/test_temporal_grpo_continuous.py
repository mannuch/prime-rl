"""Temporal GRPO with continuous stage rewards: TD credit over stages."""

import itertools
import math

import pytest

from prime_rl.orchestrator.algo import temporal_grpo as binary
from prime_rl.orchestrator.algo.temporal_grpo import StageRecord
from prime_rl.orchestrator.algo.temporal_grpo_continuous import (
    Moments,
    fit_handoff_values,
    score_pools,
    shrunk,
    stage_advantages,
)


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
        for actual, paper in zip(stage_advantages(records, 2, std_normalize=True), expected, strict=True):
            assert actual == pytest.approx(paper, abs=1e-4)


def test_late_failure_keeps_early_credit():
    adv = stage_advantages(LATE_FAILURE, 3)
    assert spans(adv[0]) == [(0.75, 10), (0.0, 10), (0.5, 20)]
    assert spans(adv[1]) == [(0.75, 10), (0.0, 10), (-1.0, 20)]
    assert spans(adv[2]) == [(-2.25, 40)]
    # Vanilla GRPO would give this rollout -0.25 on every token.
    assert spans(adv[3]) == [(0.75, 10), (0.0, 10), (0.5, 10), (-0.5, 10)]


def test_all_fail_group_still_has_signal():
    # Nobody earns task reward, so the downstream group value is 0; the binary
    # stage reward keeps stage 1 separating them, normalized or not.
    records = [StageRecord([5], 10, 0.0), StageRecord([], 10, 0.0)]
    adv = stage_advantages(records, 1)
    assert spans(adv[0]) == [(0.5, 5), (0.0, 5)]
    assert spans(adv[1]) == [(-0.5, 10)]
    normalized = stage_advantages(records, 1, std_normalize=True)
    assert spans(normalized[0]) == [(1.0, 5), (0.0, 5)]
    assert spans(normalized[1]) == [(-1.0, 10)]


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


def test_handoff_values_shape_credit():
    # Everyone completes "reproduce", so without values its tokens get zero credit
    # (all entrants agree) and the stage can't be told apart from the next one.
    records = [
        StageRecord([5], 10, 1.0, stage_values=[0.9]),  # strong repro, success
        StageRecord([5], 10, 0.0, stage_values=[0.9]),  # strong repro, lost it in the patch
        StageRecord([5], 10, 0.0, stage_values=[0.2]),  # weak repro, failed
    ]
    adv = stage_advantages(records, 1, std_normalize=False)
    # Equal handoffs earn equal stage-1 credit; the weak handoff is charged for it.
    assert stage_values(records[0], adv[0]) == pytest.approx([7 / 30, 13 / 30])
    assert stage_values(records[1], adv[1]) == pytest.approx([7 / 30, -17 / 30])
    assert stage_values(records[2], adv[2]) == pytest.approx([-14 / 30, 4 / 30])
    # The potentials telescope: totals are still the raw return minus the group mean (4/3).
    for record, tokens, total in zip(records, adv, [2.0, 1.0, 1.0], strict=True):
        assert sum(stage_values(record, tokens)) == pytest.approx(total - 4 / 3)
    # phi == 0 is the unshaped rule exactly.
    unshaped = [StageRecord(r.boundaries, r.num_tokens, r.reward) for r in records]
    zeroed = [StageRecord(r.boundaries, r.num_tokens, r.reward, stage_values=[0.0]) for r in records]
    assert stage_advantages(zeroed, 1) == stage_advantages(unshaped, 1)


def scored(scores, rewards, pool=None):
    """One intermediate stage every rollout completes, scored; the task reward is all that follows it."""
    return [
        StageRecord([5], 10, reward, stage_scores=[score], stage_pools=[pool])
        for score, reward in zip(scores, rewards, strict=True)
    ]


def test_linear_score_recovers_slope_and_conserves_credit():
    # Downstream return is exactly 2 * score + 1, so every leave-one-out slope is 2 with no error.
    records = scored([0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 5.0, 7.0])
    filled, moments = fit_handoff_values(records, 1, {})
    assert [v for r in filled for v in r.stage_values] == pytest.approx([-3.0, -1.0, 1.0, 3.0])
    assert moments[1].slope() == (pytest.approx(2.0), pytest.approx(0.0, abs=1e-6))
    # The potentials telescope: totals are still the raw return minus the group mean (5).
    adv = stage_advantages(filled, 1, std_normalize=False)
    for record, tokens in zip(filled, adv, strict=True):
        assert sum(stage_values(record, tokens)) == pytest.approx(1.0 + record.reward - 5.0)


def test_fit_leaves_own_return_out():
    records = scored([0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 5.0, 7.0])
    changed = scored([0.0, 1.0, 2.0, 3.0], [9.0, 3.0, 5.0, 7.0])
    before, _ = fit_handoff_values(records, 1, {})
    after, _ = fit_handoff_values(changed, 1, {})
    # Rollout 0's outcome moves the slope that credits the others, never its own.
    assert after[0].stage_values == before[0].stage_values
    assert after[1].stage_values != before[1].stage_values


def test_uninformative_score_is_unshaped():
    unscored = [StageRecord([5], 10, reward) for reward in (1.0, 0.0, 1.0, 0.0)]
    constant, _ = fit_handoff_values(scored([2.0] * 4, [1.0, 0.0, 1.0, 0.0]), 1, {})
    assert stage_advantages(constant, 1) == stage_advantages(unscored, 1)


def test_slope_and_standard_error_match_the_textbook():
    # b = Sxy / Sxx = 9.5 / 5, RSS = Syy - b * Sxy = 0.7, SE = sqrt(RSS / (n - 2) / Sxx).
    slope, se = Moments.of([(0.0, 1.0), (1.0, 3.0), (2.0, 4.0), (3.0, 7.0)]).slope()
    assert (slope, se) == (pytest.approx(1.9), pytest.approx(math.sqrt(0.7 / 2 / 5)))


def test_shrinkage_pulls_noisy_slopes_to_zero():
    assert shrunk(2.0, 0.0, 1.0) == 2.0  # an exact fit is kept whole
    assert shrunk(1.0, 1.0, 1.0) == pytest.approx(0.5)  # one standard error from zero keeps half
    assert shrunk(1.0, 1.0, 0.0) == 1.0  # no shrinkage trusts the fit
    assert shrunk(*Moments().slope(), 1.0) == 0.0  # an undetermined fit is no shaping
    assert shrunk(1.5, math.inf, 0.0) == 0.0  # and stays so without shrinkage (no 0 * inf)


def test_prior_moments_fit_a_group_too_small_to_fit_alone():
    # Three completers leave two points for each leave-one-out fit, which is no slope.
    records = scored([0.0, 1.0, 2.0], [0.0, 2.0, 4.0])
    alone, _ = fit_handoff_values(records, 1, {})
    assert [v for r in alone for v in r.stage_values] == pytest.approx([0.0, 0.0, 0.0])
    prior = Moments.of([(0.0, 0.0), (1.0, 2.0), (2.0, 4.0), (3.0, 6.0)])
    pooled, _ = fit_handoff_values(records, 1, {1: prior})
    assert [v for r in pooled for v in r.stage_values] == pytest.approx([-2.0, 0.0, 2.0])


def test_pooled_slope_ignores_group_levels():
    # Two groups with the same within-group slope at different score levels.
    low = [(0.0, 0.0), (1.0, 2.0), (2.0, 4.0)]
    high = [(10.0, 0.0), (11.0, 2.0), (12.0, 4.0)]
    assert (Moments.of(low) + Moments.of(high)).slope()[0] == pytest.approx(2.0)
    # Pooling the raw points would mistake the level difference for a slope.
    assert Moments.of(low + high).slope()[0] < 0.5


def test_score_pools_require_agreement_within_a_stage():
    assert score_pools(scored([0.0, 1.0], [1.0, 0.0], pool="p"), 1) == {1: "p"}
    assert score_pools(scored([0.0, 1.0], [1.0, 0.0]) + [StageRecord([5], 10, 0.0)], 1) is None
    assert score_pools(scored([0.0], [1.0], pool="p") + scored([1.0], [0.0], pool="q"), 1) is None


def test_stage_lambda_one_is_return_to_go():
    adv = stage_advantages(LATE_FAILURE, 3, std_normalize=False, stage_lambda=1.0)
    assert spans(adv[0]) == [(1.75, 10), (1.0, 20), (0.5, 10)]
    # Its return-to-go from "grasp" on equals the group's, so those stages are neutral.
    assert spans(adv[3]) == [(0.75, 10), (0.0, 20), (-0.5, 10)]
