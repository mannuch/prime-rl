"""Temporal GRPO with continuous stage rewards: TD credit over stages."""

import itertools
import math
import random
from dataclasses import replace

import numpy as np
import pytest

from prime_rl.orchestrator.algo import temporal_grpo as binary
from prime_rl.orchestrator.algo.temporal_grpo import StageRecord
from prime_rl.orchestrator.algo.temporal_grpo_continuous import (
    Fit,
    Moments,
    fit_handoff_values,
    score_pools,
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


@pytest.mark.parametrize("failure_reward", [0.0, -1.0])
def test_binary_stages_reproduce_temporal_grpo(failure_reward):
    # Every group of 4 over the binary outcomes of 2 intermediate stages + a task reward
    # (all-fail, all-pass, single-entrant and mixed groups included): per-stage
    # normalization cancels the (1 - r_fail + V[k+1]) factor, leaving the paper's rule,
    # for any task reward shared by the rollouts failing a stage.
    states = [([], failure_reward), ([10], failure_reward), ([10, 20], 0.0), ([10, 20], 1.0)]
    for group in itertools.product(states, repeat=4):
        records = [StageRecord(list(boundaries), 30, reward) for boundaries, reward in group]
        expected = binary.stage_advantages(records, 2, std_normalize=True)
        for actual, paper in zip(stage_advantages(records, 2, std_normalize=True), expected, strict=True):
            assert actual == pytest.approx(paper, abs=1e-4)


def test_derivation_identities_on_random_groups():
    # The docstring's steps 1, 3, 4, 6, 7 and 9, checked against the code on random groups with
    # failures, continuous stage and task rewards, and handoff values.
    rng = random.Random(0)
    for _ in range(60):
        num_intermediate, size = rng.randint(0, 3), rng.randint(1, 8)
        num_stages = num_intermediate + 1
        records = []
        for _ in range(size):
            completed = rng.randint(0, num_intermediate)
            entered = min(completed + 1, num_intermediate)
            records.append(
                StageRecord(
                    [10 * (k + 1) for k in range(completed)],
                    10 * num_stages,
                    rng.uniform(-1, 2),
                    [rng.uniform(-1, 1) for _ in range(entered)],
                    [rng.uniform(-1, 1) for _ in range(completed)],
                )
            )
        n = [min(len(r.boundaries) + 1, num_stages) for r in records]
        # q with the potentials folded in, G, and the group statistics, straight from the definitions
        q = []
        for r, n_i in zip(records, n, strict=True):
            q_i = [r.stage_reward(k) for k in range(1, min(n_i, num_intermediate) + 1)] + (
                [0.0] if n_i == num_stages else []
            )
            q_i[-1] += r.reward
            q.append([q_i[k - 1] + r.stage_value(k) - r.stage_value(k - 1) for k in range(1, n_i + 1)])
        G = [[sum(q_i[k - 1 :]) for k in range(1, n_i + 1)] for q_i, n_i in zip(q, n, strict=True)]
        V, mu, p = [0.0] * (num_stages + 2), [0.0] * (num_stages + 2), [0.0] * (num_stages + 2)
        for k in range(num_stages, 0, -1):  # downward, so V[k + 1] is known when V[k] is checked
            entrants = [i for i in range(size) if n[i] >= k]
            if entrants:
                V[k] = sum(G[i][k - 1] for i in entrants) / len(entrants)
                mu[k] = sum(q[i][k - 1] for i in entrants) / len(entrants)
                p[k] = sum(n[i] > k for i in entrants) / len(entrants)
                assert V[k] == pytest.approx(mu[k] + p[k] * V[k + 1])  # step 1
        delta = [
            [q[i][k - 1] + (V[k + 1] if k < n[i] else 0.0) - V[k] for k in range(1, n[i] + 1)] for i in range(size)
        ]
        for k in range(1, num_stages + 1):
            assert sum(delta[i][k - 1] for i in range(size) if n[i] >= k) == pytest.approx(0.0, abs=1e-9)  # step 3
        raw = [
            sum(r.stage_reward(k) for k in range(1, min(n_i, num_intermediate) + 1)) + r.reward
            for r, n_i in zip(records, n, strict=True)
        ]
        for i, r in enumerate(records):
            assert sum(delta[i]) == pytest.approx(
                raw[i] - sum(raw) / size
            )  # step 4, on the raw return: step 7 telescopes
        adv = stage_advantages(records, num_intermediate, std_normalize=False)
        adv_one = stage_advantages(records, num_intermediate, std_normalize=False, stage_lambda=1.0)
        for i, r in enumerate(records):
            assert stage_values(r, adv[i]) == pytest.approx(
                delta[i][: len(stage_values(r, adv[i]))]
            )  # step 2 is the code
            assert stage_values(r, adv_one[i]) == pytest.approx(
                [G[i][k - 1] - V[k] for k in range(1, n[i] + 1)][: len(stage_values(r, adv_one[i]))]
            )  # step 6
        # step 9: under lambda, handoff-out is scaled by (1 - lambda) and handoff-in keeps full weight
        lam = rng.choice([0.3, 0.7, 1.0])
        unshaped = [StageRecord(r.boundaries, r.num_tokens, r.reward, r.stage_rewards) for r in records]
        shaped_lam = stage_advantages(records, num_intermediate, std_normalize=False, stage_lambda=lam)
        plain_lam = stage_advantages(unshaped, num_intermediate, std_normalize=False, stage_lambda=lam)
        phi_mean = [0.0] * (num_stages + 1)
        for j in range(1, num_intermediate + 1):
            completers = [r for r in records if len(r.boundaries) >= j]
            if completers:
                phi_mean[j] = sum(r.stage_value(j) for r in completers) / len(completers)

        def psi(r, j):
            return r.stage_value(j) - phi_mean[j] if 1 <= j <= len(r.boundaries) else 0.0

        for i, r in enumerate(records):
            got, plain = stage_values(r, shaped_lam[i]), stage_values(r, plain_lam[i])
            want = [
                plain[k - 1] - psi(r, k - 1) + (1 - lam) * sum(lam ** (j - k) * psi(r, j) for j in range(k, n[i]))
                for k in range(1, len(got) + 1)
            ]
            assert got == pytest.approx(want)


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
    filled, moments, _ = fit_handoff_values(records, 1, {1: None}, {})
    assert [v for r in filled for v in r.stage_values] == pytest.approx([-3.0, -1.0, 1.0, 3.0])
    fit = moments[1].fit()
    assert (fit.weights, fit.se) == (pytest.approx([2.0]), pytest.approx([0.0], abs=1e-6))
    # The potentials telescope: totals are still the raw return minus the group mean (5).
    adv = stage_advantages(filled, 1, std_normalize=False)
    for record, tokens in zip(filled, adv, strict=True):
        assert sum(stage_values(record, tokens)) == pytest.approx(1.0 + record.reward - 5.0)


def test_fit_leaves_own_return_out():
    records = scored([0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 5.0, 7.0])
    changed = scored([0.0, 1.0, 2.0, 3.0], [9.0, 3.0, 5.0, 7.0])
    before, _, _ = fit_handoff_values(records, 1, {1: None}, {})
    after, _, _ = fit_handoff_values(changed, 1, {1: None}, {})
    # Rollout 0's outcome moves the slope that credits the others, never its own.
    assert after[0].stage_values == before[0].stage_values
    assert after[1].stage_values != before[1].stage_values


def test_uninformative_score_is_unshaped():
    unscored = [StageRecord([5], 10, reward) for reward in (1.0, 0.0, 1.0, 0.0)]
    constant, _, _ = fit_handoff_values(scored([2.0] * 4, [1.0, 0.0, 1.0, 0.0]), 1, {1: None}, {})
    assert stage_advantages(constant, 1) == stage_advantages(unscored, 1)


def test_slope_and_standard_error_match_the_textbook():
    # b = Sxy / Sxx = 9.5 / 5, RSS = Syy - b * Sxy = 0.7, SE = sqrt(RSS / (n - 2) / Sxx).
    fit = Moments.of([(0.0, 1.0), (1.0, 3.0), (2.0, 4.0), (3.0, 7.0)]).fit()
    assert (fit.weights, fit.se) == (pytest.approx([1.9]), pytest.approx([math.sqrt(0.7 / 2 / 5)]))
    assert fit.wald == pytest.approx((1.9 / math.sqrt(0.7 / 2 / 5)) ** 2)  # the squared t of one feature


def test_shrinkage_pulls_noisy_slopes_to_zero():
    def fit(weights, wald):
        return Fit(np.array(weights), np.ones(len(weights)), wald, len(weights))

    assert fit([2.0], math.inf).shrunk(1.0) == pytest.approx([2.0])  # an exact fit is kept whole
    assert fit([1.0], 1.0).shrunk(1.0) == pytest.approx([0.5])  # one standard error from zero keeps half
    assert fit([1.0], 1.0).shrunk(0.0) == pytest.approx([1.0])  # no shrinkage trusts the fit
    # With p features, noise puts the weights about sqrt(p) standard errors from zero, so W = p keeps half too.
    assert fit([1.0, -2.0], 2.0).shrunk(1.0) == pytest.approx([0.5, -1.0])
    assert Moments.empty(1).fit() is None  # an undetermined fit is no shaping


def test_prior_moments_fit_a_group_too_small_to_fit_alone():
    # Three completers leave two points for each leave-one-out fit, which is no slope.
    records = scored([0.0, 1.0, 2.0], [0.0, 2.0, 4.0])
    alone, _, _ = fit_handoff_values(records, 1, {1: None}, {})
    assert [v for r in alone for v in r.stage_values] == pytest.approx([0.0, 0.0, 0.0])
    prior = Moments.of([(0.0, 0.0), (1.0, 2.0), (2.0, 4.0), (3.0, 6.0)])
    pooled, _, _ = fit_handoff_values(records, 1, {1: "p"}, {"p": prior})
    assert [v for r in pooled for v in r.stage_values] == pytest.approx([-2.0, 0.0, 2.0])


def test_tied_stages_leave_a_rollouts_points_out_at_every_stage():
    # Two scored stages share a key; each rollout's return after both is its task reward.
    def tied(rewards):
        return [
            StageRecord([5, 10], 15, reward, stage_rewards=[0.0, 0.0], stage_scores=[x, x + 1], stage_pools=["p", "p"])
            for x, reward in zip([0.0, 1.0, 2.0, 3.0, 4.0], rewards, strict=True)
        ]

    before, _, keyed = fit_handoff_values(tied([0.0, 2.0, 4.0, 6.0, 8.0]), 2, {1: "p", 2: "p"}, {})
    assert [v for r in before for v in r.stage_values] == pytest.approx([-4, -4, -2, -2, 0, 0, 2, 2, 4, 4])
    assert keyed["p"].fit().weights == pytest.approx([2.0])
    after, _, _ = fit_handoff_values(tied([0.0, 2.0, 4.0, 6.0, 20.0]), 2, {1: "p", 2: "p"}, {})
    # Rollout 4's outcome moves the slope that credits the others, never its own, at either stage.
    assert after[4].stage_values == before[4].stage_values
    assert after[0].stage_values != before[0].stage_values


def test_tied_stages_count_rollouts_not_points():
    points = [(0.0, 1.0), (1.0, 3.0), (2.0, 4.0), (3.0, 7.0)]
    twice = Moments.over([dict(enumerate(points)), dict(enumerate(points))])
    # The same rollouts seen at two stages carry no more information than at one.
    assert (twice.fit().weights, twice.fit().se) == (pytest.approx([1.9]), pytest.approx(Moments.of(points).fit().se))
    # Counting points would treat them as twice the evidence.
    assert replace(twice, tied=False).fit().se < twice.fit().se


def test_pooled_slope_ignores_group_levels():
    # Two groups with the same within-group slope at different score levels.
    low = [(0.0, 0.0), (1.0, 2.0), (2.0, 4.0)]
    high = [(10.0, 0.0), (11.0, 2.0), (12.0, 4.0)]
    assert (Moments.of(low) + Moments.of(high)).fit().weights == pytest.approx([2.0])
    # Pooling the raw points would mistake the level difference for a slope.
    assert Moments.of(low + high).fit().weights[0] < 0.5


def test_score_pools_require_agreement_within_a_stage():
    assert score_pools(scored([0.0, 1.0], [1.0, 0.0], pool="p"), 1) == {1: "p"}
    assert score_pools(scored([0.0, 1.0], [1.0, 0.0]) + [StageRecord([5], 10, 0.0)], 1) is None
    assert score_pools(scored([0.0], [1.0], pool="p") + scored([1.0], [0.0], pool="q"), 1) is None
    # The parser records an unscored completed stage as a None score, as on a stage with a value.
    assert score_pools([StageRecord([5], 10, 0.0, stage_values=[0.5], stage_scores=[None])] * 2, 1) == {}
    # Completers must report the same features, and stages sharing a key the same ones.
    assert score_pools(scored([{"a": 0.0}], [1.0]) + scored([{"b": 1.0}], [0.0]), 1) is None

    def two_stages(first, second):
        return [StageRecord([5, 10], 15, 0.0, stage_scores=[first, second], stage_pools=["p", "p"])]

    assert score_pools(two_stages({"a": 0.0}, {"a": 1.0}), 2) == {1: "p", 2: "p"}
    assert score_pools(two_stages({"a": 0.0}, {"a": 1.0, "b": 2.0}), 2) is None


def test_score_features_fit_jointly():
    # Downstream return is exactly 2 * a - b + 1, with a and b uncorrelated: every
    # leave-one-out fit recovers both weights, and each value is the prediction's deviation.
    a, b = [0.0, 1.0, 0.0, 1.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    records = scored([{"a": x, "b": y} for x, y in zip(a, b)], [2 * x - y + 1 for x, y in zip(a, b)])
    filled, moments, _ = fit_handoff_values(records, 1, {1: None}, {})
    assert moments[1].fit().weights == pytest.approx([2.0, -1.0])
    predicted = [2 * x - y for x, y in zip(a, b)]
    mean = sum(predicted) / len(predicted)
    assert [r.stage_values[0] for r in filled] == pytest.approx([v - mean for v in predicted])


def test_one_named_feature_is_the_scalar_score():
    scores, rewards = [0.0, 1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 6.0, 5.0, 9.0]
    plain, _, _ = fit_handoff_values(scored(scores, rewards), 1, {1: None}, {})
    named, _, _ = fit_handoff_values(scored([{"x": s} for s in scores], rewards), 1, {1: None}, {})
    assert [r.stage_values for r in named] == [r.stage_values for r in plain]


def test_a_constant_feature_drops_out_of_the_fit():
    # "c" never varies among the completers, so the fit is "x" alone, with one feature's
    # degrees of freedom and shrinkage.
    scores, rewards = [0.0, 1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 6.0, 5.0, 9.0]
    alone, _, _ = fit_handoff_values(scored(scores, rewards), 1, {1: None}, {})
    padded, moments, _ = fit_handoff_values(scored([{"x": s, "c": 7.0} for s in scores], rewards), 1, {1: None}, {})
    assert [r.stage_values[0] for r in padded] == pytest.approx([r.stage_values[0] for r in alone])
    assert moments[1].fit().rank == 1


def test_stage_lambda_one_is_return_to_go():
    adv = stage_advantages(LATE_FAILURE, 3, std_normalize=False, stage_lambda=1.0)
    assert spans(adv[0]) == [(1.75, 10), (1.0, 20), (0.5, 10)]
    # Its return-to-go from "grasp" on equals the group's, so those stages are neutral.
    assert spans(adv[3]) == [(0.75, 10), (0.0, 20), (-0.5, 10)]
