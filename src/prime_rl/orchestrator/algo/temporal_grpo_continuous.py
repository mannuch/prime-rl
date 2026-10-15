"""Temporal GRPO with continuous stage rewards.

``temporal_grpo`` ranks the rollouts that entered a stage by a binary outcome:
whether they completed it. Here each stage pays a scalar reward, and a stage's
credit is its TD error against the group's stage values.

The record is the one ``temporal_grpo`` reads. An entry may carry ``reward``,
the scalar earned in that stage; absent, a stage pays 1 when completed and 0
when not — the paper's binary outcome. The task reward is paid in the stage the
rollout ended in: the final stage when every intermediate stage was completed,
otherwise the stage it failed. A completed entry may also carry ``value``, the
env's estimate of the return still expected from the state it handed off (7),
or instead a ``score`` of that state for the rule to calibrate (8).

Derivation
----------
For rollout i of a group and stages k = 1..K (K is the task stage):

    n_i       stages entered: completed 1..n_i - 1, ended in n_i
    c[i,k]    1 if k < n_i else 0 (completed stage k)
    E_k       entrants {i : n_i >= k}; E_{k+1} is exactly the completers of k
    q[i,k]    reward collected in stage k (the task reward lands at k = n_i)
    G[i,k]    reward-to-go: sum_{j=k..n_i} q[i,j] = q[i,k] + c[i,k] * G[i,k+1]

Over E_k: V_k = mean G[., k] (the group's value of entering stage k),
mu_k = mean q[., k], p_k = |E_{k+1}| / |E_k|. V_{K+1} = 0.

1. Recursion. Expand G in V_k; c[i,k] = 1 exactly on E_{k+1}, so

       V_k = mu_k + p_k * V_{k+1}

2. TD error. Stage k earns its reward plus the value of the state it hands
   off, minus the value of the state it started from. By 1,

       delta[i,k] = q[i,k] + c[i,k] * V_{k+1} - V_k
                  = (q[i,k] - mu_k) + V_{k+1} * (c[i,k] - p_k)

   a quality term and a progression term, both in reward units.

3. Centered. Both terms average to zero over E_k. A stage with one entrant,
   or whose entrants all agree, contributes zero.

4. Conservation. A rollout's deltas telescope:

       sum_{k=1..n_i} delta[i,k] = G[i,1] - V_1

   which is GRPO's advantage on the total return, redistributed to the stages
   that earned it. The paper's Stage-Reward GRPO baseline instead broadcasts
   this one value over every token.

5. Paper reduction. With default stage rewards, a completer of stage k < K
   collects 1 and a rollout failing it collects its task reward r_f, so
   mu_k = p_k + (1 - p_k) r_f, and both terms of 2 are multiples of
   c[i,k] - p_k. With the paper's outcome R[i,k] = c[i,k], whose mean over
   the entrants is p_k,

       delta[i,k] = (1 - r_f + V_{k+1}) * (R[i,k] - p_k)

   whenever every rollout failing stage k has the same r_f (a binary task
   reward gives r_f = 0). The factor is shared by stage k's entrants, so
   dividing by their std (``std_normalize``) gives the paper's
   (R[i,k] - p_k) / sigma_k (Eq. 23) as long as it is positive, i.e. as
   long as completing the stage is worth more than failing it; where it is
   not, this rule reverses the paper's sign and is right to. The factor is
   at least 1 for a binary task reward. A default stage reward of 0 would
   leave only V_{k+1}, which is 0 in a group where every rollout fails the
   task, and would erase the signal in the groups the method exists for.
   Failures with differing partial credit break the reduction: their quality
   terms then differ, which is the point of a stage reward.

6. Lambda. A[i,n_i] = delta[i,n_i] and A[i,k] = delta[i,k] + stage_lambda *
   A[i,k+1]. At 0 a stage answers only for its reward and the state it hands
   off; at 1 it telescopes as in 4 to G[i,k] - V_k, the remaining return.

7. Handoff values. The TD error prices every handoff out of stage k at the
   same V_{k+1}, so it drops the carry-forward of stage k's actions: the true
   advantage adds V(s_{k+1}) - V_{k+1}, the value of the state actually handed
   off. An entry's ``value`` is the env's estimate phi_k of that state's
   remaining return, and the stage reward becomes

       q'[i,k] = q[i,k] + phi_k - phi_{k-1}

   with phi_0 = 0 and phi = 0 for a failed stage and for K. This is
   potential-based shaping: the potentials telescope, so the return and 4 are
   unchanged, and phi == 0 is this rule exactly. Expanding,

       delta'[i,k] = delta[i,k] + c[i,k] * (phi_k - mean phi_k over E_{k+1})
                                - (phi_{k-1} - mean phi_{k-1} over E_k)

   so only phi's deviations within a stage matter, and 3 still holds. At
   stage_lambda 0 the dropped term shrinks to the centered
   E[V(s_{k+1}) - phi_k | a_k], the carry-forward phi fails to predict; with
   phi = V the estimator is the one-step advantage under an exact critic, up
   to the sampling error of the group means. Each stage is judged on what it
   added to the state it received, not on what it inherited. phi is in
   reward units: a score of beta * V leaves (1 - beta) of the bias, and
   ``std_normalize`` rescales delta, not phi against q.

8. Fitted handoff values. A completed entry may carry a ``score`` x instead,
   any measurement of the handoff state, on any scale. Over stage k's
   completers, regress the raw downstream return on it:

       phi[i,k] = b_k * (x[i,k] - mean x[.,k]),
       b_k      = Cov(G[.,k+1], x[.,k]) / Var(x[.,k])

   Only the slope matters: a constant added to phi cancels in 3, so the fit
   is centered and carries no intercept, and a completion is priced at
   V_{k+1} + phi, its own predicted handoff. The handoff-out term becomes
   stage_lambda = 1 projected onto the score: it keeps the part of stage k's
   realised downstream deviation that x predicts and drops the residual,
   which is later stages' own actions and luck. A score that measures the
   state with noise attenuates the slope, toward no shaping. As in 7, totals
   are unchanged whatever the fit; a bad slope can only move credit between
   a rollout's own stages. Three guards:
     - each rollout's slope is fit without its own return (leave-one-out),
       so no outcome helps set the slope that credits it;
     - a slope t = b / se standard errors from zero is scaled by
       t^2 / (t^2 + score_shrinkage), so a noisy fit decays to phi = 0,
       which is this rule unshaped;
     - moments are taken about each group's own means and summed, so groups
       sharing a ``pool`` key share a within-group slope that their
       differences in level cannot confound. A key's moments come only from
       earlier groups, decayed by score_decay per group.

Why the TD error: with "entered stage k" as the state and the stage's tokens
as the action, delta is the actor-critic advantage with the group mean as the
critic; GRPO is the one-stage case. It charges a stage only for what it
controlled, which is the paper's rule against credit aliasing, in units where
stage quality and progression add. It assumes a stage's entrants are
interchangeable; ``stage_lambda`` > 0 lets later outcomes flow back when the
quality of a stage's work carries into the next.

Knobs
-----
``stage_lambda`` (default 0) sets how much of a rollout's later outcome its
earlier stages answer for. At 0, every rollout that enters stage k+1 hands
off the same value V_{k+1}, so a stage is not blamed for what happens after
it: the lowest-variance setting, and the one that keeps credit aliasing out.
That is right when completing a stage captures everything later stages need.
It is wrong when stages complete well or badly in ways the next stage feels:
a grasp that holds but slips in transport, a patch that passes ``reproduce``
but makes ``verify`` harder. At 0 those handoffs earn the same credit. The
symptom is a stage's completion rate rising while the next stage's completion
rate among its entrants falls. Two fixes, best first: give the stage a
``value`` (7), or split the stage, so the TD error sees the handoff directly;
or raise ``stage_lambda`` so later outcomes flow back, at the cost of
variance and some of the aliasing the method removes.
1 gives each stage its full remaining return, G[i,k] - V_k.

``std_normalize`` (default off) divides each stage's credits by their std over
the stage's entrants. Off, credit stays in reward units: conservation (4)
holds, a stage that decides more of the return gets more weight, and this
matches prime-rl's GRPO and ``temporal_grpo``, neither of which normalizes by
default. On, every stage that separates its entrants contributes unit-scale
signal, whatever its reward scale or entrant count, and with binary rewards
this is exactly the paper's rule (5). The cost of on: a late stage with few
entrants, or continuous rewards that differ only by noise, get that noise
blown up to full size, and 4 no longer holds, so a stage's weight stops
reflecting how much it moves the return. Turn it on to reproduce the paper,
or to put stage rewards on unrelated scales on an equal footing. With it on,
the lambda sum mixes per-stage-normalized deltas, so 1 is exactly
G[i,k] - V_k only off.

``score_shrinkage`` (default 1) and ``score_decay`` (default 0.9) act only on
scored stages. A group fit from 8 to 16 completers is noisy, with a slope
standard error of about sd(residual) / (sd(x) * sqrt(n)), and shrinkage
turns that noise into a bias toward no shaping rather than toward a wrong
slope. A fit from this group alone is never stale: it describes the same
policy that produced the returns it redistributes. A pooled fit trades that
for lower noise, and ``score_decay`` sets the trade: a key remembers about
1 / (1 - score_decay) groups.

How to write ``reward`` and ``value`` from an env — what each answers, when
to override the reward default, how to calibrate a value, a worked record —
is in ``docs/algorithms.md`` under "Writing the record".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import verifiers.v1 as vf

from prime_rl.configs.algorithm import TemporalGRPOContinuousAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm, iter_trainable_traces
from prime_rl.orchestrator.algo.routing import assign_advantages
from prime_rl.orchestrator.algo.temporal_grpo import StageRecord, grpo_credit, stage_records
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.orchestrator.clients import InferenceClient


@dataclass(frozen=True)
class Moments:
    """Second moments of (score, downstream return) about each group's own
    means, summed over groups, so a pooled slope is never confounded by groups
    that differ in level."""

    dof: float = 0.0  # sum over groups of (points - 1)
    xx: float = 0.0
    xy: float = 0.0
    yy: float = 0.0

    @classmethod
    def of(cls, points: list[tuple[float, float]]) -> Moments:
        if len(points) < 2:
            return cls()
        mean_x = sum(x for x, _ in points) / len(points)
        mean_y = sum(y for _, y in points) / len(points)
        return cls(
            len(points) - 1,
            sum((x - mean_x) ** 2 for x, _ in points),
            sum((x - mean_x) * (y - mean_y) for x, y in points),
            sum((y - mean_y) ** 2 for _, y in points),
        )

    def __add__(self, other: Moments) -> Moments:
        return Moments(self.dof + other.dof, self.xx + other.xx, self.xy + other.xy, self.yy + other.yy)

    def decayed(self, decay: float) -> Moments:
        return Moments(decay * self.dof, decay * self.xx, decay * self.xy, decay * self.yy)

    def slope(self) -> tuple[float, float]:
        """Least-squares slope of return on score and its standard error; (0, inf) when undetermined."""
        if self.xx <= 0.0 or self.dof < 2:
            return 0.0, math.inf
        slope = self.xy / self.xx
        residual = max(self.yy - slope * self.xy, 0.0)
        return slope, math.sqrt(residual / (self.dof - 1) / self.xx)


def shrunk(slope: float, se: float, shrinkage: float) -> float:
    """A slope t = slope / se standard errors from zero keeps t^2 / (t^2 + shrinkage) of itself."""
    if slope == 0.0 or not math.isfinite(se):
        return 0.0
    return slope * slope**2 / (slope**2 + shrinkage * se**2)


def collected_rewards(record: StageRecord, num_intermediate: int) -> list[float]:
    """Reward collected in each entered stage; the stage the rollout ended in also pays the task reward."""
    num_stages = num_intermediate + 1
    entered = min(len(record.boundaries) + 1, num_stages)
    q = [record.stage_reward(k) for k in range(1, min(entered, num_intermediate) + 1)]
    if entered == num_stages:
        q.append(0.0)
    q[-1] += record.reward
    return q


def score_pools(records: list[StageRecord], num_intermediate: int) -> dict[int, str | None] | None:
    """The pool key of each scored stage, or None when a stage's completers
    disagree on whether it is scored or on its key."""
    pools: dict[int, str | None] = {}
    for k in range(1, num_intermediate + 1):
        completers = [record for record in records if len(record.boundaries) >= k]
        scored = {record.stage_score(k) is not None for record in completers}
        keys = {record.stage_pool(k) for record in completers}
        if len(scored) > 1 or len(keys) > 1:
            return None
        if scored == {True}:
            pools[k] = keys.pop()
    return pools


def fit_handoff_values(
    records: list[StageRecord],
    num_intermediate: int,
    priors: dict[int, Moments],
    *,
    shrinkage: float = 1.0,
) -> tuple[list[StageRecord], dict[int, Moments]]:
    """Fill each scored stage's handoff values from its completers' scores.

    For every intermediate stage whose completers carry a ``score``, regress
    their raw downstream return on it, with ``priors[k]`` (moments pooled from
    earlier groups) added to this group's, and set each completer's value to
    the shrunk slope times its score's deviation from the completers' mean.
    Each rollout's slope leaves its own return out. Returns the records with
    those values filled in and each scored stage's moments from this group.
    """
    returns = [collected_rewards(record, num_intermediate) for record in records]
    values = [[record.stage_value(k) for k in range(1, len(record.boundaries) + 1)] for record in records]
    moments: dict[int, Moments] = {}
    for k in range(1, num_intermediate + 1):
        points: dict[int, tuple[float, float]] = {}
        for i, record in enumerate(records):
            score = record.stage_score(k)
            if score is not None:
                points[i] = (score, sum(returns[i][k:]))
        if not points:
            continue
        mean_score = sum(x for x, _ in points.values()) / len(points)
        prior = priors.get(k, Moments())
        for i, (score, _) in points.items():
            others = Moments.of([point for j, point in points.items() if j != i])
            slope = shrunk(*(prior + others).slope(), shrinkage)
            values[i][k - 1] = slope * (score - mean_score)
        moments[k] = Moments.of(list(points.values()))
    filled = [replace(record, stage_values=v) for record, v in zip(records, values, strict=True)]
    return filled, moments


def stage_advantages(
    records: list[StageRecord],
    num_intermediate: int,
    *,
    std_normalize: bool = False,
    stage_lambda: float = 0.0,
    eps: float = 1e-6,
) -> list[list[float]]:
    """Per-token advantages for each rollout: each stage's TD error against
    the group's stage values, optionally normalized per stage, accumulated
    backwards with ``stage_lambda``.

    Stage k (1-based) is entered by rollouts that completed k-1 intermediate
    stages. The final stage K = num_intermediate + 1 has no record entry; its
    only reward is the task reward. Over a stage's entrants the TD errors are
    mean-zero by construction, so a stage whose entrants all agree, or with a
    single entrant, contributes zero.
    """
    num_stages = num_intermediate + 1
    entered = [min(len(record.boundaries) + 1, num_stages) for record in records]
    # rewards[i][k - 1]: reward collected in stage k, shaped by the handoff values.
    rewards: list[list[float]] = []
    for record, n in zip(records, entered, strict=True):
        q = collected_rewards(record, num_intermediate)
        # Potential shaping: a stage is paid the value of the state it hands off and charged the value it received.
        for k in range(1, n + 1):
            q[k - 1] += record.stage_value(k) - record.stage_value(k - 1)
        rewards.append(q)

    # Group value of stage k: mean reward-to-go over its entrants.
    values = [0.0] * (num_stages + 2)
    for k in range(num_stages, 0, -1):
        returns = [sum(q[k - 1 :]) for q, n in zip(rewards, entered, strict=True) if n >= k]
        values[k] = sum(returns) / len(returns) if returns else 0.0

    # TD error; completing stage k (k < n) hands over the next stage's group value.
    deltas = [
        [q[k - 1] + (values[k + 1] if k < n else 0.0) - values[k] for k in range(1, n + 1)]
        for q, n in zip(rewards, entered, strict=True)
    ]
    if std_normalize:
        for k in range(1, num_stages + 1):
            column = [d[k - 1] for d, n in zip(deltas, entered, strict=True) if n >= k]
            if not column:
                continue
            scale = (sum(x * x for x in column) / len(column)) ** 0.5 + eps  # deltas are mean-zero over entrants
            for d, n in zip(deltas, entered, strict=True):
                if n >= k:
                    d[k - 1] /= scale

    out: list[list[float]] = []
    for record, n, d in zip(records, entered, deltas, strict=True):
        tokens = [0.0] * record.num_tokens
        advantage = 0.0
        for k in range(n, 0, -1):
            advantage = d[k - 1] + stage_lambda * advantage
            start = record.boundaries[k - 2] if k >= 2 else 0
            end = record.boundaries[k - 1] if k < n else record.num_tokens  # the stage it ended in takes the suffix
            for t in range(start, end):
                tokens[t] = advantage
        out.append(tokens)
    return out


class TemporalGRPOContinuousAlgorithm(Algorithm):
    """Temporal GRPO with per-stage scalar rewards. Groups whose traces lack a
    consistent stage record fall back to plain group-mean GRPO credit."""

    def __init__(self, config: TemporalGRPOContinuousAlgoConfig, clients: InferenceClient):
        super().__init__(config, clients)
        self.stage_key = config.stage_key
        self.std_normalize = config.std_normalize
        self.stage_lambda = config.stage_lambda
        self.success_threshold = config.success_threshold
        self.score_shrinkage = config.score_shrinkage
        self.score_decay = config.score_decay
        self.pooled: dict[str, Moments] = {}  # per ``pool`` key: score moments from earlier groups, decayed

    async def score_group(self, episodes: list[vf.Episode]) -> None:
        traces = [trace for _, trace in iter_trainable_traces(episodes)]
        if not traces:
            return
        group = stage_records(traces, self.stage_key, self.success_threshold)
        if group is None:
            return grpo_credit(traces)
        records, stage_names = group
        pools = score_pools(records, len(stage_names))
        if pools is None:
            get_logger().debug(
                "temporal_grpo_continuous: plain GRPO credit for group (stage scores or pool keys differ within the group)"
            )
            return grpo_credit(traces)
        priors = {k: self.pooled.get(key, Moments()) for k, key in pools.items() if key is not None}
        records, moments = fit_handoff_values(records, len(stage_names), priors, shrinkage=self.score_shrinkage)
        self._record_fits(traces, stage_names, pools, priors, moments)
        updates: dict[str, Moments] = {}
        for k, key in pools.items():
            if key is not None:
                updates[key] = updates.get(key, Moments()) + moments[k]
        for key, group_moments in updates.items():
            self.pooled[key] = self.pooled.get(key, Moments()).decayed(self.score_decay) + group_moments
        per_token = stage_advantages(
            records,
            len(stage_names),
            std_normalize=self.std_normalize,
            stage_lambda=self.stage_lambda,
        )
        for trace, values in zip(traces, per_token, strict=True):
            assign_advantages(trace, values)

    @staticmethod
    def _record_fits(
        traces: list[vf.Trace],
        stage_names: list[str],
        pools: dict[int, str | None],
        priors: dict[int, Moments],
        moments: dict[int, Moments],
    ) -> None:
        """Log each scored stage's slope (this group alone, and pooled when keyed).
        A slope within about one standard error of zero is noise the shrinkage
        is discarding; a negative one says the score predicts worse outcomes."""
        fits: dict[str, float] = {}
        for k, key in pools.items():
            name = stage_names[k - 1]
            slope, se = moments[k].slope()
            if math.isfinite(se):
                fits[f"stage/{name}/score_slope"] = slope
                fits[f"stage/{name}/score_slope_se"] = se
            if key is not None:
                pooled_slope, pooled_se = (priors[k] + moments[k]).slope()
                if math.isfinite(pooled_se):
                    fits[f"stage/{name}/score_slope_pooled"] = pooled_slope
        for trace in traces:
            trace.record_metrics(fits)
