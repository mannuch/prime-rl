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
   any measurement of the handoff state, on any scale, or several named ones
   (features), making x a vector. Over stage k's completers, regress the raw
   downstream return on it:

       phi[i,k] = b_k . (x[i,k] - mean x[.,k]),
       b_k      = Cov(x[.,k])^-1 Cov(x[.,k], G[.,k+1])

   a linear critic over the features the env chose; one score is the
   one-feature case, b_k = Cov / Var. Only the weights matter: a constant
   added to phi cancels in 3, so the fit is centered and carries no
   intercept, and a completion is priced at V_{k+1} + phi, its own predicted
   handoff. The handoff-out term becomes stage_lambda = 1 projected onto the
   features: it keeps the part of stage k's realised downstream deviation
   that x predicts and drops the residual, which is later stages' own actions
   and luck. A feature that measures the state with noise attenuates its
   weight, toward no shaping. As in 7, totals are unchanged whatever the fit;
   bad weights can only move credit between a rollout's own stages. Three
   guards:
     - each rollout's weights are fit without its own return (leave-one-out),
       so no outcome helps set the weights that credit it;
     - weights W = b' Cov(b)^-1 b squared standard errors from zero (t^2 for
       one feature) are scaled by W / (W + score_shrinkage * p), p the
       features the fit can tell apart. Noise alone puts W near p, so a
       noisy fit decays to phi = 0, which is this rule unshaped, at any p;
     - moments are taken about each stage's own means in each group and
       summed, so stages and groups sharing a ``pool`` key share one set of
       weights that their differences in level cannot confound. Within a
       group, the stages under one key fit together; across groups, a key's
       moments come from earlier groups, decayed by score_decay per group (0
       keeps none, so a key only ties stages within a group).
   Each feature costs about one completer: a fit needs more degrees of
   freedom than features, and every feature, informative or not, adds noise
   to the values. A feature constant across a fit drops out of it.
   Correlated features make the individual weights unstable, but not the
   values, which are all that enter the credit.
   A key that ties several stages, such as stages cut by an env's clock,
   takes several points from each rollout, and they share its outcome. Its
   leave-one-out drops all of a rollout's points, and its covariance counts
   rollouts, not points. Sharing weights changes what the fit's error does:
   separate per-stage weights err independently, and the credit across a
   short stage, phi_k - phi_{k-1}, picks up their difference however little
   the state changed; shared weights err once, so that credit is mis-scaled
   but no longer swamped. A key's history shares the weights once it has
   built up; fitting the tied stages together within the group shares them
   from the first group, from the current policy alone.

9. Lambda with handoff values. For a completed stage j, write
   psi[i,j] = phi[i,j] - (mean phi_j over E_{j+1}). In the lambda-sum of
   shaped credits, psi[i,j] enters delta'_j as handoff-out with weight
   lambda^(j-k) and delta'_{j+1} as handoff-in with weight -lambda^(j-k+1).
   Collecting terms, with A the unshaped lambda-credit and no normalization:

       A'[i,k] = A[i,k] - psi[i,k-1]
                 + (1 - lambda) * sum_{j=k..n_i-1} lambda^(j-k) * psi[i,j]

   Handoff-in keeps full weight at every lambda; handoff-out is scaled by
   (1 - lambda). At 0 this is 7. At 1, A' = G[i,k] - V_k - psi[i,k-1]: the
   carry-forward comes from the realised return, and phi survives only
   through psi[i,k-1], a function of the state stage k started from. Since
   E[grad log pi(a_k | s_k) f(s_k)] = 0 for any such f, it is a baseline:
   whatever its calibration it cannot bias stage k, only change its
   variance (a good phi lowers it; a badly over-scaled one can raise it).
   In between, the carry-forward is (1 - lambda) parts phi's prediction and
   lambda parts sampled return; for a fitted score (8), lambda adds back
   that fraction of the residual the projection drops. With phi = V
   exactly, every delta'_j has zero mean given s_j up to the sampling error
   of the group means (as in 7), so A' is unbiased at every lambda in that
   sense and raising lambda only adds variance. The baseline
   argument needs phi[i,k-1] to depend on nothing rollout i did after stage
   k - 1: for a value, that it is computed from the handoff state alone; for
   a fitted score, the leave-one-out of 8. With ``std_normalize`` each stage
   is rescaled before the sum: psi[i,j] is paid at stage j's scale and
   charged at stage j + 1's, so it cancels only to the extent neighbouring
   stages' scales agree, and the baseline argument at lambda = 1 fails with
   it (two rollouts with the same start but different handoffs then get
   different stage-(j + 1) credit). To lean on either, turn
   ``std_normalize`` off.

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
1 gives each stage its full remaining return, G[i,k] - V_k. With a value or
score in place, 9 says what ``stage_lambda`` does to it: at 0 the signal
carries the carry-forward and its calibration matters; near 1, with
``std_normalize`` off, it is only a baseline for the next stage, so a
miscalibrated one costs variance, never bias (normalized, it keeps carrying
credit at every lambda). Start at 0 with a signal in place, and raise it
only if the symptom persists.

``std_normalize`` (default off) divides each stage's credits by their std over
the stage's entrants. Off, credit stays in reward units: conservation (4)
holds, a stage that decides more of the return gets more weight, the lambda
results of 9 hold exactly, and this matches prime-rl's GRPO and
``temporal_grpo``, neither of which normalizes by default. On, every stage
that separates its entrants contributes unit-scale signal, whatever its
reward scale or entrant count, and with binary rewards this is exactly the
paper's rule (5). The cost of on: a late stage with few entrants, or
continuous rewards that differ only by noise, get that noise blown up to
full size; 4 no longer holds, so a stage's weight stops reflecting how much
it moves the return; and the cancellation of 9 fails. Turn it on to
reproduce the paper, or to put stage rewards on unrelated scales on an
equal footing.

``score_shrinkage`` (default 1) and ``score_decay`` (default 0.9) act only on
scored stages. A group fit from 8 to 16 completers is noisy, with a weight
standard error of about sd(residual) / (sd(x) * sqrt(n)) per feature, and
shrinkage turns that noise into a bias toward no shaping rather than toward
wrong weights. A fit from this group alone is never stale: it describes the same
policy that produced the returns it redistributes. A pooled fit trades that
for lower noise, and ``score_decay`` sets the trade: a key remembers about
1 / (1 - score_decay) groups, the current one included, and 0 keeps no
history.

How to write ``reward``, ``value``, ``score`` and ``pool`` from an env is in
``docs/algorithms.md``: the fields and a worked record under "The record"; a
decision chart from milestone or clock stages to the signals to write under
"Choosing a setup"; designing a handoff signal and keeping it current under
"Designing a handoff signal" and "Avoiding stale fits".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import verifiers.v1 as vf

from prime_rl.configs.algorithm import TemporalGRPOContinuousAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm, iter_trainable_traces
from prime_rl.orchestrator.algo.routing import assign_advantages
from prime_rl.orchestrator.algo.temporal_grpo import StageRecord, grpo_credit, stage_records
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.orchestrator.clients import InferenceClient


Point = tuple[np.ndarray, float]  # (score features, downstream return)


def pseudo_inverse(matrix: np.ndarray) -> tuple[np.ndarray, int]:
    """Inverse of a symmetric PSD matrix on its well-determined directions, and their number.

    Rows and columns are first scaled to unit diagonal, so the cut does not
    depend on the features' units; a zero diagonal (a feature constant across
    the fit) drops that feature."""
    diag = np.diag(matrix)
    scale = np.where(diag > 0.0, 1.0 / np.sqrt(np.where(diag > 0.0, diag, 1.0)), 0.0)
    eigenvalues, vectors = np.linalg.eigh(matrix * np.outer(scale, scale))
    kept = eigenvalues > 1e-10 * max(eigenvalues.max(initial=0.0), 0.0)
    if not kept.any():
        return np.zeros_like(matrix), 0
    inverse = (vectors[:, kept] / eigenvalues[kept]) @ vectors[:, kept].T
    return inverse * np.outer(scale, scale), int(kept.sum())


def wald(weights: np.ndarray, covariance: np.ndarray) -> float:
    """weights' covariance^-1 weights: the fit's squared distance from zero in
    standard errors; inf for a nonzero weight with no error."""
    se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    if np.any((se == 0.0) & (weights != 0.0)):
        return math.inf
    determined = se > 0.0
    t = weights[determined] / se[determined]
    correlation = covariance[np.ix_(determined, determined)] / np.outer(se[determined], se[determined])
    inverse, _ = pseudo_inverse(correlation)
    return float(t @ inverse @ t)


@dataclass(frozen=True)
class Fit:
    weights: np.ndarray  # one per score feature
    se: np.ndarray  # their standard errors, from the covariance the guard used
    wald: float  # weights' Cov^-1 weights; the squared t when there is one feature
    rank: int  # features the fit could tell apart

    def shrunk(self, shrinkage: float) -> np.ndarray:
        """The weights scaled by wald / (wald + shrinkage * rank), toward no shaping when noisy."""
        if math.isinf(self.wald):
            return self.weights
        if self.wald <= 0.0:
            return np.zeros_like(self.weights)
        return self.weights * self.wald / (self.wald + shrinkage * self.rank)


@dataclass(frozen=True)
class Moments:
    """Second moments of (score features, downstream return) about each stage's
    own means in each group, summed, so pooled weights are never confounded by
    stages or groups that differ in level.

    A fit over several stages (tied by a ``pool`` key) takes several points
    from one rollout, and they share its outcome, so they are not independent.
    Its covariance counts rollouts, not points: of the textbook covariance with
    rollouts as the degrees of freedom and the cluster-robust one, the guard
    uses whichever puts the weights fewer standard errors from zero. The
    cluster-robust one is built from per-rollout sums a = sum x*y (a vector)
    and C = sum x x' (a matrix) over its centred points."""

    xx: np.ndarray  # (p, p)
    xy: np.ndarray  # (p,)
    yy: float
    dof: float  # sum over stages and groups of (points - 1)
    tied: bool  # some rollout contributed more than one point
    rollouts: float
    aa: np.ndarray  # sums over rollouts of a a' (p, p), a (x) C (p, p, p) and C (x) C (p, p, p, p)
    ac: np.ndarray
    cc: np.ndarray

    @classmethod
    def empty(cls, num_features: int) -> Moments:
        p = num_features
        return cls(
            np.zeros((p, p)),
            np.zeros(p),
            0.0,
            0.0,
            False,
            0.0,
            np.zeros((p, p)),
            np.zeros((p,) * 3),
            np.zeros((p,) * 4),
        )

    @classmethod
    def of(cls, points: list[tuple]) -> Moments:
        """One stage's points, one per rollout; a single feature may be a bare number."""
        return cls.over([dict(enumerate(points))])

    @classmethod
    def over(cls, stages: list[dict[int, tuple]], num_features: int | None = None) -> Moments:
        """Points from several stages, keyed by rollout, each stage centred on its own means."""
        stages = [{i: (np.atleast_1d(np.asarray(x, dtype=float)), float(y)) for i, (x, y) in s.items()} for s in stages]
        if num_features is None:
            num_features = next(len(x) for points in stages for x, _ in points.values())
        empty = cls.empty(num_features)
        xx, xy, yy, dof = empty.xx, empty.xy, 0.0, 0.0
        num_points = 0
        sums: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for points in stages:
            if len(points) < 2:
                continue
            mean_x = sum(x for x, _ in points.values()) / len(points)
            mean_y = sum(y for _, y in points.values()) / len(points)
            dof += len(points) - 1
            num_points += len(points)
            for i, (x, y) in points.items():
                dx, dy = x - mean_x, y - mean_y
                xx, xy, yy = xx + np.outer(dx, dx), xy + dx * dy, yy + dy * dy
                a, c = sums.get(i, (empty.xy, empty.xx))
                sums[i] = (a + dx * dy, c + np.outer(dx, dx))
        return cls(
            xx,
            xy,
            yy,
            dof,
            tied=num_points > len(sums),
            rollouts=len(sums),
            aa=sum((np.einsum("u,v->uv", a, a) for a, _ in sums.values()), empty.aa),
            ac=sum((np.einsum("u,vw->uvw", a, c) for a, c in sums.values()), empty.ac),
            cc=sum((np.einsum("uw,vx->uwvx", c, c) for _, c in sums.values()), empty.cc),
        )

    def __add__(self, other: Moments) -> Moments:
        return Moments(
            self.xx + other.xx,
            self.xy + other.xy,
            self.yy + other.yy,
            self.dof + other.dof,
            self.tied or other.tied,
            self.rollouts + other.rollouts,
            self.aa + other.aa,
            self.ac + other.ac,
            self.cc + other.cc,
        )

    def decayed(self, decay: float) -> Moments:
        return Moments(
            decay * self.xx,
            decay * self.xy,
            decay * self.yy,
            decay * self.dof,
            self.tied,
            decay * self.rollouts,
            decay * self.aa,
            decay * self.ac,
            decay * self.cc,
        )

    def fit(self) -> Fit | None:
        """Least-squares weights of return on the score features; None when
        the fit has no more degrees of freedom than features."""
        dof = self.rollouts - 1 if self.tied else self.dof
        inverse, rank = pseudo_inverse(self.xx)
        if rank == 0 or dof - rank < 1:
            return None
        weights = inverse @ self.xy
        residual = max(self.yy - weights @ self.xy, 0.0)
        covariance = residual / (dof - rank) * inverse
        if self.tied:
            # Points from one rollout share its outcome: of the textbook covariance counting
            # rollouts, not points, and the cluster-robust one, keep the more doubtful.
            cross = np.einsum("uvw,w->uv", self.ac, weights)
            spread = self.aa - cross - cross.T + np.einsum("uwvx,w,x->uv", self.cc, weights, weights)
            clustered = self.rollouts / (self.rollouts - 1) * inverse @ spread @ inverse
            if wald(weights, clustered) < wald(weights, covariance):
                covariance = clustered
        return Fit(weights, np.sqrt(np.maximum(np.diag(covariance), 0.0)), wald(weights, covariance), rank)


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
    disagree on whether it is scored, on its features or on its key, or when
    stages sharing a key have different features."""
    pools: dict[int, str | None] = {}
    features: dict[str, tuple[str, ...]] = {}
    for k in range(1, num_intermediate + 1):
        completers = [record for record in records if len(record.boundaries) >= k]
        names = {None if (score := record.stage_score(k)) is None else tuple(sorted(score)) for record in completers}
        keys = {record.stage_pool(k) for record in completers}
        if len(names) > 1 or len(keys) > 1:
            return None
        if names in ({None}, set()):
            continue
        stage_names, key = names.pop(), keys.pop()
        if key is not None and features.setdefault(key, stage_names) != stage_names:
            return None
        pools[k] = key
    return pools


def score_features(records: list[StageRecord], k: int) -> tuple[str, ...]:
    """The names of stage k's score features, in fit order."""
    return next(tuple(sorted(score)) for record in records if (score := record.stage_score(k)) is not None)


def fit_handoff_values(
    records: list[StageRecord],
    num_intermediate: int,
    pools: dict[int, str | None],
    priors: dict[str, Moments],
    *,
    shrinkage: float = 1.0,
) -> tuple[list[StageRecord], dict[int, Moments], dict[str, Moments]]:
    """Fill each scored stage's handoff values from its completers' scores.

    ``pools`` maps each scored stage to its pool key (``score_pools``). An
    unkeyed stage regresses its completers' raw downstream return on their
    score features alone. Stages sharing a key fit one set of weights together,
    from all their completers in this group plus ``priors[key]`` (moments from
    earlier groups). Each completer's value is the shrunk weights times its
    features' deviation from its stage's means. Each rollout's weights leave
    out every point of its own, since its returns at every stage share its
    outcome. Returns the filled records, each scored stage's moments from this
    group alone, and each key's moments from this group.
    """
    returns = [collected_rewards(record, num_intermediate) for record in records]
    values = [[record.stage_value(k) for k in range(1, len(record.boundaries) + 1)] for record in records]
    names = {k: score_features(records, k) for k in pools}
    points: dict[int, dict[int, Point]] = {}
    for k in pools:
        points[k] = {}
        for i, record in enumerate(records):
            score = record.stage_score(k)
            if score is not None:
                points[k][i] = (np.array([score[name] for name in names[k]]), sum(returns[i][k:]))
    fits: list[tuple[Moments, list[int]]] = []
    tied: dict[str, list[int]] = {}
    for k, key in pools.items():
        if key is None:
            fits.append((Moments.empty(len(names[k])), [k]))
        else:
            tied.setdefault(key, []).append(k)
    fits += [(priors.get(key, Moments.empty(len(names[stages[0]]))), stages) for key, stages in tied.items()]
    for prior, stages in fits:
        num_features = len(names[stages[0]])
        mean_score = {k: sum(x for x, _ in points[k].values()) / len(points[k]) for k in stages}
        for i in {i for k in stages for i in points[k]}:
            others = prior + Moments.over(
                [{j: p for j, p in points[k].items() if j != i} for k in stages], num_features
            )
            fit = others.fit()
            weights = np.zeros(num_features) if fit is None else fit.shrunk(shrinkage)
            for k in stages:
                if i in points[k]:
                    values[i][k - 1] = float(weights @ (points[k][i][0] - mean_score[k]))
    filled = [replace(record, stage_values=v) for record, v in zip(records, values, strict=True)]
    stage_moments = {k: Moments.over([points[k]], len(names[k])) for k in pools}
    key_moments = {
        key: Moments.over([points[k] for k in stages], len(names[stages[0]])) for key, stages in tied.items()
    }
    return filled, stage_moments, key_moments


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
        # Per ``pool`` key and score features: moments from earlier groups, decayed.
        self.pooled: dict[tuple[str, tuple[str, ...]], Moments] = {}

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
        # A key's history is kept per feature set, so an env that changes its features starts afresh.
        features = {key: score_features(records, k) for k, key in pools.items() if key is not None}
        priors = {key: self.pooled.get((key, names), Moments.empty(len(names))) for key, names in features.items()}
        records, stage_moments, key_moments = fit_handoff_values(
            records, len(stage_names), pools, priors, shrinkage=self.score_shrinkage
        )
        self._record_fits(traces, records, stage_names, pools, priors, stage_moments, key_moments)
        for key, group_moments in key_moments.items():
            # Decayed after adding this group, so score_decay = 0 keeps no history.
            self.pooled[key, features[key]] = (priors[key] + group_moments).decayed(self.score_decay)
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
        records: list[StageRecord],
        stage_names: list[str],
        pools: dict[int, str | None],
        priors: dict[str, Moments],
        stage_moments: dict[int, Moments],
        key_moments: dict[str, Moments],
    ) -> None:
        """Log each scored stage's weight per score feature (this stage in this
        group alone, and the key's shared fit when keyed) and the spread of its
        handoff values. A weight within about one standard error of zero is
        noise the shrinkage is discarding; a negative one says the feature
        predicts worse outcomes. With correlated features the weights are
        unstable even when the values are not; the value spread is what moves
        credit."""
        fits: dict[str, float] = {}
        for k, key in pools.items():
            name = stage_names[k - 1]
            features = [f"/{feature}" if feature else "" for feature in score_features(records, k)]
            fit = stage_moments[k].fit()
            if fit is not None:
                for feature, weight, se in zip(features, fit.weights, fit.se, strict=True):
                    fits[f"stage/{name}/score_slope{feature}"] = float(weight)
                    fits[f"stage/{name}/score_slope_se{feature}"] = float(se)
            if key is not None and (pooled := (priors[key] + key_moments[key]).fit()) is not None:
                for feature, weight in zip(features, pooled.weights, strict=True):
                    fits[f"stage/{name}/score_slope_pooled{feature}"] = float(weight)
            values = [record.stage_value(k) for record in records if len(record.boundaries) >= k]
            mean = sum(values) / len(values)
            fits[f"stage/{name}/score_value_std"] = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        for trace in traces:
            trace.record_metrics(fits)
