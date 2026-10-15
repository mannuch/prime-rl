"""Temporal GRPO with continuous stage rewards.

``temporal_grpo`` ranks the rollouts that entered a stage by a binary outcome:
whether they completed it. Here each stage pays a scalar reward, and a stage's
credit is its TD error against the group's stage values.

The record is the one ``temporal_grpo`` reads. An entry may carry ``reward``,
the scalar earned in that stage; absent, a stage pays 1 when completed and 0
when not — the paper's binary outcome. The task reward is paid in the stage the
rollout ended in: the final stage when every intermediate stage was completed,
otherwise the stage it failed. A completed entry may also carry ``value``, the
env's estimate of the return still expected from the state it handed off (7).

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

5. Paper reduction. With default stage rewards and a binary task reward,
   q[i,k] is the paper's outcome R[i,k], and mu_k = p_k for k < K, so

       delta[i,k] = (1 + V_{k+1}) * (R[i,k] - mu_k)

   The factor is shared by stage k's entrants, so dividing by their std
   (``std_normalize``) gives the paper's (R[i,k] - mu_k) / sigma_k (Eq. 23).
   The factor is at least 1. A default stage reward of 0 would leave only
   V_{k+1}, which is 0 in a group where every rollout fails the task, and
   would erase the signal in the groups the method exists for.

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
   unchanged, and phi == 0 is this rule exactly. At stage_lambda 0 the dropped
   term shrinks to E[V(s_{k+1}) - phi_k | a_k], the carry-forward phi fails to
   predict; with phi = V the estimator is the one-step advantage under an
   exact critic. Each stage is judged on what it added to the state it
   received, not on what it inherited. phi is in reward units: a score of
   beta * V leaves (1 - beta) of the bias, and ``std_normalize`` rescales
   delta, not phi against q.

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

``std_normalize`` (default on) divides each stage's credits by their std over
the stage's entrants. On, every stage that separates its entrants contributes
unit-scale signal, whatever its reward scale or entrant count; stage rewards
on unrelated scales then need no calibration, and with binary rewards this is
the paper's rule. The cost: a late stage with few entrants, or continuous
rewards that differ only by noise, get that noise blown up to full size, and
4 no longer holds, so a stage's weight in the update stops reflecting how much
it moves the return. Off, credit stays in reward units: conservation holds, a
stage that decides more of the return gets more weight, and this matches
prime-rl's GRPO, which does not normalize. Turn it off when stage rewards are
on one meaningful scale and you want each stage weighted by how much it
matters; keep it on for binary or mixed-scale rewards. With it on, the lambda
sum mixes per-stage-normalized deltas, so 1 is exactly G[i,k] - V_k only off.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import verifiers.v1 as vf

from prime_rl.configs.algorithm import TemporalGRPOContinuousAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm, iter_trainable_traces
from prime_rl.orchestrator.algo.routing import assign_advantages
from prime_rl.orchestrator.algo.temporal_grpo import StageRecord, grpo_credit, stage_records

if TYPE_CHECKING:
    from prime_rl.orchestrator.clients import InferenceClient


def stage_advantages(
    records: list[StageRecord],
    num_intermediate: int,
    *,
    std_normalize: bool = True,
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
    # rewards[i][k - 1]: reward collected in stage k. The stage a rollout ended in also pays the task reward.
    rewards: list[list[float]] = []
    for record, n in zip(records, entered, strict=True):
        q = [record.stage_reward(k) for k in range(1, min(n, num_intermediate) + 1)]
        if n == num_stages:
            q.append(0.0)
        q[-1] += record.reward
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

    async def score_group(self, episodes: list[vf.Episode]) -> None:
        traces = [trace for _, trace in iter_trainable_traces(episodes)]
        if not traces:
            return
        group = stage_records(traces, self.stage_key, self.success_threshold)
        if group is None:
            return grpo_credit(traces)
        records, stage_names = group
        per_token = stage_advantages(
            records,
            len(stage_names),
            std_normalize=self.std_normalize,
            stage_lambda=self.stage_lambda,
        )
        for trace, values in zip(traces, per_token, strict=True):
            assign_advantages(trace, values)
