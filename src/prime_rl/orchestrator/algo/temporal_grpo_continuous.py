"""Temporal GRPO with continuous stage rewards.

``temporal_grpo`` ranks the rollouts that entered a stage by a binary outcome:
whether they completed it. Here each stage pays a scalar reward instead, and a
stage's credit is its TD error against the group's stage values. For stage k
of a rollout,

    delta_k = q_k + completed_k * V[k + 1] - V[k]

where ``q_k`` is the reward collected in stage k and ``V[k]`` is the group's
value of entering stage k: the mean reward-to-go of the rollouts that entered
it. Stage k's tokens get ``delta_k`` plus ``stage_lambda`` times the next
stage's advantage.

The record is the one ``temporal_grpo`` reads. An entry may carry ``reward``,
the scalar earned in that stage; absent, a stage pays 1 when completed and 0
when not — the paper's binary outcome. The task reward is paid in the stage the
rollout ended in: the final stage when every intermediate stage was completed,
otherwise the stage it failed. A rollout's return is therefore the sum of its
stage rewards plus the task reward.

With binary stage rewards and ``std_normalize`` this equals ``temporal_grpo``
with ``std_normalize`` (the paper's rule) in every group: the TD error is then
``(1 + V[k + 1]) * (completed_k - mean)`` and normalization cancels the factor.
Unnormalized, a rollout's stage advantages sum to its return minus the group
mean — GRPO's credit, redistributed to the stages that earned it — and
``stage_lambda = 1`` credits each stage with the rollout's remaining return
minus ``V[k]``.
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
