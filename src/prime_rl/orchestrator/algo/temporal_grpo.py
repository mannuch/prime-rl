"""Temporal GRPO: stage-conditioned credit assignment (arXiv:2608.13026).

GRPO broadcasts one rollout-level advantage over every sampled token, so a
rollout that clears several stages and fails late is penalized for the early
progress too ("trajectory-level credit aliasing"). Temporal GRPO splits each
rollout into ordered stage intervals, compares only the rollouts that *entered*
a stage, and assigns each stage's group-relative advantage to that stage's
tokens only.

The env declares stage progress. Each trace carries, in
``trace.info[stage_key]``, the ordered *intermediate* stages of its task:

    [{"name": "reproduce", "node": 4}, {"name": "patch", "node": 11}, {"name": "verify", "node": None}]

``node`` is the index into ``trace.nodes`` of the sampled (assistant) node
whose action completed the stage; the stage's interval ends after that turn's
tokens. ``None`` means never completed. ``token`` may be given instead of
``node``: an exclusive end offset into the trace's sampled-token stream (the
order ``assign_advantages`` walks), for stages inside one long response. Stages
are linear prerequisites, so everything after the first uncompleted stage is
ignored. The final stage is implicit: the task reward itself, over the tokens
after the last intermediate stage — so the task objective is preserved exactly.
An entry may also carry ``reward``, the scalar earned in that stage, and a
completed entry either a ``value``, the env's estimate of the return still
expected from the state it handed off, or a ``score`` of that state (a number,
or a dict of named numbers, with an optional ``pool`` key) for the algorithm to
calibrate; this binary rule ignores them all (``temporal_grpo_continuous`` is
the rule that uses them).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import verifiers.v1 as vf

from prime_rl.configs.algorithm import TemporalGRPOAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm, iter_trainable_traces
from prime_rl.orchestrator.algo.routing import assign_advantages, trainable_nodes
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.orchestrator.clients import InferenceClient


Score = float | dict[str, float]  # one handoff measurement, or several named ones


@dataclass
class StageRecord:
    """One rollout reduced to what the credit rules need."""

    boundaries: list[int]  # exclusive sampled-token end offset of each completed intermediate stage
    num_tokens: int  # sampled tokens in the rollout
    reward: float  # the task reward
    stage_rewards: list[float] | None = None  # per entered intermediate stage; None = 1.0 if completed else 0.0
    stage_values: list[float] | None = None  # per completed intermediate stage: value of the state it handed off
    stage_scores: list[Score | None] | None = None  # per completed intermediate stage: uncalibrated handoff score
    stage_pools: list[str | None] | None = None  # per completed intermediate stage: key sharing the score's fit

    def stage_reward(self, k: int) -> float:
        if self.stage_rewards is not None:
            return self.stage_rewards[k - 1]
        return 1.0 if k <= len(self.boundaries) else 0.0

    def stage_value(self, k: int) -> float:
        """Value of the state stage k handed off; 0 unless k is a completed stage with one."""
        if self.stage_values is None or not 1 <= k <= len(self.boundaries):
            return 0.0
        return self.stage_values[k - 1]

    def stage_score(self, k: int) -> dict[str, float] | None:
        """Stage k's score as named features; a single number is the one feature named ""."""
        if self.stage_scores is None or not 1 <= k <= len(self.boundaries):
            return None
        score = self.stage_scores[k - 1]
        return score if isinstance(score, dict) else {"": float(score)}

    def stage_pool(self, k: int) -> str | None:
        if self.stage_pools is None or not 1 <= k <= len(self.boundaries):
            return None
        return self.stage_pools[k - 1]


class ParsedStages(NamedTuple):
    names: list[str]
    boundaries: list[int]  # per completed intermediate stage
    rewards: list[float]  # per entered intermediate stage
    values: list[float]  # per completed intermediate stage
    scores: list[Score | None]  # per completed intermediate stage
    pools: list[str | None]  # per completed intermediate stage


def sampled_turns(trace: vf.Trace) -> list[tuple[int, int]]:
    """(index into ``trace.nodes``, sampled-token count) per trainable sampled
    node, in the order ``assign_advantages`` walks."""
    selected = {id(node) for node in trainable_nodes(trace)}
    return [(index, sum(node.mask)) for index, node in enumerate(trace.nodes) if id(node) in selected]


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _parse_score(score: Any) -> Score | None:
    """A number, or a non-empty dict of finite numbers under non-empty names; None if malformed."""
    if isinstance(score, dict):
        if not score or not all(isinstance(name, str) and name and _finite(x) for name, x in score.items()):
            return None
        return {name: float(x) for name, x in score.items()}
    return float(score) if _finite(score) else None


def parse_stages(raw: Any, turns: list[tuple[int, int]], num_nodes: int) -> ParsedStages | None:
    """The record's stages, with the reward of each entered intermediate stage
    and the handoff fields of each completed one, or None if malformed."""
    if not isinstance(raw, list) or not all(isinstance(entry, dict) and "name" in entry for entry in raw):
        return None
    total = sum(length for _, length in turns)
    names = [str(entry["name"]) for entry in raw]
    boundaries: list[int] = []
    rewards: list[float] = []
    values: list[float] = []
    scores: list[Score | None] = []
    pools: list[str | None] = []
    for entry in raw:
        node, token = entry.get("node"), entry.get("token")
        completed = node is not None or token is not None
        reward = entry.get("reward")
        if reward is None:
            reward = float(completed)
        if not _finite(reward):
            return None
        rewards.append(float(reward))
        value, score, pool = entry.get("value"), entry.get("score"), entry.get("pool")
        if not completed:
            if value is not None or score is not None or pool is not None:
                return None  # a failed stage hands nothing off
            break  # first uncompleted stage: later stages are unreached
        if value is not None and score is not None:
            return None  # a handoff is either forecast by the env or fitted from its score, not both
        if pool is not None and (score is None or not isinstance(pool, str)):
            return None
        if value is None:
            value = 0.0
        if not _finite(value):
            return None
        if score is not None:
            score = _parse_score(score)
            if score is None:
                return None
        values.append(float(value))
        scores.append(score)
        pools.append(pool)
        if node is not None:
            node = int(node)
            if not 0 <= node < num_nodes:
                return None
            end = sum(length for index, length in turns if index <= node)
        else:
            end = min(max(int(token), 0), total)
        # A stage is only checked after its prerequisite, so it cannot end earlier.
        boundaries.append(max(end, boundaries[-1] if boundaries else 0))
    return ParsedStages(names, boundaries, rewards, values, scores, pools)


def stage_records(
    traces: list[vf.Trace], stage_key: str, success_threshold: float
) -> tuple[list[StageRecord], list[str]] | None:
    """Every trace's record plus the group's stage names, or None (reason
    logged) when the group has no consistent stage record to credit by."""
    records: list[StageRecord] = []
    stage_names: list[str] | None = None
    for trace in traces:
        turns = sampled_turns(trace)
        parsed = parse_stages(trace.info.get(stage_key), turns, len(trace.nodes))
        if parsed is None:
            return _unusable("missing or malformed stage record")
        if stage_names is None:
            stage_names = parsed.names
        elif parsed.names != stage_names:
            return _unusable("stage lists differ within the group")
        succeeded = trace.reward >= success_threshold
        if succeeded and len(parsed.boundaries) < len(parsed.names):
            return _unusable("task succeeded without completing every recorded stage")
        num_tokens = sum(length for _, length in turns)
        records.append(
            StageRecord(
                parsed.boundaries,
                num_tokens,
                trace.reward,
                parsed.rewards,
                parsed.values,
                parsed.scores,
                parsed.pools,
            )
        )
    assert stage_names is not None
    return records, stage_names


def _unusable(reason: str) -> None:
    get_logger().debug(f"temporal_grpo: plain GRPO credit for group ({reason})")


def grpo_credit(traces: list[vf.Trace]) -> None:
    """Plain group-mean GRPO credit."""
    mean = sum(trace.reward for trace in traces) / len(traces)
    for trace in traces:
        assign_advantages(trace, trace.reward - mean)


def stage_advantages(
    records: list[StageRecord],
    num_intermediate: int,
    *,
    std_normalize: bool = False,
    trajectory_weight: float = 0.0,
    eps: float = 1e-6,
) -> list[list[float]]:
    """Per-token advantages for each rollout under stage-conditioned comparison.

    Stage k (1-based, k <= num_intermediate) is entered by rollouts that
    completed k-1 stages; its outcome is whether they completed k. The final
    stage K = num_intermediate + 1 is entered by rollouts that completed every
    intermediate stage; its outcome is the task reward. A stage whose entrants
    all share an outcome carries no ranking signal and contributes zero.
    """
    num_stages = num_intermediate + 1
    stage_adv: list[dict[int, float]] = [{} for _ in records]
    for k in range(1, num_stages + 1):
        entrants = [i for i, r in enumerate(records) if len(r.boundaries) >= k - 1]
        if len(entrants) < 2:
            continue
        if k < num_stages:
            outcomes = [1.0 if len(records[i].boundaries) >= k else 0.0 for i in entrants]
        else:
            outcomes = [records[i].reward for i in entrants]
        mean = sum(outcomes) / len(outcomes)
        if all(o == outcomes[0] for o in outcomes):
            continue
        scale = 1.0
        if std_normalize:
            scale = (sum((o - mean) ** 2 for o in outcomes) / len(outcomes)) ** 0.5 + eps
        for i, outcome in zip(entrants, outcomes, strict=True):
            stage_adv[i][k] = (outcome - mean) / scale

    rewards = [r.reward for r in records]
    trajectory_mean = sum(rewards) / len(rewards)
    out: list[list[float]] = []
    for i, record in enumerate(records):
        tokens = [0.0] * record.num_tokens
        completed = len(record.boundaries)
        for k in range(1, min(completed + 1, num_stages) + 1):
            start = record.boundaries[k - 2] if k >= 2 else 0
            # Completed intermediate stage: up to its boundary. Failed or final
            # stage: the whole remaining suffix.
            end = record.boundaries[k - 1] if k <= completed and k < num_stages else record.num_tokens
            value = stage_adv[i].get(k, 0.0)
            for t in range(start, end):
                tokens[t] = value
        if trajectory_weight:
            trajectory = record.reward - trajectory_mean
            tokens = [(1.0 - trajectory_weight) * v + trajectory_weight * trajectory for v in tokens]
        out.append(tokens)
    return out


class TemporalGRPOAlgorithm(Algorithm):
    """Stage-conditioned GRPO. Groups whose traces lack a consistent stage
    record fall back to plain group-mean GRPO credit."""

    def __init__(self, config: TemporalGRPOAlgoConfig, clients: InferenceClient):
        super().__init__(config, clients)
        self.stage_key = config.stage_key
        self.std_normalize = config.std_normalize
        self.trajectory_weight = config.trajectory_weight
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
            trajectory_weight=self.trajectory_weight,
        )
        for trace, values in zip(traces, per_token, strict=True):
            assign_advantages(trace, values)
