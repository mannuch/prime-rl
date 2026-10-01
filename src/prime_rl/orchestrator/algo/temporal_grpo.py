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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import verifiers.v1 as vf

from prime_rl.configs.algorithm import TemporalGRPOAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm, iter_trainable_traces
from prime_rl.orchestrator.algo.routing import assign_advantages, trainable_sampled_nodes
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.orchestrator.clients import InferenceClient


@dataclass
class StageRecord:
    """One rollout reduced to what the credit rule needs."""

    boundaries: list[int]  # exclusive sampled-token end offset of each completed intermediate stage
    num_tokens: int  # sampled tokens in the rollout
    reward: float  # final-stage outcome (the task reward)


def sampled_turns(trace: vf.Trace) -> list[tuple[int, int]]:
    """(index into ``trace.nodes``, sampled-token count) per trainable sampled
    node, in the order ``assign_advantages`` walks."""
    return [(index, sum(node.mask)) for index, node in trainable_sampled_nodes(trace)]


def parse_stages(raw: Any, turns: list[tuple[int, int]], num_nodes: int) -> tuple[list[str], list[int]] | None:
    """Stage names and completed-stage token boundaries, or None if malformed."""
    if not isinstance(raw, list) or not all(isinstance(entry, dict) and "name" in entry for entry in raw):
        return None
    total = sum(length for _, length in turns)
    names = [str(entry["name"]) for entry in raw]
    boundaries: list[int] = []
    for entry in raw:
        if entry.get("node") is not None:
            node = int(entry["node"])
            if not 0 <= node < num_nodes:
                return None
            end = sum(length for index, length in turns if index <= node)
        elif entry.get("token") is not None:
            end = min(max(int(entry["token"]), 0), total)
        else:
            break  # first uncompleted stage: later stages are unreached
        # A stage is only checked after its prerequisite, so it cannot end earlier.
        boundaries.append(max(end, boundaries[-1] if boundaries else 0))
    return names, boundaries


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

    def _fallback(self, traces: list[vf.Trace], reason: str) -> None:
        get_logger().debug(f"temporal_grpo: plain GRPO credit for group ({reason})")
        mean = sum(trace.reward for trace in traces) / len(traces)
        for trace in traces:
            assign_advantages(trace, trace.reward - mean)

    async def score_group(self, episodes: list[vf.Episode]) -> None:
        traces = [trace for _, trace in iter_trainable_traces(episodes)]
        if not traces:
            return
        records: list[StageRecord] = []
        stage_names: list[str] | None = None
        for trace in traces:
            turns = sampled_turns(trace)
            parsed = parse_stages(trace.info.get(self.stage_key), turns, len(trace.nodes))
            if parsed is None:
                return self._fallback(traces, "missing or malformed stage record")
            names, boundaries = parsed
            if stage_names is None:
                stage_names = names
            elif names != stage_names:
                return self._fallback(traces, "stage lists differ within the group")
            succeeded = trace.reward >= self.success_threshold
            if succeeded and len(boundaries) < len(names):
                return self._fallback(traces, "task succeeded without completing every recorded stage")
            records.append(StageRecord(boundaries, sum(length for _, length in turns), trace.reward))

        assert stage_names is not None
        per_token = stage_advantages(
            records,
            len(stage_names),
            std_normalize=self.std_normalize,
            trajectory_weight=self.trajectory_weight,
        )
        for trace, values in zip(traces, per_token, strict=True):
            assign_advantages(trace, values)
