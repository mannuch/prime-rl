"""Stage-conditioned credit (Temporal GRPO, arXiv:2608.13026)."""

import asyncio
from unittest.mock import MagicMock

import pytest
import verifiers.v1 as vf
from verifiers.v1.types import AssistantMessage, UserMessage

from prime_rl.configs.algorithm import TemporalGRPOAlgoConfig
from prime_rl.orchestrator.algo.temporal_grpo import (
    StageRecord,
    StageRecordError,
    TemporalGRPOAlgorithm,
    parse_stages,
    stage_advantages,
)
from tests.unit.orchestrator.test_algorithms import _node
from tests.unit.orchestrator.test_train_sink import received, train_trace, whole_episode


def spans(values):
    """Collapse per-token advantages into (value, run_length) pairs."""
    out = []
    for v in values:
        if out and out[-1][0] == pytest.approx(v):
            out[-1][1] += 1
        else:
            out.append([v, 1])
    return [(round(v, 4), n) for v, n in out]


def rejects(raw, turns, num_nodes):
    """The detail of the malformed-record error parse_stages raises."""
    with pytest.raises(StageRecordError) as error:
        parse_stages(raw, turns, num_nodes)
    assert error.value.reason == "malformed_record"
    return str(error.value)


def test_parse_nodes_and_prerequisites():
    # Sampled assistant nodes at trace.nodes positions 1, 3, 5, 7 (prompts/tool results in between).
    turns = [(1, 10), (3, 20), (5, 30), (7, 40)]
    raw = [{"name": "a", "node": 3}, {"name": "b", "node": 1}, {"name": "c", "node": None}, {"name": "d", "node": 7}]
    # "b" cannot end before its prerequisite; "d" is unreached because "c" never completed.
    # Without a reward field, a stage pays its binary outcome; without handoff fields, it hands off 0.
    expected = (["a", "b", "c", "d"], [30, 30], [1.0, 1.0, 0.0], [0.0, 0.0], [None, None], [None, None])
    assert parse_stages(raw, turns, num_nodes=9) == expected


def test_parse_rejects_bad_turn():
    assert rejects([{"name": "a", "node": 9}], [(1, 5)], num_nodes=3)
    assert rejects([{"name": "a", "node": 1, "reward": "x"}], [(1, 5)], num_nodes=3)
    assert rejects([{"name": "a", "node": 1, "value": "x"}], [(1, 5)], num_nodes=3)
    assert rejects([{"name": "a", "node": 1, "score": "x"}], [(1, 5)], num_nodes=3)
    # A non-finite number would poison every advantage in the group.
    for field in ("reward", "value", "score"):
        assert rejects([{"name": "a", "node": 1, field: float("nan")}], [(1, 5)], num_nodes=3)
        assert rejects([{"name": "a", "node": 1, field: float("inf")}], [(1, 5)], num_nodes=3)
    assert rejects(None, [(1, 5)], num_nodes=3)


def test_parse_rejects_bad_handoff():
    def parse(**fields):
        return parse_stages([{"name": "a", "node": 1, **fields}], [(1, 5)], num_nodes=3)

    def rejected(**fields):
        return rejects([{"name": "a", "node": 1, **fields}], [(1, 5)], num_nodes=3)

    # A failed stage hands nothing off.
    assert rejects([{"name": "a", "node": None, "value": 0.5}], [(1, 5)], num_nodes=3)
    assert rejects([{"name": "a", "node": None, "score": 0.5}], [(1, 5)], num_nodes=3)
    # A handoff is forecast by the env or fitted from its score, not both.
    assert rejected(value=0.5, score=0.5) == "stage 'a' carries both a value and a score"
    # A pool key shares a score's fit, so it needs a score, and it must be a string.
    assert rejected(pool="p")
    assert rejected(score=0.5, pool=3)
    assert parse(score=0.5, pool="p").pools == ["p"]
    # A score may be several named features, each a finite number.
    assert parse(score={"tests": 3, "coverage": 0.5}).scores == [{"tests": 3.0, "coverage": 0.5}]
    for bad in ({}, {"tests": "x"}, {"": 1.0}, {"tests": float("nan")}):
        assert rejected(score=bad)


def test_parse_token_offsets():
    raw = [{"name": "a", "token": 12, "reward": 0.4, "value": 0.5}, {"name": "b", "token": 999, "score": 3}]
    expected = (["a", "b"], [12, 50], [0.4, 1.0], [0.5, 0.0], [None, 3.0], [None, None])
    assert parse_stages(raw, [(1, 50)], num_nodes=2) == expected


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


def _forked_trace(stages):
    """A shared answer A (tokens 2, 3), then a fork into B1 (4, 5) and B2 (6)."""
    nodes = [
        _node(UserMessage(content="U"), parent=None, sampled=False, token_ids=[1]),
        _node(AssistantMessage(content="A"), parent=0, sampled=True, token_ids=[2, 3]),
        _node(AssistantMessage(content="B1"), parent=1, sampled=True, token_ids=[4, 5]),
        _node(AssistantMessage(content="B2"), parent=1, sampled=True, token_ids=[6]),
    ]
    trace = vf.Trace(
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt=None)),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=nodes,
        rewards={},
        ok=True,
    )
    trace.info["stages"] = stages
    return trace


def test_stage_credit_reaches_each_branch():
    """The env worker parses the record against the graph; the orchestrator credits each
    branch's slice of the sampled-token stream, the shared answer training in the first
    branch only."""
    algorithm = TemporalGRPOAlgorithm(TemporalGRPOAlgoConfig(), MagicMock())
    traces = [_forked_trace([{"name": "a", "node": 1}]), _forked_trace([{"name": "a", "node": None}])]
    episodes = [whole_episode([trace]) for trace in traces]
    for episode in episodes:
        asyncio.run(algorithm.score_episode(episode))
    heads = [received(episode) for episode in episodes]
    asyncio.run(algorithm.score_group(heads))

    [b1, b2] = (
        next(b.index for b in traces[0].branches if b.nodes[-1].message.content == leaf) for leaf in ("B1", "B2")
    )
    credited = [
        {
            (sample.branch_index, int(token_id)): float(advantage)
            for sample in train_trace(head.traces[0])
            for token_id, trainable, advantage in zip(sample.token_ids, sample.mask, sample.advantages, strict=True)
            if trainable
        }
        for head in heads
    ]
    # Completing stage "a" earns +0.5 on A; the final stage has one entrant, so no signal.
    assert credited[0] == {(b1, 2): 0.5, (b1, 3): 0.5, (b1, 4): 0.0, (b1, 5): 0.0, (b2, 6): 0.0}
    assert credited[1] == {(b1, 2): -0.5, (b1, 3): -0.5, (b1, 4): -0.5, (b1, 5): -0.5, (b2, 6): -0.5}
    assert all(head.traces[0].metrics["temporal_grpo/fallback"] == 0.0 for head in heads)
