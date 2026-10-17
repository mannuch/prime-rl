# Algorithms

This page covers the math and the configurable algorithmic components: the algorithm abstraction and its algorithms, how off-policy training works, the loss components and advantage functions, curricula, and how multi-turn rollouts get merged into training samples.

## Table of Contents

- [The Algorithm Abstraction](#the-algorithm-abstraction)
  - [Model References](#model-references)
  - [The Algorithms](#the-algorithms)
  - [Customizing Components](#customizing-components)
  - [Per-Env Algorithms](#per-env-algorithms)
  - [The Algorithm Classes](#the-algorithm-classes)
- [Async / Off-Policy Training](#async--off-policy-training)
- [Loss](#loss)
  - [Loss Components](#loss-components)
  - [IPO Loss](#ipo-loss)
  - [Custom Loss](#custom-loss)
- [Advantage](#advantage)
  - [Default Advantage](#default-advantage)
  - [Hierarchical GRPO](#hierarchical-grpo)
  - [Self-Play Advantage (RAE)](#self-play-advantage-rae)
  - [Authoring an Algorithm](#authoring-an-algorithm)
  - [Reference Scoring](#reference-scoring)
- [Curricula](#curricula)
- [Multi-Turn Trajectories](#multi-turn-trajectories)
  - [Extension Property](#extension-property)
  - [Best-Effort Interleaving](#best-effort-interleaving)
  - [Renderers](#renderers)
  - [Discontinuous Trajectories](#discontinuous-trajectories)

## The Algorithm Abstraction

A training algorithm in `prime-rl` is configured under `[orchestrator.train.algo]`, where **`type` names the algorithm** (`grpo`, `opd`, `sft`, …) and the class defaults are its vetted setting. It has two parts:

1. **Sampling** (`algo.sampling`) — how train rollouts are produced: which model generates them. `source` is a [model reference](#model-references): `"policy"` (the live policy, the default) or an inline frozen hosted model. Group sizing stays on the env config (`group_size`).
2. **The per-token training signal** — credit assignment and loss routing, fused; the algorithm's own parameters sit directly on `algo`. One mapping from a finalized rollout to per-token *(loss component, weight)* pairs — the credit a token gets and the loss that consumes it are two coordinates of the same output. Group-relative algorithms compute credit on the orchestrator and ship per-token advantage streams; reference-KL algorithms query a reference model at batch-ship time (bounded concurrency) and ship its prefill logprobs for the trainer to evaluate against the live policy. The `type` determines which loss component consumes the action tokens (`rl` / `ce` / `ref_kl`) and what happens to env-provided observation tokens in multi-turn rollouts (masked out by default; `echo` trains on them with weighted CE).

The trainer is algorithm-blind: the loss is a sum of three components (rl, ce, ref_kl), each normalized by its own global token count; per-token streams ship on the wire (the `rl_weights` / `ce_weights` / `ref_kl_weights` component weights plus the `advantages` stream on each training sample) and the trainer just executes them. Adding an algorithm never touches the dispatcher, batch packing, or trainer hot path.

### Model References

`prime-rl` hosts exactly one model: the trainable policy (`[orchestrator.model]`). Every other model an algorithm uses is an external OpenAI-compatible endpoint, declared *inline on the component that uses it*. A model reference is either the string `"policy"` (the live policy) or a frozen hosted model (`name` + `base_url`):

```toml
[orchestrator.train.algo]
type = "opd"

[orchestrator.train.algo.teacher]   # opd's teacher: the frozen model it scores against
name = "Qwen/Qwen3-32B"
base_url = "http://localhost:8001/v1"
```

Model *roles* are algorithm-local vocabulary — each algorithm names its reference on the field where the model is actually used, and there is no shared `teacher` slot. `opd` declares a `teacher` field (the frozen model whose reverse KL the policy distills toward); `sft`'s teacher *is* its `sampling.source` (the frozen model it imitates); `opsd` self-distills against the live policy and names no model at all. No role exists outside the algorithm that declares it: the dispatcher, sink, and trainer branch on liveness alone, never on what an algorithm calls a model.

So for `opd` set `[orchestrator.train.algo.teacher]`; for `sft` set `[orchestrator.train.algo.sampling.source]`; `opsd` needs neither. `opd`'s teacher must be a frozen endpoint — it is typed `FrozenModelConfig`, so `"policy"` isn't representable (the KL would be identically zero); `opsd`'s teacher *is* the live policy by definition (self-distillation conditioned on a demonstration), so it exposes no reference to configure.

Liveness is a property of the reference, not of any role: rollouts sampled from `"policy"` get version-salted prefix caches, carry sampling logprobs for importance ratios, and age off-policy as weights update; rollouts and scores from frozen models get a stable prefix cache and never go stale. Frozen models are externally hosted (`base_url` is required) — `prime-rl` never launches or updates them, and each env's algorithm builds its own client pool to the endpoints it declares.

### The Algorithms

The `algo.type` names the algorithm, and each type's class defaults are its vetted setting — picking a type with no other keys IS the algorithm:

```toml
[orchestrator.train.algo]
type = "grpo"  # the default
```

| `type` | Sampling | Loss | What it is |
|---|---|---|---|
| `grpo` | policy | `rl` on actions | Standard group-relative RL. |
| `max_rl` | policy | `rl` on actions | MaxRL ([arXiv:2602.02710](https://arxiv.org/abs/2602.02710)): GRPO's centered reward normalized by the group **mean** instead of the standard deviation — the gradient is unbiased for the order-`group_size` truncation of the maximum-likelihood objective, upweighting hard examples like `1/p`. |
| `rae` | policy | `rl` on actions | RAE (SPIRAL, [arXiv:2506.24119](https://arxiv.org/abs/2506.24119)): reward minus a per-agent EMA baseline of that agent's own rewards — the estimator for multi-agent self-play envs, where the group mean would mix the agents' opposite reward scales. See [Self-Play Advantage](#self-play-advantage-rae). |
| `hierarchical_grpo` | policy | `rl` on actions | GRPO for proposer-solver envs. Solvers are compared only with attempts on the same proposed problem; proposers are compared with the other proposals in the group. See [Hierarchical GRPO](#hierarchical-grpo). |
| `opd` | policy | `ref_kl` on actions | On-policy distillation ([Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/)): the policy samples, per-token reverse KL against a reference model as the gradient signal. Needs a `teacher`. |
| `sft` | *(the teacher)* | `ce` on actions | Hard distillation: a frozen model generates rollouts, the policy trains with CE on its tokens. Needs a frozen `sampling.source` (the teacher it samples from). |
| `opsd` | policy | `ref_kl` on actions | SDFT ([arXiv:2601.19897](https://arxiv.org/abs/2601.19897)): the model is its own reference, conditioned on an expert demonstration. The teacher *is* the live policy (the paper's setting, no extra deployment) — no model to configure. |
| `echo` | policy | `rl` on actions + weighted `ce` on observations | ECHO: standard GRPO plus a cross-entropy loss on env-provided tokens already present in the rollout, selected by message role (needs the renderer's role attribution). Defaults to tool-response bodies at `alpha = 0.1` (ECHO's λ); set `roles` to train other roles, each at its own weight. |

### Customizing Components

Every key beyond `type` is visibly your own assembly — there is no preset layer to diverge from. The vetted setting is the class defaults; what you set is what runs:

```toml
# echo on tool AND user feedback tokens, each at its own weight.
# Setting any role replaces the whole table.
[orchestrator.train.algo]
type = "echo"

[orchestrator.train.algo.roles.tool]
alpha = 0.25

[orchestrator.train.algo.roles.user]
alpha = 0.05
```

A new algorithm is a named class in code, not a config that points at an import path — see [Authoring an Algorithm](#authoring-an-algorithm).

Echo also takes an optional user-supplied token filter that narrows the role selection per rollout — e.g. dropping warning lines from tool output, or tokens the sampler found unlikely:

```toml
[orchestrator.train.algo.filter]
import_path = "my_module.drop_warnings"

[orchestrator.train.algo.filter.kwargs]
patterns = ["WARNING"]
```

```python
# my_module.py — sees the raw rollout (message text, sampling logprobs);
# returns one keep-mask per trainable branch, spanning that branch's
# token_ids. False = never echo-trained.
def drop_warnings(rollout, *, patterns: list[str]) -> list[list[bool]]: ...
```

Component compatibility is validated at config time: frozen-model sampling can only feed the `ce` loss component — the `rl` and `ref_kl` components need the live policy's own sampling logprobs for importance ratios — `opd` pointed at `"policy"` is rejected as degenerate (zero KL), `sft` without a frozen source is rejected (CE on the policy's own tokens is not a distillation target). A group-relative algorithm with `group_size = 1` produces all-zero advantages; the resulting empty batch is caught at runtime (the orchestrator warns and skips it), not at config time.

### Per-Env Algorithms

Both components resolve per environment. Each env inherits `[orchestrator.train.algo]` like any other group default: an env that sets only some params keeps the group's algorithm with those params changed, and an env that sets a different `type` runs its own algorithm. So a single run can mix algorithms across envs — e.g. GRPO on math, ECHO on a terminal env:

```toml
[orchestrator.train.algo]
type = "grpo"

[[orchestrator.train.source]]
name = "math"  # inherits the group's grpo
env.taskset.id = "math"
env.agent.harness.id = "null"
env.agent.runtime.type = "subprocess"

[[orchestrator.train.source]]
name = "terminal"
env.taskset.id = "terminal"
env.agent.harness.id = "bash"
env.agent.runtime.type = "subprocess"
# this env runs its own algorithm
algo.type = "echo"
```

### The Algorithm Classes

At runtime, each env's `TrainEnv` (`prime_rl.orchestrator.envs`) resolves the `sampling.source` model into the inference pool used for train episodes, and its resolved config builds one of the named algorithm classes in `prime_rl.orchestrator.train.algo` (one module per algorithm: `algo/grpo.py`, `algo/opd.py`, …) from the algorithm config. Algorithm dispatch is keyed on `algo.type` — it names the algorithm, and each config class's defaults are its vetted parameterization:

| `algo.type` | Class | hook(s) — stage |
|---|---|---|
| `grpo` | `GRPOAlgorithm` | `score_group`: group-norm credit (optional length penalty) |
| `echo` | `EchoAlgorithm` | `score_episode`: weighted ce on observation tokens; `score_group`: group-norm credit (inherited) |
| `max_rl` | `MaxRLAlgorithm` | `score_group`: mean-normalized group credit |
| `rae` | `RAEAlgorithm` | `score_group`: per-agent EMA-baseline credit |
| `hierarchical_grpo` | `HierarchicalGRPOAlgorithm` | `score_group`: GRPO baseline per episode for solvers, per group for the proposer |
| `temporal_grpo` | `TemporalGRPOAlgorithm` | `score_group`: stage-conditioned credit on stage token intervals |
| `temporal_grpo_continuous` | `TemporalGRPOContinuousAlgorithm` | `score_group`: stage TD credit on stage token intervals |
| `opd` | `OPDAlgorithm` | `score_episode`: own-context prefill under the teacher |
| `opsd` | `OPSDAlgorithm` | `score_episode`: demo-conditioned prefill under the live policy |
| `sft` | `SFTDistillAlgorithm` | no credit assignment; CE on sampled tokens |

Algorithms operate on native verifier artifacts and annotate their message graphs directly:

- `async score_episode(episode)` — rollout-local scoring as one episode arrives.
- `async score_group(episodes)` — group-relative scoring after the cohort completes.

The pipeline calls `score_group` directly and `score_episode` through `finalize_episode`, which skips episodes with no trainable sampled tokens. Advantages, reference logprobs, and named loss weights stay on verifier nodes through admission. Only admitted traces are flattened into `TrainingSample`s.

The algorithm config states which loss component its action tokens feed (`action_loss_type`, a class variable on each config class). Every class is constructed with its algorithm config plus the one host-owned resource it can't rebuild — the live policy clients (`self.clients`). Everything else an algorithm needs it builds from its own config in `setup()`: `opd` connects its frozen `teacher`; `opsd` builds the renderer for its demonstration hint (tokenizer is always the live policy's — self-distillation has no separate model). Writing your own algorithm is subclassing `Algorithm` and overriding the hooks its signal needs (see [Authoring an Algorithm](#authoring-an-algorithm)). Shared math (efficiency shaping, prefill alignment) lives as plain functions in `prime_rl.orchestrator.train.algo.advantage`.

## Async / Off-Policy Training

`prime-rl` is asynchronous by default. The trainer and inference always run one step overlapped: while the trainer is producing $\pi_n$ from rollouts at step $n$, inference is already generating the rollouts for step $n+1$ using $\pi_{n-1}$. With matched trainer and inference step times this produces fully-overlapped pipeline parallelism — neither side ever idles.

![Async pipeline: trainer step n produces $\theta_n$, inference at step n samples with $\theta_{n-1}$](assets/async-pipeline.png)

At step $n = 1, 2, 3, \dots$:

- **Trainer** produces policy $\pi_n$ with weights $\theta_n$ from rollouts $(x_n, y_n)$.
- **Inference** produces rollouts $(x_n, y_n)$ from policy $\pi_{\max(0,\,n-1)}$.

Step indices are 1-indexed; policy versions are 0-indexed, with $\pi_0$ the base model. At step 1 inference samples from $\pi_0$.

## Loss

### Loss Components

The training loss is a **sum of three components**, each with its own per-token weight stream and its own normalization:

$$
\mathcal{L} = \frac{\sum \mathcal{L}_{rl}}{N_{rl}} + \frac{\sum \mathcal{L}_{ce}}{N_{ce}} + \frac{\sum \mathcal{L}_{ref\_kl}}{N_{ref\_kl}}
$$

- `rl` — the configured RL loss (`[trainer.loss]`): IPO by default, or optionally [IcePop](#icepop-loss) or a [custom loss](#custom-loss). Fed by the advantage-assigning algorithms (`grpo`, `max_rl`, `rae`, `hierarchical_grpo`, and `echo`'s action tokens).
- `ce` — masked NLL. Used for frozen-model tokens (`sft`) and env-observation tokens (`echo`).
- `ref_kl` — the per-token reverse KL to a reference model ($\log \pi_{\text{ref}} - \log \pi$) as the policy-gradient signal, importance-ratio corrected with a one-sided trust region (`opd`, `opsd`). Requires `ref_logprobs` from a [reference scoring](#reference-scoring); the scoring model must be a vLLM server (it's the only one that exposes `prompt_logprobs`).

The orchestrator stamps each sample's component membership as per-token weight streams (`rl_weights` / `ce_weights` / `ref_kl_weights` on the wire): a weight scales that component's per-token loss, `0.0` leaves the token out of the component entirely (mask *and* denominator), and components may overlap on the same token — their gradients sum. Each $N$ is the global (all-reduced) count of that component's member tokens — for `rl`, the sum of its weights, which is the same count while weights are 0/1 — so the components don't dilute each other: adding echo observation tokens never changes the rl term's effective per-token learning rate, and an sft env packed next to a GRPO env doesn't soften its gradient. Tokens of different components pack freely into the same micro batch, and a plain GRPO run ships no weight streams at all (absent streams mean rl weight 1.0 on every trainable token — the unchanged hot path). Advantages always ship per token (`advantages` on the wire), assigned as per-token streams from the start — uniform group credit is broadcast over completion tokens at assignment; algorithms with no rl credit (opd, opsd) ship none.

### IPO Loss

The default RL loss is INTELLECT Policy Optimization (IPO). It is an importance-weighted policy-gradient term. A symmetric trust region removes tokens whose absolute probability change exceeds $\epsilon$:

$$
\mathcal{L}(\theta) = -\frac{1}{N}\sum_t
\mathbb{1}\!\left(\left|\pi(y_t)-\mu(y_t)\right| \le \epsilon\right)
\tau_A \hat{A}_t \frac{\pi(y_t)}{\mu(y_t)}.
$$

$\mu$ is the policy that generated the rollout. $\pi$ is the current trainer policy. $\hat{A}_t$ is the token-level advantage. The trust region uses the sampled token probabilities, not their ratio.

The knobs under `[trainer.loss]` are:

| Knob | Default | What it does |
|---|---|---|
| `eps` | 0.3 | Maximum absolute probability change before a token is masked. |
| `adv_tau` | 1.0 | Temperature on the advantage term. |

Omit `[trainer.loss]` to use these defaults. Set `type = "ipo"` when you specify the section. The `ce` and `ref_kl` components are fixed and unaffected by `[trainer.loss]`.

See [IPO](ipo.md) for why IPO masks by probability instead of ratio, and the GLM-4.5-Air ablation against IcePop behind `eps = 0.3`.

### IcePop Loss

IcePop is an opt-in RL loss that drops tokens whose trainer-to-inference
importance ratio falls outside a fixed acceptance band, introduced to stabilize
MoE RL in [Every Step Evolves: Scaling Reinforcement Learning for Trillion-Scale
Mixture-of-Experts Reasoning Models](https://arxiv.org/abs/2510.18855). Accepted
tokens retain the importance-weighted policy-gradient term:

$$
\mathcal{L}(\theta) = -\frac{1}{N}\sum_t
\mathbb{1}\!\left(\alpha \le \frac{\pi(y_t)}{\mu(y_t)} \le \beta\right)
\tau_A \hat{A}_t \frac{\pi(y_t)}{\mu(y_t)}.
$$

Enable it explicitly:

```toml
[trainer.loss]
type = "icepop"
ratio_low = 0.2
ratio_high = 5.0
```

| Knob | Default | What it does |
|---|---|---|
| `ratio_low` | 0.2 | Lower accepted trainer-to-inference probability ratio. |
| `ratio_high` | 5.0 | Upper accepted trainer-to-inference probability ratio. |
| `adv_tau` | 1.0 | Temperature on the advantage term. |

### Custom Loss

`[trainer.loss] type = "custom"` replaces the `rl` component. The loss is computed **per sequence**: you write a function that takes one sequence's tensors and returns a scalar loss. The trainer iterates and aggregates. `inputs.loss_mask` selects exactly the rl member tokens (for a plain GRPO run, all trainable tokens).

```python
# my_module.py
import torch
from prime_rl.trainer.rl.loss import LossInputs, LossOutputs

def ppo_clip_loss(inputs: LossInputs, clip_eps: float = 0.2) -> LossOutputs:
    ratio = torch.exp(inputs.trainer_logprobs - inputs.inference_logprobs)
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    surr1 = ratio * inputs.advantages
    surr2 = clipped * inputs.advantages
    loss = -torch.min(surr1, surr2)[inputs.loss_mask].sum()
    return LossOutputs(
        loss=loss,
        metrics={
            "clip_frac": (ratio != clipped)[inputs.loss_mask].float().mean(),
        },
    )
```

Wire it up:

```toml
[trainer.loss]
type = "custom"
import_path = "my_module.ppo_clip_loss"

[trainer.loss.kwargs]
clip_eps = 0.2
```

The dataclasses:

```python
@dataclass
class LossInputs:
    trainer_logprobs: Float[Tensor, "seq"]      # current policy
    inference_logprobs: Float[Tensor, "seq"]    # rollout-time policy
    ref_logprobs: Float[Tensor, "seq"] | None   # set by reference-scoring algorithms
    advantages: Float[Tensor, "seq"]
    loss_mask: Bool[Tensor, "seq"]              # this component's member tokens
    loss_weights: Float[Tensor, "seq"] | None   # the component's weight stream (None = 1.0)

@dataclass
class LossOutputs:
    loss: Float[Tensor, ""]
    metrics: dict[str, Tensor]
```

Anything you put in `metrics` is averaged across sequences and logged with the other trainer metrics.

## Advantage

The per-token training signal is set by `algo.type` and the [algorithm](#the-algorithm-abstraction)'s parameters — every signal is a per-token advantage stream, varying in evaluation site (orchestrator vs. trainer). The `algo.type` values:

| Type | Component | Effect |
|---|---|---|
| `grpo` | `rl` | Group-norm: reward minus per-group baseline, optional length penalty. |
| `max_rl` | `rl` | Mean-normalized group credit (maximum-likelihood RL). |
| `rae` | `rl` | Reward minus a per-agent EMA baseline (SPIRAL's role-conditioned advantage estimation) — for multi-agent self-play envs. |
| `hierarchical_grpo` | `rl` | GRPO for proposer-solver envs: solvers are compared within one proposed problem, while proposers are compared across proposals. |
| `echo` | `rl` + `ce` | Group-norm on action tokens, plus weighted CE on env-provided tokens selected by message role (each role's `alpha` is its ECHO λ), optionally narrowed by a user filter. |
| `opd` | `ref_kl` | On-policy distillation: per-token reverse KL to a reference model (`teacher`, an inline frozen hosted model), evaluated in the trainer from shipped reference logprobs. No credit — rollouts keep `advantages = None` and ship no advantage stream; `group_size` only fans out sampling. |
| `opsd` | `ref_kl` | SDFT: per-token reverse KL to a demo-conditioned reference. No credit — rollouts keep `advantages = None` and ship no advantage stream. |
| `sft` | `ce` | Cross-entropy on the sampled tokens. Assigns no advantage — trains on every sampled token. |

### Default Advantage

The default advantage is per-group reward minus a length-weighted baseline, without std normalization. For each prompt's group of `group_size` rollouts, every trainable token in rollout $i$ receives advantage $s_i - b$, where $b = \sum_i L_i s_i / \sum_i L_i$ and $L_i$ counts trainable (policy-sampled, loss-masked) tokens across all turns of rollout $i$.

This is intentionally simple — it does the right thing for most envs. Write a named algorithm class when you need group-aware shaping that depends on trajectory metadata (sub-agent rollouts, relative-rank shaping, …) — see [Authoring an Algorithm](#authoring-an-algorithm).

A **length penalty** (`length_penalty` on the `grpo`-family algorithms) can be layered on top to discourage rambling. The `linear` penalty subtracts a single `pass_rate`-scaled penalty from each reward before the GRPO baseline, combining output tokens (`num_output_tokens_weight`), input / context tokens (`num_input_tokens_weight`), and turns (`num_turns_weight`) — each normalized by the group's own max for that quantity, with `num_input_tokens_weight` and `num_turns_weight` defaulting to `0.1`.

```toml
[orchestrator.train.algo]
type = "grpo"

[orchestrator.train.algo.length_penalty]
type = "linear"
```

`grpo` and `echo` always use the **length-weighted baseline**, computed after the length penalty. With token-level loss normalization, long rollouts carry more gradient weight, so this baseline makes the per-token advantage zero-mean across the group's trainable tokens.

**Prompt-mean loss aggregation** (`loss_aggregation = "prompt"` on `grpo` and `echo`; default `"token"`). By default every `rl` token in the batch weighs the same, so prompts whose groups produce long trajectories dominate the gradient. With `"prompt"`, the loss is MiMo-V2.6's prompt-mean (Eq. 1): the mean over prompt groups $q$ of $\frac{1}{T_q}\sum_{t \in q} \mathcal{L}_{rl,t}$, where $T_q$ is the group's total trainable tokens. MiMo uses it to keep response length from growing too fast. `score_group` gives each trainable token of the group the `rl` weight $1/T_q$, so each group's weights sum to 1 and $N_{rl}$ (the summed weights) is the number of groups in the step. Zero-advantage tokens are still dropped from the `rl` component, so a group with a zero-advantage rollout keeps less than its full weight of 1. All envs with an `rl` loss in a run must use the same `loss_aggregation`.

```toml
[orchestrator.train.algo]
type = "grpo"
loss_aggregation = "prompt"
```

### Hierarchical GRPO

GRPO gives each rollout its reward minus the average reward of comparable rollouts. In an ordinary single-agent group, every rollout answers the same task, so one group average is enough.

A proposer-solver env is different. Starting from one source task, it produces several proposed problems, then runs several solver attempts on each problem:

```text
one source task
├── proposed problem A
│   ├── proposer trace
│   ├── solver attempt 1
│   └── solver attempt 2
└── proposed problem B
    ├── proposer trace
    ├── solver attempt 1
    └── solver attempt 2
```

The solver attempts for A should not be compared with the solver attempts for B: the two problems may have very different difficulty. Proposer and solver rewards should not be compared either: they measure different jobs.

`hierarchical_grpo` therefore chooses the average separately for each role:

| Trace | Compared with | Why |
|---|---|---|
| Solver | Other solver attempts on the same proposed problem | They attempted the same problem. |
| Proposer | Other proposer traces in the group | They started from the same source task and proposed alternatives. |

For example, if three solvers receive rewards `[1, 1, 0]` on one proposed problem, their average is `2/3` and their advantages are `[1/3, 1/3, -2/3]`. Solver rewards from other proposed problems do not affect those values. The proposers are scored separately according to how useful their problems were for the solvers, then compared with the other proposers in the group.

Configure which roles are compared within a single proposed problem with `episode_agents`. For `proposer-solver`, that role is `solver`:

```toml
[orchestrator.train.algo]
type = "hierarchical_grpo"
episode_agents = ["solver"]

[[orchestrator.train.source]]
name = "proposer-solver"
group_size = 4  # proposed problems per source task
env.n = 4  # solver attempts per proposed problem
env.taskset.id = "proposer-solver"
env.proposer.harness.id = "null"
env.proposer.runtime.type = "subprocess"
env.solver.harness.id = "null"
env.solver.runtime.type = "subprocess"
```

`group_size` controls how many problems are proposed from each source task. `env.n` controls how many solvers attempt each proposed problem. If a comparison contains only one trace—for example, a solver when `env.n = 1`—its advantage is zero.

This algorithm is accepted only for proposer-solver envs. Use the env's `train_proposer` and `train_solver` settings if you want to train only one role.

### Temporal GRPO

GRPO broadcasts one advantage over every sampled token. A rollout that clears three stages of a task and fails the fourth is penalized for the three it cleared. `temporal_grpo` ([arXiv:2608.13026](https://arxiv.org/abs/2608.13026)) splits each rollout into ordered stage intervals and compares, for each stage, only the rollouts that *entered* it (completed its prerequisite). Stage *k*'s advantage is `completed_k - mean` over its entrants and lands only on stage *k*'s tokens. The final stage is the task reward, over the tokens after the last intermediate stage, so the task objective is unchanged. A stage whose entrants all agree contributes zero; with `constant_trainer_batch_size` (on by default), those zero-advantage tokens are also pruned from the rl denominator. A group where every rollout fails the task but they diverge at an earlier stage still carries signal.

The env declares progress in `trace.info["stages"]`: the task's ordered intermediate stages, each marking the `trace.nodes` index of the assistant node whose action completed it (`None` if never completed), or a `token` offset for stages inside one long response. An entry may also carry a scalar `reward` for the stage and, on a completed stage, a `value` for the state it handed off; `temporal_grpo` ignores both (see [continuous stage rewards](#temporal-grpo-with-continuous-stage-rewards)). Where to write it depends on who can see progress:

- **Online, in `Env.run`** — for games and simulated-user envs, where the env steps the ground truth between turns. Right after the turn that caused the change, record the latest sampled node's index. This also catches the final move, whose observation is never sent back to the model and so never becomes a node.
- **Offline, in `Task.finalize`** — replay the transcript: the first tool result or user turn that proves a stage is credited to the assistant turn it answers (the tool call's issuer via `tool_call_id`, or a user turn's nearest sampled ancestor).

```python
# Env.run, one move per segment
done: list[int] = []  # node index that completed each stage, in order
segment = await interaction.turn()
while not segment.terminated:
    game.step(segment.last_reply)
    if len(done) < len(STAGES) and game.reached(STAGES[len(done)]):
        done.append(max(i for i, n in enumerate(interaction.trace.nodes) if n.sampled))
    ...
interaction.trace.info["stages"] = [{"name": n, "node": done[k] if k < len(done) else None} for k, n in enumerate(STAGES)]
```

```toml
[orchestrator.train.algo]
type = "temporal_grpo"
trajectory_weight = 0.0   # >0 blends back trajectory credit when early choices shape later success
```

Every trace in a group must report the same stage list; a group with a missing or inconsistent record, or a success whose record misses a stage, falls back to plain GRPO credit. The final stage covers only the tokens after the last intermediate boundary: if a rollout completes its last intermediate stage on its final turn, the task-reward comparison has no tokens to land on for that rollout. Credit each stage to the turn that caused it, and keep the task's last piece of real work out of the intermediate list.

### Temporal GRPO with continuous stage rewards

`temporal_grpo` ranks a stage's entrants by a binary outcome: did they complete it. `temporal_grpo_continuous` lets each stage pay a scalar reward instead and credits a stage with its TD error against the group's stage values. A stage's value `V_k` is the mean reward-to-go of the rollouts that entered it; stage *k*'s advantage for a rollout is `q_k + V_{k+1}·completed_k − V_k`, where `q_k` is the reward it collected in stage *k*. That is the policy gradient of the rollout's return with the group as critic, applied at every stage node (plain GRPO is the one-stage case): unnormalized, a rollout's stage advantages sum to its return minus the group mean, so this redistributes GRPO's credit to the stages that earned it rather than changing the objective. By default credit stays in reward units (`std_normalize = false`), like prime-rl's GRPO. Setting `std_normalize = true` divides each stage's TD errors by their std over the stage's entrants; with binary stage rewards that makes it exactly the paper's rule (`temporal_grpo` with `std_normalize = true`), in every group, for any task reward that is the same for every rollout failing a given stage.

The derivation, including why the TD error and what handoff values do to it, is in the module docstring of `temporal_grpo_continuous.py`.

#### Writing the record

The record is the same `trace.info["stages"]` list `temporal_grpo` reads: the task's ordered intermediate stages, each with a `name` and a completion mark (`node` or `token`, `None` if never completed). The task stage is implicit and is paid `trace.reward`. Two optional scalars on an entry shape the credit.

**`reward` answers "what did this stage earn?"** It is the piece of the return the stage is responsible for, in `trace.reward`'s units. The default, 1 if completed and 0 if not, makes completing stages the objective in groups where nobody succeeds, and is the right default for stages that are pass/fail. Override it when:

- completion is graded: a 0.7 repro and a 0.3 repro both "complete";
- a failed stage still earned something: put partial credit on the failed entry;
- stages are always completed: the default then pays everyone 1 and says nothing.

Stage rewards add to `trace.reward`; the return this rule optimizes is their sum. Don't also fold them into the rubric, or they count twice. `success_threshold` reads `trace.reward` alone. A completed stage with an explicit `reward` of 0 still earns the next stage's group value: completion is structural, the reward is only the scalar.

**`value` answers "from the state this stage handed off, how much is still to be earned?"** It is a forecast, not a grade: for a success-or-nothing task, the probability of eventual success from here times the task reward, plus any stage rewards still ahead. It goes only on completed entries; a `value` on a failed stage makes the record malformed, since a failed stage hands nothing off. The stage is paid its `value` and the next stage is charged it, so it moves credit between neighbours and never changes the total: each stage is judged on how much it improved the outlook it was handed, not on what it inherited. Reward measures the stage, value measures the state after it; a stage that did its work but left a mess has high reward and low value.

Reach for `value` when a stage can complete well or badly in ways the next stage feels (a grasp that holds but is off-centre, a repro that fails on the wrong assertion), or when stages are always completed, so only a quality signal can tell their tokens apart. Without it, the rule prices every handoff out of stage *k* at the same `V_{k+1}` and can't see those differences. A `value` must be computed from the handoff state alone and calibrated to reward units; see [Designing a handoff signal](#designing-a-handoff-signal).

A record for reproduce → patch → verify, with `trace.reward = 0`:

```python
interaction.trace.info["stages"] = [
    {"name": "reproduce", "node": 4, "value": 0.6},
    {"name": "patch", "node": 11, "reward": 0.8, "value": 0.9},
    {"name": "verify", "node": None, "reward": 0.2},
]
```

reproduce completed (reward 1 by default) and left a state worth 0.6; patch completed with quality 0.8 and raised the outlook to 0.9; verify failed, earning 0.2, and the task reward 0 lands there. After shaping the stages pay 1.6, 1.1 and −0.7: verify is charged for the 0.9 outlook it squandered. The total, 2.0, is the raw sum, as it always is.

**`score` is a `value` the algorithm calibrates for you.** Most env authors have *measurements* of a handoff (the diff size, how many visible tests pass, the grasp's offset from centre) but not *forecasts* in reward units. Put the measurement in `score`, on any scale and in either direction. Within each group, the algorithm regresses what actually happened after the stage on it, and uses the slope to turn the score into a value. A completed entry carries `value` or `score`, never both, and within a group a stage's completers must all carry a `score` or none of them. Three things keep a bad score from doing harm:

- a slope that's within about a standard error of zero is shrunk toward zero (`score_shrinkage`), which means no shaping, i.e. the rule without a handoff signal;
- each rollout's slope is fit without its own outcome;
- whatever the slope, shaping only moves credit between a rollout's own stages; it never changes the rollout's total.

A group has 8–16 rollouts, so the fit is noisy. To pool it, give the entry a `pool` key: entries that share a key share one fit, across the stages of a group and across groups, remembered over about `1 / (1 − score_decay)` groups, the current one included (`score_decay = 0` keeps no history). Stages under one key in a group fit from all their completers together; since a rollout's points at those stages share its outcome, the fit leaves out all of them when crediting that rollout, and its standard error counts rollouts, not points. What a key means is your call, and nothing else is pooled. Use the task's identity in the key (`f"{task_id}/grasp"`) to pool one task with itself across epochs; use one key for a family of tasks only when you're confident the same score predicts the same thing across them. Pooling trades noise for staleness: a pooled fit remembers how an older policy turned handoffs into return (see [Avoiding stale fits](#avoiding-stale-fits)). The pooled state lives in orchestrator memory and starts empty after a restart, so the first `1 / (1 − score_decay)` groups under each key fit from themselves alone.

Each scored stage logs `stage/<name>/score_slope` and `stage/<name>/score_slope_se` (from the group alone), plus `stage/<name>/score_slope_pooled`, the key's shared fit including earlier groups, when it has a key. A slope within about one standard error of zero means the score isn't telling the rule much; a confidently negative slope means a higher score predicts *worse* outcomes downstream, which usually means the stage is being gamed.

Write the record where you write the completion marks (`Env.run` online or `Task.finalize` offline, as described for `temporal_grpo`).

#### Using a `score` instead of a `value`

Start a stage's handoff signal with a `score`, not a `value`. A `value` has to be calibrated: its differences must equal differences in expected return, which an env that sees one rollout at a time rarely knows. A miscalibrated `value` distorts credit, and `std_normalize` won't correct it. A `score` only has to measure; the group calibrates it.

**The handoff diagnostic.** Watch each stage's mean reward, and the next stage's mean reward among the rollouts that entered it. If the first rises while the second falls, the stage is getting better at its own job in a way that hurts the next one: a handoff effect the record isn't expressing yet, which a `score` or `value` is the place to express.

Add handoff signals on top of the stage rewards in this order, stopping once the handoff diagnostic stops firing:

1. No handoff signal: stage rewards and completion marks alone. This is the rule's base case and often enough. Write a `reward` wherever a stage's quality is graded; pass/fail stages can keep the default of 1 or 0.
2. `score`, fit per group.
3. If the logged slope is noisy: a more informative score, larger groups, or a `pool` key.
4. `value`, calibrated offline from the logged slopes, so you stop paying for the per-group noise.

**Start directly with `value`** only when any of these holds. A `value` costs no estimation noise and works from a single completer, so when you can write a calibrated one, it's the better signal:

- **You can compute the forecast.** The env sees the state well enough to know what's left: a simulator with privileged state and a known success model, or remaining work that's fixed arithmetic (say, the stage rewards still ahead, each with a known pass rate from this state).
- **You can calibrate offline.** You have logged rollouts from an earlier run, or the logged `score_slope`s from a `score`-based one, and can fit the mapping from your measurement to downstream return once, outside training.
- **The stage is rarely completed by many rollouts in a group.** A per-group fit needs at least 4 completers of the stage before it produces a slope at all (each rollout's fit leaves itself out and needs 3 points), so a late stage that 2 or 3 rollouts in a group reach gets no shaping from a `score` unless it has a `pool` key with history.
- **Several measurements matter and you know how they combine.** A fit uses one score. If the handoff depends on, say, both test count and diff size in a known way, combine them into a value yourself rather than collapsing them into one uncalibrated number.

**When every stage always completes**, the default is still a `score`, but checked sooner. A `score` is easiest to fit here: every rollout completes every stage, so each stage's fit uses the whole group and the few-completers reason to start with a `value` never applies. It also matters more. With no failures there is no progression signal to fall back on, so a `score` that shrinks to zero leaves per-stage GRPO, and with default stage rewards only the final segment trains. Three adjustments:

1. **Check whether a stricter completion check would split the rollouts.** A stage that always completes may have too loose a predicate. Tightening it ("holds through a shake", not "gripper closed") turns quality into the gate and restores discrete conditioning with no calibration at all; a `score` can still sit on top.
2. **Score the handoff state, not the stage's work.** With every rollout a completer, a `score` that is a linear function of the stage's `reward` merges into the quality term, and `std_normalize` cancels it; the same holds for a `value`. Measure where things ended up, not how well the stage scored.
3. **Look at the logged slope early**, and move to a `pool` key, larger groups or an offline-calibrated `value` sooner than you would when stages can fail.

#### Designing a handoff signal

A stage's `reward` and its handoff signal (`score` or `value`) are both numbers on the stage's entry, but they do different jobs:

| | `reward` | `score` / `value` |
|---|---|---|
| Answers | What did this stage earn? | How much is still to be earned from the state it handed off? |
| Effect on the return | Adds to it; the policy is trained to collect it | None; moves credit from the next stage to this one, and the rollout's total is unchanged |
| On a failed stage | Allowed, as partial credit | Not allowed; a failed stage hands nothing off |
| If it's wrong | The policy optimizes the wrong thing | Credit for the right objective lands on the wrong stage |

To decide where a quantity goes, ask whether you'd pay for it even if it changed nothing downstream. A clean patch, a fast solve or a well-formed answer: yes, so it's a `reward`. Visible tests passing after a patch or a grasp's offset from centre: you care only because they predict what comes next, so they're a handoff signal. Some quantities are both, and then they're usually two numbers: reward the stage's work, and score the state it left. Don't reuse the `reward` as the `score`. That pays the stage a second time for its own work, at the next stage's expense.

For both `score` and `value`:

- **Measure the handoff state at the boundary.** Compute the signal from what the next stage inherits, at the moment the stage completes. A signal that peeks at what happened afterwards leaks the outcome into earlier stages, which is `stage_lambda = 1` with extra steps.
- **Make it vary where it matters.** The signal acts only through differences between a stage's completers in a group. If it's the same for all of them it does nothing, and a constant offset changes nothing.
- **Prefer measurements that are hard to move without doing the work.** The stage is paid for raising the signal, so a policy can learn to inflate it without improving the state, taking credit from the next stage. Tests passing is harder to inflate than lines changed. A fitted `score` catches gaming: its `score_slope` falls or turns negative, and the shaping shrinks with it. A `value` keeps paying for it.

For a `score`:

- **Use one number, roughly linear in what's left.** The fit is a straight line from score to downstream return. Fractions beat counts when tasks differ in size. Take the log of a quantity that spans orders of magnitude, such as diff size. Clip outliers, since one extreme score dominates a least-squares fit. A 0/1 indicator is fine: its slope is the difference in mean downstream return between the two sides.
- **Scale and direction are free, meaning isn't.** Any units, and higher can be better or worse; the fit sorts that out. Under one `pool` key, though, the score must mean the same thing in every group. If you change how it's computed, change the key.

For a `value`:

- **Forecast the whole remaining return.** That is the task reward plus every later stage's `reward`, including the default 1 for each later stage the rollout will complete. With default stage rewards, a probability of eventual success alone misses those.
- **Get the differences right, not the level.** Only differences between completers matter, so a value off by a constant is fine. One whose spread is off by a factor is not: half the true spread leaves half the carry-forward unaccounted for, double over-corrects, and `std_normalize` repairs neither. A heuristic that only ranks handoffs needs a mapping to reward units before it's a value; until then, write it as a `score`.
- **Check the calibration by writing it as a `score`.** The logged `score_slope`, averaged over groups, should sit near 1. Below 1, your values are spread too wide; above 1, too narrow.
- **Recalibrate as the policy changes.** A value forecasts what the current policy will earn from a state, and as later stages improve, the same handoff becomes worth more. A `value` calibrated once goes stale; a `score` is refit every group. See [Avoiding stale fits](#avoiding-stale-fits).

#### Avoiding stale fits

A pooled slope and any `value`, whether fit from data or derived from the environment, forecast how the policy playing the later stages turns a handoff into return. That changes with every optimizer step, not only between checkpoints. If the patch stage gets better at recovering from a sloppy repro, the repro's score predicts less and the true slope falls. Only differences between completers matter, so drift in the level is harmless: if better downstream play adds the same amount to every handoff, the signal stays calibrated. Staleness bites when learning treats good and bad handoffs differently, as in the repro example. It is a matter of degree: a forecast is roughly right for policies near the one it describes, and drifts as the downstream stages learn. Ways to keep it in check:

1. **Start without a `pool` key.** A per-group fit only sees the current policy (give or take the async off-policy lag), so it can't go stale. If it's too noisy, first try the fixes that add no staleness: a more informative score, or larger groups. Pooling is the fix that trades noise for staleness, so reach for it last.
2. **If you pool, budget the memory in optimizer steps, not groups.** A key remembers about `1 / (1 − score_decay)` groups *under that key*, so its age in steps depends on how often the key appears:
    - **A key shared by a family of tasks** may appear several times per step. The default 0.9 (about 10 groups) then spans a step or two, which is fine.
    - **A per-task key** appears about once per epoch. The same 10 groups then reach back about 10 epochs. Either lower `score_decay` a lot, which leaves little to pool, or use a broader key if the score means the same thing across those tasks.
3. **Watch the gap between the two logged slopes.** `score_slope` is the current group alone, and `score_slope_pooled` includes history. Per-group slopes are noisy, but their trend over steps is current. If the pooled slope consistently lags that trend, the pool is stale: shorten the memory or drop the key.
4. **Don't count on shrinkage to catch it.** Shrinkage discounts a slope that is uncertain, and pooling makes the slope look certain because it has many points. A stale pooled fit is confidently wrong, and shrinkage passes it through. Staleness is the one failure the guards don't catch; watching the slope gap is the defence.
5. **Recheck any `value` on a schedule.** Every so often, write it as a `score` for a while: a logged slope that moves away from 1 over training is staleness showing up. How fast a `value` drifts depends on where it came from:
    - **Fit to data from an earlier policy** (offline calibration): right at the start of the run, then drifts as the later stages change. Refit it from recent rollouts every so often, not only at the start of a run.
    - **Derived from the environment** (privileged simulator state, a physics or success model): it has an assumption about later play built in, and is off by however much the real policy differs from it. A formula that assumes competent play ("is the task still solvable from here", distance to goal for a good planner) is too optimistic early and becomes more accurate as the policy improves; one that assumes, say, a fixed pass rate per stage can drift either way.
    - **Truly policy-independent**: the remaining return from the handoff is already locked in or set by the environment, whatever the policy does later. Only this never goes stale, and it is rare, since later stages the policy can't influence aren't worth training.

    Better still, keep the measurement as a `score` and let each group refit it. If you must use a fixed `value`, run with `stage_lambda = 1` and `std_normalize = false`, where a miscalibrated value costs variance but not bias.
6. **Prefer scores whose link to the outcome comes from the task, not the policy.** Fraction of tests passing predicts eventual success mostly because of what the task is. Diff size predicts it partly because of how the current patch stage behaves, so its slope drifts more as that stage learns. This is a heuristic, not a guarantee.

However stale, a handoff signal only moves credit between a rollout's own stages and never changes its total.

#### Stages from an environment's clock

A task with no natural milestones can still be staged. Most interactive environments advance on a clock of their own, which ticks at each opponent's move, simulated user's turn or simulator step. End a stage at a tick, every tick or every few. Every rollout then completes every stage until its episode ends, which is the always-complete case, so the `score` does all the conditioning.

Writing the record:

- **List the same stages in every trace.** Every trace in a group must report the same stage list, so list every stage the longest episode the environment allows would reach, except the last, which is the task stage. Mark each stage at the agent's last sampled node before the tick that ends it.
- **Leave the stage an episode ends in unmarked.** A game that ends during stage 30 has failed stage 30: its task reward lands on that stage's tokens and is compared with the group's value of playing on. Marking that stage would leave the task reward no tokens to land on.
- **Write `reward: 0` on every completed stage**, or a real per-stage reward. The default pays 1 per completed stage, which on a clock pays for lasting longer.
- **Score every completed stage** from the state the environment produced at its tick: an engine's evaluation of the position after the opponent's reply, a user simulator's estimate of how close the user is to done. Without a score the rule is per-stage GRPO, and the outcome never reaches the earlier stages.
- **Set `success_threshold = inf`** when an episode can end early in success. Otherwise a win before the last stage reads as a success with unmarked stages, and the group falls back to plain GRPO credit.

```python
# Env.run, one stage per opponent reply: marks[k] is the node index of the agent's last
# turn before reply k, evals[k] the engine's evaluation after it
interaction.trace.info["stages"] = [
    {"name": f"move{k + 1}", "node": marks[k], "reward": 0, "score": evals[k]}
    if k < len(marks)
    else {"name": f"move{k + 1}", "node": None}
    for k in range(MAX_MOVES - 1)
]
```

Choosing how long a stage is, and when to share a slope:

- **End stages between decisions.** A stage boundary inside a turn splits one decision's tokens between two advantages and asks for a score of a half-written response, so one agent turn is the shortest sensible stage.
- **Start with stages several ticks long and no `pool` key.** Each stage then fits its own slope, which assumes nothing about how the score's meaning changes over the episode and keeps no history. Its limit is noise: if most stages' `score_slope` sit within one `score_slope_se` of zero, the stages are shorter than the fits can resolve, so lengthen them or improve the score.
- **Read the per-stage slopes.** They show whether a given change in the score predicts about the same change in outcome at every stage.
    - **Roughly holds:** a chess engine's evaluation in pawns predicts winning about as well throughout the game. Not exactly, since a pawn late in the game is more decisive, with less time to recover.
    - **Fails:** a score whose meaning shifts over the episode. "Tests passing", when the early tests are trivial and the late ones hard, so one more passing test means much more late. "Distance to goal", when the early moves cross open ground and the last few steps are the hard part. Different measurements at different stages, such as formatting checks early and test results late. A behaviour that helps early and hurts late, which flips the slope's sign.
- **Share a key only across stages whose slopes agree:** one key for every stage if they agree throughout, or one per phase (`opening`, `middlegame`, `endgame`) if they agree only within phases. Sharing is what lets stages get shorter. The rule turns a score into a value by multiplying it by a fitted slope, and a stage's credit is roughly the value at its end minus the value at its start. With a slope per stage, those two values use two separately fitted slopes, each with its own error, so a stage where the position barely changed can still get sizeable credit from the difference between those errors, and shorter stages don't make that error smaller. With one slope, a stage's credit is that slope times how much the score changed: no change means no credit, and a slightly wrong slope only makes every stage's credit a bit too big or too small. A slope whose size drifts is tolerable; a slope shared across stages where the score means opposite things gives half of them credit pointing the wrong way, which is worse than not sharing.
- **Share without history at first.** All of a group's stages under one key are fit together, so a shared slope works from the first group. `score_decay = 0` keeps no history, so it can't go stale; raise it only if the shared slope is still noisy. `score_decay` applies to every key in the run; see [Avoiding stale fits](#avoiding-stale-fits).
- **Shorter stages are then limited by the score itself.** A stage's credit comes from how much the score changed across it. If the score's change over a stage is mostly measurement noise, shorter stages add little: improve the score before shortening them.
- **Rescale `stage_lambda` with stage length.** λ counts stages, not ticks: credit looks about `1 / (1 − λ)` stages ahead. When you halve stage length, use √λ to keep the same horizon.
- **Keep `std_normalize = false`.** Normalizing scales every stage's credit to the same size, including stages whose real variation is mostly noise.

#### Configuration

`stage_lambda` is λ over stages: 0 credits each stage against the group's value of the stage it leads to, which keeps downstream variance out of early stages; 1 credits each stage with the rollout's full remaining return minus `V_k`, which blames a stage for everything after it. Raise it when the quality of a stage's work carries into later stages in ways neither "entered stage k+1" nor a `value` captures.

With a `value` or `score` in place, λ also decides what that signal does: each handoff a stage is paid is scaled by (1 − λ), while the charge to the next stage keeps full weight. In practice:

- **Start at λ = 0.** The signal then carries the downstream credit itself, so its calibration matters. Raise λ only if the handoff diagnostic (a stage improving while the next one gets worse) still fires with the signal in place.
- **If you have a measurement but no calibration, write a `score`, not a rough `value`.** The group fits the scale for you, whatever `std_normalize` is set to.
- **High λ makes a rough `value` safe only with `std_normalize = false`, the default.** There, a `value` at λ = 1 acts purely as a baseline for the next stage, where a wrong scale costs variance but never bias. With `std_normalize = true`, each handoff is paid at one stage's scale and charged at the next's, so it keeps carrying credit at every λ and a miscalibrated `value` still distorts it.

```toml
[orchestrator.train.algo]
type = "temporal_grpo_continuous"
std_normalize = false   # true reproduces the paper's rule for binary stages
stage_lambda = 0.0      # 1 credits each stage with the rollout's remaining return
score_shrinkage = 1.0   # only for stages whose entries carry a score
score_decay = 0.9       # only for entries that carry a pool key
```

### Self-Play Advantage (RAE)

Group-relative baselines assume the group is exchangeable attempts by one agent. A multi-agent self-play env breaks that: one episode yields one trace per agent, all trainable, and in a zero-sum game the rewards sum to ~0 whatever the policy does — the group mean carries no information, and centering against it converts any structural asymmetry (a first-mover edge) into permanent credit for one agent.

`rae` implements SPIRAL's role-conditioned advantage estimation ([arXiv:2506.24119](https://arxiv.org/abs/2506.24119)): each agent keeps an exponential-moving-average baseline of its own rewards, and every trace's advantage is its reward minus its agent's baseline — measured against the *pre-update* baseline (the unbiased order), then folded in at `decay` (SPIRAL's α, default 0.95). The algorithm instance is per-env, so baselines are keyed per (env, agent) — the paper's per (game, role). Advantages are not normalized, and `group_size` is free (RAE needs no sibling rollouts; `group_size = 1` is fine). Baselines live in orchestrator memory and re-warm from 0 over ~`1/(1 − decay)` traces per agent after a restart.

```toml
[orchestrator.train.algo]
type = "rae"
decay = 0.95

[[orchestrator.train.source]]
name = "kuhn-poker"
env.taskset.id = "kuhn-poker"
env.player0.harness.id = "null"
env.player0.runtime.type = "subprocess"
env.player1.harness.id = "null"
env.player1.runtime.type = "subprocess"
```

Both of `kuhn-poker`'s agents late-bind to the run's own model — shared-policy self-play against a continuously improving opponent. Pin one agent to a frozen endpoint (`env.player1.model = ...`) for asymmetric play; its traces are marked untrainable by the env and never reach the advantage computation. A single-agent env under `rae` degrades to REINFORCE with an EMA baseline.

### Authoring an Algorithm

There is no config hook that points at user code — a new credit-assignment scheme is a new named algorithm in the repo. Subclass `Algorithm`, assign credit in the scoring hook whose timing fits your signal, and register the class:

```python
# src/prime_rl/orchestrator/algo/my_algo.py
import torch

from prime_rl.orchestrator.train.algo.base import Algorithm
from prime_rl.orchestrator.train.algo.routing import assign_advantages


class MyAlgorithm(Algorithm):
    async def score_group(self, episodes):
        traces = [trace for episode in episodes for trace in episode.traces]
        rewards = torch.tensor([trace.reward for trace in traces], dtype=torch.float32)
        advantages = ...  # one value per trace
        for trace, advantage in zip(traces, advantages.tolist(), strict=True):
            assign_advantages(trace, advantage)
```

Add a typed `MyAlgoConfig` to `prime_rl.configs.algorithm` and its discriminated union, then register `"my_algo": MyAlgorithm` in `ALGORITHM_CLASSES`. Pick `score_episode` for rollout-local work or model calls, and `score_group` for group-relative credit. `assign_advantages` takes a scalar or a list aligned to the trace graph's sampled tokens.

Curriculum admission and metrics can inspect the resulting graph-native streams after group scoring.

### Reference Scoring

`OPDAlgorithm` / `OPSDAlgorithm` do their model I/O in `score_episode`: as each episode arrives they query a reference and attach sampled-token reference logprobs to its graph nodes:

- `opd` — score each sample's own context under the `teacher` (a frozen [model reference](#model-references)) via prefill; fills `ref_logprobs` for the `ref_kl` loss component (on-policy distillation). The `teacher` is typed `FrozenModelConfig`, so `"policy"` isn't representable (the KL would be identically zero).
- `opsd` — SDFT: prepend an expert demonstration as a leading system message (`template`, with a `{demonstration}` placeholder) and score the sample under that demo-conditioned context. The sample is scored verbatim (`hint_block + token_ids`, slicing the hint's logprobs back off), so the join is BPE-clean and it's robust to tool/multimodal prompts and any number of turns. The scoring reference *is* the live policy — self-distillation names no teacher. opsd builds its own renderer to tokenize the hint block: the tokenizer is always the live policy's (not configurable — there is no separate model), and only the `renderer` family is settable (defaults to `"auto"`, resolved from the policy tokenizer; set it to match a non-auto policy renderer). The demonstration is read from the example's `info[demo_key]`, falling back to a top-level rollout field of the same name (e.g. `answer`).

```toml
[orchestrator.train.algo]
type = "opsd"
demo_key = "demonstration"
```

Scoring runs before curriculum admission, so a rollout that is later rejected still costs its reference compute.

The orchestrator filters samples with no training signal. This includes samples with zero advantage on all RL tokens. Samples that still carry CE or reference-KL components are retained. Filtering an RL token also removes its trainer/inference mismatch-KL contribution.

`orchestrator.constant_trainer_batch_size` defaults to `true`. The orchestrator filters samples before they count toward the batch target. It collects replacements, so rollout-based batches contain `orchestrator.batch_size` training traces. Set the option to `false` to filter after collection without replacement. This setting can improve orchestrator throughput, but it produces smaller trainer batches.

## Curricula

Each training source has a `Curriculum` composed from one `TaskSampler` and any number of named `AdmissionGate`s. The sampler chooses tasks and observes every finalized `list[vf.Episode]`; each gate inspects the same native episodes. The group trains only if every gate admits it. Rejected groups remain observable while the orchestrator samples again to fill the batch.

Samplers and gates can be stateful. Their `state_dict`, `load_state_dict`, and `metrics` methods are included in orchestrator checkpoints and logged under `curriculum/<env>/`.

Three small implementations are included:

- `StandardSampler` is the default: it advances the task iterator and cycles finite tasksets in source order.
- `DifficultyPoolSampler` samples finite tasksets with replacement and tracks each task's latest valid mean group reward. Each named pool has an inclusive reward threshold and a relative per-task sampling weight; weight `0` disables sampling from that pool. Unseen tasks use neutral weight `1.0`, so pool observations affect sampling immediately without waiting for a full taskset pass.
- `AdvRangeGate` rejects a group when every trainable-token advantage falls inside `reject_min` through `reject_max`. Unlike the built-in token filter, it can reject a configurable advantage range. Groups without an advantage stream are admitted.

```toml
[orchestrator.train.source.curriculum.sampler]
type = "difficulty_pool"

[orchestrator.train.source.curriculum.sampler.pools.hard]
threshold = 0.25
weight = 0.2

[orchestrator.train.source.curriculum.sampler.pools.normal]
threshold = 0.75
weight = 1.0

[orchestrator.train.source.curriculum.sampler.pools.easy]
threshold = 1.0
weight = 0.2

[orchestrator.train.source.curriculum.gates.low_signal]
type = "advantage_range"
reject_min = -0.05
reject_max = 0.05
```

## Multi-Turn Trajectories

For multi-turn rollouts (tool use, browser environments, long conversations), `prime-rl` records each LLM request/response as an independent **trajectory step** and merges them at training time using best-effort interleaving — with [renderers](#renderers) as the mechanism that keeps the merge safe by construction.

### Extension Property

A sequence of trajectory steps has the **extension property** when each successive step's prompt contains all previous prompts and completions as an exact prefix. The trainer relies on this property — when it holds:

- Multiple steps merge into one training sample.
- Compute scales as $O(T)$ in the trajectory length.

When it breaks (chat template strips past thinking, environment compacts context, an agent hands off to a sub-agent, etc.), the trainer starts a new training sample from that step:

- Graceful fallback to multiple samples — no corrupted data.
- Worst case (every step breaks extension) is $O(T^2)$.

### Best-Effort Interleaving

Concretely:

```
5-step trajectory where extension breaks at step 4:

steps 1–3: extension holds   → merged into Sample 1
step 4:    extension breaks  (e.g. thinking stripped from history)
steps 4–5: extension holds   → merged into Sample 2

result: 2 training samples instead of 5
```

The orchestrator enforces an **exact prefix invariant**: the prompt at turn $t$ must be the exact concatenation of prior messages exactly as the LLM originally generated them. If turn 2's prompt is `U1, A1', U2` while `A1' ≠ A1`, the orchestrator can't safely merge — either choice produces logprob drift between trainer and inference. Starting a fresh sample is the only correct behavior, so that's what happens.

### Renderers

Best-effort interleaving works because the renderer guarantees the exact-prefix invariant *by construction* — it never re-renders prior turns, so it can't lose tokens to chat-template normalization, BPE retokenization drift, or thinking stripping. A renderer turns a model's chat template into a Python object that can:

- `render_ids(messages)` — tokenize messages to ids the inference engine accepts.
- `parse_response(completion_ids)` — recover structured `(content, reasoning_content, tool_calls)` from sampled ids.
- `bridge_to_next_turn(prev_prompt_ids, prev_completion_ids, new_messages)` — extend the previous turn's tokens verbatim with the new environment turn, instead of re-rendering history.

When `bridge_to_next_turn` succeeds, the trainer sees the exact token stream the sampler produced; when it can't be proven safe (e.g. the renderer is `DefaultRenderer` and the template's stop sequence is unknown), it returns `None` and the orchestrator falls back to a full re-render — which triggers the new-sample fallback above.

A common source of breakage in the absence of a hand-coded renderer is models like Qwen3 whose chat templates strip past `<think>` blocks across user turns:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
messages = [
    {"role": "user", "content": "U1"},
    {"role": "assistant", "content": "<think>R1</think>A1"},
    {"role": "user", "content": "U2"},
]
tok.apply_chat_template(messages[:1], tokenize=False)
# <|im_start|>user
# U1<|im_end|>

tok.apply_chat_template(messages, tokenize=False)
# <|im_start|>user\nU1<|im_end|>\n<|im_start|>assistant\nA1<|im_end|>\n<|im_start|>user\nU2<|im_end|>
# (the <think>R1</think> from turn 2 is gone)
```

Hand-coded renderers ship for `qwen3`, `qwen3-vl`, `qwen3.5`, `glm-5`, `glm-4.5`, `minimax-m2`, `deepseek-v3`, `kimi-k2`, `kimi-k2.5`, `nemotron-3`, `gpt-oss`; anything else falls back to `DefaultRenderer` (a generic `apply_chat_template` wrapper). Pick one via:

```toml
[orchestrator.renderer]
name = "auto"   # detect from tokenizer; pass an explicit name for fine-tunes
```

A renderer outside the `renderers` package loads as a custom renderer: `name = "custom"` with `import_path = "my_module.Class"` (or `path/to/file.py:Class`), see [Custom renderers](training.md#dataset-format).

For the full design rationale (failure modes ruled out, empirical token-identity comparison against `apply_chat_template`, when to write a hand-coded renderer), see [the renderers writeup on the Prime Intellect blog](https://www.primeintellect.ai/blog/renderers) — the canonical reference.

### Discontinuous Trajectories

Some envs are discontinuous by design — e.g. a main agent delegating to a sub-agent and getting back only a summarized result, not the sub-agent's whole conversation. Best-effort interleaving handles this naturally: each agent's contiguous turns merge, the handoff starts a new sample. The trainer never sees fabricated extension where there is none.
