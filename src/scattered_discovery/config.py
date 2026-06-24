from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from scattered_discovery.rewards import SCATTERED_CAUSAL_REWARD


@dataclass(frozen=True)
class WorldConfig:
    num_branches: int = 4
    branch_depth: int = 3
    distractors_per_node: int = 2
    true_mean: float = 1.0
    false_mean: float = 0.0
    noise_sigma: float = 0.35
    accept_threshold: float = 0.82
    reject_threshold: float = 0.18
    base_budget: int = 10
    test_cost: int = 1
    intervene_cost: int = 2
    invalid_action_cost: int = 1
    valid_hypothesis_reward: float = SCATTERED_CAUSAL_REWARD.valid_hypothesis_reward
    false_penalty: float = SCATTERED_CAUSAL_REWARD.false_penalty
    non_final_penalty: float = SCATTERED_CAUSAL_REWARD.non_final_penalty
    unsupported_penalty: float = SCATTERED_CAUSAL_REWARD.unsupported_penalty
    budget_penalty: float = SCATTERED_CAUSAL_REWARD.budget_penalty
    clean_invalid_final_reward: float = (
        SCATTERED_CAUSAL_REWARD.clean_invalid_final_reward
    )


@dataclass(frozen=True)
class SweepConfig:
    dispersions: list[float] = field(
        default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0]
    )


@dataclass(frozen=True)
class EvalConfig:
    protocol: Literal["single", "set"] = "single"
    num_worlds: int = 2
    rollouts_per_world: int = 3
    max_steps: int = 10
    final_commit_attempt: bool = True
    output_dir: str = "results"
    run_name: str = "debug"
    seed: int = 1
    set_budget_multiplier: int | None = None


@dataclass(frozen=True)
class ModelConfig:
    provider: Literal["ollama"] = "ollama"
    model: str = "qwen3:1.7b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    top_p: float = 0.9
    num_predict: int = 320
    request_timeout_s: float = 180.0
    think: bool | str | None = True
    finalize_empty_content: bool = True
    finalizer_num_predict: int = 160
    finalizer_max_thinking_chars: int = 12_000


@dataclass(frozen=True)
class AgentConfig:
    include_hidden_debug_in_prompt: bool = False
    include_evidence_status_in_prompt: bool = False
    max_evidence_items: int = 12


@dataclass(frozen=True)
class ExperimentConfig:
    world: WorldConfig = field(default_factory=WorldConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)


def _dataclass_from_dict(cls: type[Any], data: dict[str, Any] | None) -> Any:
    if data is None:
        return cls()
    field_names = cls.__dataclass_fields__.keys()  # type: ignore[attr-defined]
    filtered = {key: value for key, value in data.items() if key in field_names}
    return cls(**filtered)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    return ExperimentConfig(
        world=_dataclass_from_dict(WorldConfig, raw.get("world")),
        sweep=_dataclass_from_dict(SweepConfig, raw.get("sweep")),
        eval=_dataclass_from_dict(EvalConfig, raw.get("eval")),
        model=_dataclass_from_dict(ModelConfig, raw.get("model")),
        agent=_dataclass_from_dict(AgentConfig, raw.get("agent")),
    )
