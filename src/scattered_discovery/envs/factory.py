from __future__ import annotations

from dataclasses import asdict
import os
from typing import Any

from scattered_discovery.config import AgentConfig, WorldConfig
from scattered_discovery.envs.base import DiscoveryEnv, EnvSpec
from scattered_discovery.envs.causal_micro_lab import CausalMicroLabEnv
from scattered_discovery.envs.hypospace_3d import HypoSpace3DEnv
from scattered_discovery.envs.hypospace_boolean import HypoSpaceBooleanEnv
from scattered_discovery.envs.hypospace_causal import HypoSpaceCausalEnv
from scattered_discovery.envs.scattered_causal import ScatteredCausalDiscoveryEnv
from scattered_discovery.rewards import reward_config_from_task


def make_env(spec: EnvSpec | dict[str, Any]) -> DiscoveryEnv:
    if isinstance(spec, dict):
        spec = EnvSpec(**spec)
    task = dict(spec.task)
    dense_cml_reward = os.environ.get("CAUSAL_MICRO_LAB_DENSE_REWARD", "0") == "1"
    nonempty_output_reward = float(task.pop("nonempty_output_reward", 0.2 if dense_cml_reward else 0.0))
    rule_marker_reward = float(
        task.pop("rule_marker_reward", 0.05 if dense_cml_reward else 0.0)
    )
    parse_valid_reward = float(
        task.pop("parse_valid_reward", 0.1 if dense_cml_reward else 0.0)
    )
    syntax_valid_reward = float(
        task.pop("syntax_valid_reward", 0.2)
    )
    evidence_consistent_reward = float(
        task.pop("evidence_consistent_reward", 0.4 if dense_cml_reward else 0.0)
    )
    valid_hypothesis_reward = float(task.pop("valid_hypothesis_reward", 1.0))
    reward_config = reward_config_from_task(spec.env_type, task)
    task.pop("reward", None)
    if spec.env_type == "causal_micro_lab":
        return CausalMicroLabEnv(
            state=task.get("state"),
            seed=int(task.get("seed", spec.seed)),
            target_mode_count=int(task.get("target_mode_count", 4)),
            nonempty_output_reward=nonempty_output_reward,
            rule_marker_reward=rule_marker_reward,
            parse_valid_reward=parse_valid_reward,
            syntax_valid_reward=syntax_valid_reward,
            evidence_consistent_reward=evidence_consistent_reward,
            valid_hypothesis_reward=valid_hypothesis_reward,
        )
    if spec.env_type == "scattered_causal":
        world_raw = task.pop("world", {})
        agent_raw = task.pop("agent", {})
        world_values = {**reward_config.world_kwargs(), **world_raw}
        return ScatteredCausalDiscoveryEnv(
            WorldConfig(**world_values),
            world_seed=int(task.get("world_seed", spec.seed)),
            episode_seed=int(task.get("episode_seed", spec.seed * 1009 + 19)),
            dispersion=float(task.get("dispersion", 0.0)),
            protocol=spec.protocol,
            max_commit=spec.max_commit,
            budget=task.get("budget"),
            agent_config=AgentConfig(**agent_raw),
            reward_config=reward_config,
        )
    if spec.env_type == "hypospace_causal":
        return HypoSpaceCausalEnv(
            nodes=tuple(task.get("nodes", ("A", "B", "C"))),
            max_edges=task.get("max_edges", 2),
            target_edges=tuple(tuple(edge) for edge in task["target_edges"])
            if "target_edges" in task
            else None,
            seed=int(task.get("seed", spec.seed)),
            query_budget=task.get("query_budget"),
            protocol=spec.protocol,
            max_commit=spec.max_commit,
            show_version_space_size=bool(task.get("show_version_space_size", False)),
            reward_config=reward_config,
        )
    if spec.env_type == "hypospace_boolean":
        return HypoSpaceBooleanEnv(
            variables=tuple(task.get("variables", ("x", "y"))),
            operators=tuple(task.get("operators", ("AND", "OR", "NOT", "XOR", "NOR"))),
            max_depth=int(task.get("max_depth", 2)),
            target_expression=task.get("target_expression"),
            seed=int(task.get("seed", spec.seed)),
            query_budget=task.get("query_budget"),
            protocol=spec.protocol,
            max_commit=spec.max_commit,
            show_version_space_size=bool(task.get("show_version_space_size", False)),
            reward_config=reward_config,
        )
    if spec.env_type == "hypospace_3d":
        return HypoSpace3DEnv(
            grid_size=int(task.get("grid_size", 2)),
            max_height=int(task.get("max_height", 3)),
            max_blocks=task.get("max_blocks", 3),
            target_heights=tuple(tuple(row) for row in task["target_heights"])
            if "target_heights" in task
            else None,
            seed=int(task.get("seed", spec.seed)),
            query_budget=task.get("query_budget"),
            protocol=spec.protocol,
            max_commit=spec.max_commit,
            show_version_space_size=bool(task.get("show_version_space_size", False)),
            reward_config=reward_config,
        )
    raise ValueError(f"Unsupported env_type: {spec.env_type}")


def spec_to_dict(spec: EnvSpec) -> dict[str, Any]:
    return asdict(spec)
