from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
from random import Random
from typing import Any, Iterable

import yaml

from scattered_discovery.config import AgentConfig, WorldConfig
from scattered_discovery.envs.base import EnvSpec
from scattered_discovery.envs.factory import spec_to_dict
from scattered_discovery.rewards import REWARD_PROFILES

ENV_TYPES = [
    "scattered_causal",
    "hypospace_causal",
    "hypospace_boolean",
    "hypospace_3d",
]

WORLD_CONFIG_FIELDS = {field.name for field in fields(WorldConfig)}
AGENT_CONFIG_FIELDS = {field.name for field in fields(AgentConfig)}


def build_specs(
    *,
    env_type: str,
    count: int,
    seed: int,
    protocol: str,
    max_steps: int,
    max_commit: int,
    task_overrides: dict[str, Any] | None = None,
    dispersion_values: list[float] | None = None,
    world_values: dict[str, list[Any]] | None = None,
) -> list[EnvSpec]:
    task_overrides = task_overrides or {}
    if world_values:
        fixed_world = task_overrides.get("world") or {}
        overlap = sorted(set(world_values) & set(fixed_world))
        if overlap:
            raise ValueError(
                "Fields cannot appear in both world_values and task.world: "
                f"{', '.join(overlap)}. Put sampled fields only in world_values "
                "and fixed fields only in task.world."
            )
    specs: list[EnvSpec] = []
    for index in range(count):
        task = _deep_merge({}, task_overrides)
        task.setdefault("seed", seed + index)
        if env_type == "scattered_causal":
            task.setdefault("world_seed", seed + index)
            task.setdefault("episode_seed", (seed + index) * 1009 + 19)
            if dispersion_values is not None:
                task["dispersion"] = dispersion_values[index % len(dispersion_values)]
            else:
                task.setdefault("dispersion", 0.0)
            if world_values:
                sampled_world = _sample_world_values(
                    world_values=world_values,
                    seed=seed,
                    index=index,
                )
                task["world"] = _deep_merge(task.get("world") or {}, sampled_world)
        specs.append(
            EnvSpec(
                env_type=env_type,
                task=task,
                protocol=protocol,
                max_steps=max_steps,
                max_commit=max_commit,
                seed=seed + index,
            )
        )
    return specs


def _sample_world_values(
    *,
    world_values: dict[str, list[Any]],
    seed: int,
    index: int,
) -> dict[str, Any]:
    rng = Random(seed * 1_000_003 + index * 9_176 + 17)
    return {
        key: values[rng.randrange(len(values))]
        for key, values in sorted(world_values.items())
    }


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _filtered_dataclass_values(
    data: dict[str, Any],
    allowed_fields: set[str],
    section_name: str,
) -> dict[str, Any]:
    unknown = sorted(set(data) - allowed_fields)
    if unknown:
        raise ValueError(
            f"Unknown {section_name} field(s): {', '.join(unknown)}. "
            f"Allowed fields: {', '.join(sorted(allowed_fields))}."
        )
    return dict(data)


def _task_from_definition(definition: dict[str, Any]) -> dict[str, Any]:
    task = dict(definition.get("task") or {})

    # Convenience aliases so YAML can keep scattered-causal world/agent controls
    # at the dataset level or under `task`.
    if "world" in definition:
        task["world"] = _deep_merge(task.get("world") or {}, definition["world"] or {})
    if "agent" in definition:
        task["agent"] = _deep_merge(task.get("agent") or {}, definition["agent"] or {})
    if "reward_profile" in definition:
        task["reward_profile"] = definition["reward_profile"]
    if "reward" in definition:
        task["reward"] = _deep_merge(
            task.get("reward") or {}, definition["reward"] or {}
        )

    if "world" in task:
        task["world"] = _filtered_dataclass_values(
            task["world"] or {},
            WORLD_CONFIG_FIELDS,
            "world",
        )
    if "agent" in task:
        task["agent"] = _filtered_dataclass_values(
            task["agent"] or {},
            AGENT_CONFIG_FIELDS,
            "agent",
        )
    return task


def _dataset_definitions_from_yaml(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Dataset YAML must contain a mapping.")

    defaults = raw.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ValueError("Dataset YAML `defaults` must be a mapping.")

    datasets = raw.get("datasets")
    if datasets is None:
        datasets = [raw]
        defaults = {}
    if not isinstance(datasets, list):
        raise ValueError("Dataset YAML `datasets` must be a list.")

    definitions = []
    for dataset in datasets:
        if not isinstance(dataset, dict):
            raise ValueError("Each dataset YAML entry must be a mapping.")
        definitions.append(_deep_merge(defaults, dataset))
    return definitions


def _dispersion_values_from_definition(
    definition: dict[str, Any],
) -> list[float] | None:
    values = definition.get("dispersion_values")
    if values is None:
        return None
    if str(definition["env_type"]) != "scattered_causal":
        raise ValueError("dispersion_values is only supported for scattered_causal.")
    if not isinstance(values, list) or not values:
        raise ValueError("dispersion_values must be a non-empty list.")
    task = definition.get("task") or {}
    if isinstance(task, dict) and "dispersion" in task:
        raise ValueError("Use either task.dispersion or dispersion_values, not both.")
    return [float(value) for value in values]


def _world_values_from_definition(
    definition: dict[str, Any],
) -> dict[str, list[Any]] | None:
    values = definition.get("world_values")
    if values is None:
        return None
    if str(definition["env_type"]) != "scattered_causal":
        raise ValueError("world_values is only supported for scattered_causal.")
    if not isinstance(values, dict) or not values:
        raise ValueError("world_values must be a non-empty mapping.")
    unknown = sorted(set(values) - WORLD_CONFIG_FIELDS)
    if unknown:
        raise ValueError(
            f"Unknown world_values field(s): {', '.join(unknown)}. "
            f"Allowed fields: {', '.join(sorted(WORLD_CONFIG_FIELDS))}."
        )
    result: dict[str, list[Any]] = {}
    for key, options in values.items():
        if not isinstance(options, list) or not options:
            raise ValueError(f"world_values.{key} must be a non-empty list.")
        result[key] = list(options)
    return result


def build_rows_from_definition(
    definition: dict[str, Any],
) -> tuple[Path, list[dict[str, Any]]]:
    env_type = str(definition["env_type"])
    if env_type not in ENV_TYPES:
        raise ValueError(f"Unsupported env_type: {env_type}")
    dispersion_values = _dispersion_values_from_definition(definition)
    world_values = _world_values_from_definition(definition)
    task = _task_from_definition(definition)
    if world_values:
        fixed_world = task.get("world") or {}
        overlap = sorted(set(world_values) & set(fixed_world))
        if overlap:
            raise ValueError(
                "Fields cannot appear in both world_values and task.world: "
                f"{', '.join(overlap)}. Put sampled fields only in world_values "
                "and fixed fields only in task.world."
            )
    specs = build_specs(
        env_type=env_type,
        count=int(definition.get("count", 128)),
        seed=int(definition.get("seed", 1)),
        protocol=str(definition.get("protocol", "single")),
        max_steps=int(definition.get("max_steps", 8)),
        max_commit=int(definition.get("max_commit", 1)),
        task_overrides=task,
        dispersion_values=dispersion_values,
        world_values=world_values,
    )
    return Path(definition["output"]), specs_to_rows(
        specs,
        agent_name=str(definition.get("agent_name", "discovery_agent_loop")),
        data_source=str(definition.get("data_source", env_type)),
    )


def write_rows(rows: list[dict[str, Any]], output: str | Path) -> Path:
    path = Path(output)
    if path.suffix == ".parquet":
        write_parquet(rows, path)
    else:
        write_jsonl(rows, path)
    return path


def generate_from_config(path: str | Path) -> list[Path]:
    outputs = []
    for definition in _dataset_definitions_from_yaml(path):
        output, rows = build_rows_from_definition(definition)
        outputs.append(write_rows(rows, output))
        print(f"wrote {len(rows)} rows to {output}")
    return outputs


def specs_to_rows(
    specs: Iterable[EnvSpec], *, agent_name: str, data_source: str | None = None
) -> list[dict[str, Any]]:
    rows = []
    for index, spec in enumerate(specs):
        env_spec = spec_to_dict(spec)
        rows.append(
            {
                "index": index,
                "data_source": data_source or spec.env_type,
                "agent_name": agent_name,
                "prompt": "Interactive discovery task. The custom AgentLoop will construct the full prompt.",
                "raw_prompt": "Interactive discovery task.",
                "env_spec_json": json.dumps(env_spec, sort_keys=True),
                "reward_model": {"style": "rule"},
            }
        )
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Parquet output requires pandas and pyarrow. Install with "
            "`uv sync --extra verl` or write JSONL instead."
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _task_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    task: dict[str, Any] = {}
    if args.nodes:
        task["nodes"] = tuple(
            item.strip() for item in args.nodes.split(",") if item.strip()
        )
    if args.variables:
        task["variables"] = tuple(
            item.strip() for item in args.variables.split(",") if item.strip()
        )
    if args.operators:
        task["operators"] = tuple(
            item.strip().upper() for item in args.operators.split(",") if item.strip()
        )
    if args.max_edges is not None:
        task["max_edges"] = args.max_edges
    if args.max_depth is not None:
        task["max_depth"] = args.max_depth
    if args.grid_size is not None:
        task["grid_size"] = args.grid_size
    if args.max_height is not None:
        task["max_height"] = args.max_height
    if args.max_blocks is not None:
        task["max_blocks"] = args.max_blocks
    if args.query_budget is not None:
        task["query_budget"] = args.query_budget
    if args.dispersion is not None:
        task["dispersion"] = args.dispersion
    if args.budget is not None:
        task["budget"] = args.budget
    if args.world_seed is not None:
        task["world_seed"] = args.world_seed
    if args.episode_seed is not None:
        task["episode_seed"] = args.episode_seed
    if args.reward_profile is not None:
        task["reward_profile"] = args.reward_profile

    world = {
        name: getattr(args, name)
        for name in WORLD_CONFIG_FIELDS
        if getattr(args, name) is not None
    }
    if world:
        task["world"] = world

    agent = {
        name: getattr(args, name)
        for name in AGENT_CONFIG_FIELDS
        if getattr(args, name) is not None
    }
    if agent:
        task["agent"] = agent
    return task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="YAML dataset config.")
    parser.add_argument(
        "--env-type",
        choices=ENV_TYPES,
    )
    parser.add_argument("--output")
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--protocol", choices=["single", "set"], default="single")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-commit", type=int, default=1)
    parser.add_argument("--agent-name", default="discovery_agent_loop")
    parser.add_argument(
        "--nodes", default=None, help="Comma-separated causal nodes, e.g. A,B,C."
    )
    parser.add_argument(
        "--variables",
        default=None,
        help="Comma-separated Boolean variables, e.g. x,y,z.",
    )
    parser.add_argument(
        "--operators", default=None, help="Comma-separated Boolean operators."
    )
    parser.add_argument("--max-edges", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--max-height", type=int, default=None)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--query-budget", type=int, default=None)
    parser.add_argument("--dispersion", type=float, default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--world-seed", type=int, default=None)
    parser.add_argument("--episode-seed", type=int, default=None)
    parser.add_argument(
        "--reward-profile",
        choices=sorted(REWARD_PROFILES),
        default=None,
        help="Named reward profile to store in task.reward_profile.",
    )
    parser.add_argument("--num-branches", type=int, default=None)
    parser.add_argument("--branch-depth", type=int, default=None)
    parser.add_argument("--distractors-per-node", type=int, default=None)
    parser.add_argument("--true-mean", type=float, default=None)
    parser.add_argument("--false-mean", type=float, default=None)
    parser.add_argument("--noise-sigma", type=float, default=None)
    parser.add_argument("--accept-threshold", type=float, default=None)
    parser.add_argument("--reject-threshold", type=float, default=None)
    parser.add_argument("--base-budget", type=int, default=None)
    parser.add_argument("--test-cost", type=int, default=None)
    parser.add_argument("--intervene-cost", type=int, default=None)
    parser.add_argument("--invalid-action-cost", type=int, default=None)
    parser.add_argument("--valid-hypothesis-reward", type=float, default=None)
    parser.add_argument("--false-penalty", type=float, default=None)
    parser.add_argument("--non-final-penalty", type=float, default=None)
    parser.add_argument("--unsupported-penalty", type=float, default=None)
    parser.add_argument("--budget-penalty", type=float, default=None)
    parser.add_argument(
        "--include-hidden-debug-in-prompt",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--max-evidence-items", type=int, default=None)
    args = parser.parse_args()

    if args.config:
        generate_from_config(args.config)
        return
    if not args.env_type or not args.output:
        parser.error("--env-type and --output are required unless --config is used.")

    specs = build_specs(
        env_type=args.env_type,
        count=args.count,
        seed=args.seed,
        protocol=args.protocol,
        max_steps=args.max_steps,
        max_commit=args.max_commit,
        task_overrides=_task_overrides_from_args(args),
    )
    rows = specs_to_rows(specs, agent_name=args.agent_name)
    output = write_rows(rows, args.output)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
