from __future__ import annotations

import json
from pathlib import Path
from random import Random
from typing import Any, Callable

from scattered_discovery.envs.base import EnvSpec
from scattered_discovery.envs.causal_micro_lab.prompt_builder import build_prompt
from scattered_discovery.envs.causal_micro_lab.rewards import group_metrics
from scattered_discovery.envs.causal_micro_lab.signatures import ModeTable, build_mode_table
from scattered_discovery.envs.causal_micro_lab.state_generator import (
    EvidenceState,
    find_states,
)
from scattered_discovery.envs.causal_micro_lab.verifier import verify_output


SPLIT_NAMES = ("train", "val", "test")
ProgressFn = Callable[[str], None]


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return output


def write_table(rows: list[dict[str, Any]], path: str | Path) -> Path:
    output = Path(path)
    if output.suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                "Parquet output requires pandas and pyarrow. Use .jsonl for stdlib output."
            ) from exc
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(output, index=False)
        return output
    return write_jsonl(rows, output)


def mode_rows(*, mode_table: ModeTable | None = None) -> list[dict[str, Any]]:
    table = mode_table or build_mode_table()
    return [
        {
            "mode_id": mode.mode_id,
            "syntactic_count": mode.syntactic_count,
            "family_source": mode.family[0],
            "family_operator": mode.family[1],
            "canonical_json": json.dumps(mode.canonical.to_json(), sort_keys=True),
            "prediction_signature": "".join(
                str(bit) for outcome in mode.signature for bit in outcome
            ),
        }
        for mode in table.modes
    ]


def experiment_rows(*, mode_table: ModeTable | None = None) -> list[dict[str, Any]]:
    table = mode_table or build_mode_table()
    return [experiment.to_json() for experiment in table.experiments]


def generate_states(
    *,
    target_counts: tuple[int, ...] = (4, 8, 16),
    states_per_count: int = 8,
    seed: int = 1,
    max_evidence: int = 8,
    beam_width: int = 256,
    separation_bucket: str | None = None,
    mode_table: ModeTable | None = None,
) -> list[EvidenceState]:
    table = mode_table or build_mode_table()
    rng = Random(seed)
    states: list[EvidenceState] = []
    seen: set[str] = set()
    priority = list(range(min(64, len(table.modes))))
    rest = [index for index in range(len(table.modes)) if index not in set(priority)]
    rng.shuffle(rest)
    mode_indices = priority + rest
    for target_count in target_counts:
        for mode_index in mode_indices:
            candidates = find_states(
                mode_index,
                target_count,
                max_evidence=max_evidence,
                beam_width=beam_width,
                mode_table=table,
            )
            for state in candidates:
                if (
                    separation_bucket
                    and separation_bucket != "any"
                    and state.separation_bucket != separation_bucket
                ):
                    continue
                if state.state_id in seen:
                    continue
                seen.add(state.state_id)
                states.append(state)
                if sum(s.valid_mode_count == target_count for s in states) >= states_per_count:
                    break
            if sum(s.valid_mode_count == target_count for s in states) >= states_per_count:
                break
    return sorted(states, key=lambda item: (item.valid_mode_count, item.state_id))


def split_mode_ids(
    *,
    seed: int = 1,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    mode_table: ModeTable | None = None,
) -> dict[str, set[str]]:
    if len(ratios) != 3:
        raise ValueError("ratios must contain train, val, and test proportions")
    if any(value < 0 for value in ratios):
        raise ValueError("ratios must be non-negative")
    total = sum(ratios)
    if total <= 0:
        raise ValueError("at least one ratio must be positive")

    table = mode_table or build_mode_table()
    rng = Random(seed)
    mode_ids = [mode.mode_id for mode in table.modes]
    rng.shuffle(mode_ids)
    train_count = int(len(mode_ids) * ratios[0] / total)
    val_count = int(len(mode_ids) * ratios[1] / total)
    train = set(mode_ids[:train_count])
    val = set(mode_ids[train_count : train_count + val_count])
    test = set(mode_ids[train_count + val_count :])
    return {"train": train, "val": val, "test": test}


def generate_split_states(
    *,
    split_mode_ids: set[str],
    target_counts: tuple[int, ...] = (4, 8, 16),
    states_per_count: int = 8,
    seed: int = 1,
    max_evidence: int = 8,
    beam_width: int = 256,
    separation_buckets: tuple[str, ...] = (),
    split_name: str = "split",
    progress: ProgressFn | None = None,
    progress_every: int = 25,
    exclude_state_ids: set[str] | None = None,
    mode_table: ModeTable | None = None,
) -> list[EvidenceState]:
    table = mode_table or build_mode_table()
    rng = Random(seed)
    hidden_mode_ids = sorted(split_mode_ids)
    rng.shuffle(hidden_mode_ids)
    requested_buckets = tuple(
        bucket for bucket in separation_buckets if bucket and bucket != "any"
    )
    seen: set[str] = set(exclude_state_ids or ())
    states: list[EvidenceState] = []

    for target_count in target_counts:
        collected_for_target = 0
        bucket_quotas: dict[str, int] = {}
        if states_per_count <= 0:
            if progress:
                progress(
                    f"[{split_name}] skip M={target_count}: target states=0"
                )
            continue
        if requested_buckets:
            base, remainder = divmod(states_per_count, len(requested_buckets))
            bucket_quotas = {
                bucket: base + int(index < remainder)
                for index, bucket in enumerate(requested_buckets)
            }
        if progress:
            progress(
                f"[{split_name}] start M={target_count}: target states={states_per_count}"
            )
        for hidden_index, hidden_mode_id in enumerate(hidden_mode_ids, start=1):
            if progress and (hidden_index == 1 or hidden_index % progress_every == 0):
                progress(
                    f"[{split_name}] M={target_count}: "
                    f"{collected_for_target}/{states_per_count} states, "
                    f"searched {hidden_index}/{len(hidden_mode_ids)} hidden modes"
                )
            candidates = find_states(
                hidden_mode_id,
                target_count,
                max_evidence=max_evidence,
                beam_width=beam_width,
                mode_table=table,
            )
            for state in candidates:
                if state.state_id in seen:
                    continue
                if requested_buckets:
                    if bucket_quotas.get(state.separation_bucket, 0) <= 0:
                        continue
                    bucket_quotas[state.separation_bucket] -= 1
                seen.add(state.state_id)
                states.append(state)
                collected_for_target += 1
                if progress and (
                    collected_for_target == 1
                    or collected_for_target % progress_every == 0
                    or collected_for_target == states_per_count
                ):
                    progress(
                        f"[{split_name}] M={target_count}: collected "
                        f"{collected_for_target}/{states_per_count} "
                        f"state={state.state_id} sep={state.separation_bucket} "
                        f"mean_sep={state.mean_separation:.4f}"
                    )
                if collected_for_target >= states_per_count:
                    break
            if collected_for_target >= states_per_count:
                break
            if requested_buckets and all(value <= 0 for value in bucket_quotas.values()):
                break
        if progress:
            progress(
                f"[{split_name}] done M={target_count}: "
                f"{collected_for_target}/{states_per_count} states"
            )
    return sorted(states, key=lambda item: (item.valid_mode_count, item.state_id))


def state_rows(states: list[EvidenceState], *, mode_table: ModeTable | None = None) -> list[dict[str, Any]]:
    table = mode_table or build_mode_table()
    return [state.to_record(mode_table=table, include_private=True) for state in states]


def verl_rows_for_states(
    states: list[EvidenceState],
    *,
    agent_name: str = "causal_micro_lab_agent_loop",
    data_source: str = "causal_micro_lab",
    task_overrides: dict[str, Any] | None = None,
    agent_overrides: dict[str, Any] | None = None,
    mode_table: ModeTable | None = None,
) -> list[dict[str, Any]]:
    table = mode_table or build_mode_table()
    rows = []
    for index, state in enumerate(states):
        state_record = state.to_record(mode_table=table, include_private=True)
        task = {"state": state_record, **(task_overrides or {})}
        prompt = build_prompt(
            state,
            output_mode=task.get("output_mode", "single"),
            answer_count=int(task.get("answer_count", 1)),
        )
        spec = EnvSpec(
            env_type="causal_micro_lab",
            task=task,
            protocol="single",
            max_steps=1,
            max_commit=1,
            seed=index,
        )
        rows.append(
            {
                "index": index,
                "data_source": data_source,
                "agent_name": agent_name,
                "prompt": prompt,
                "raw_prompt": prompt,
                "env_spec_json": json.dumps(
                    {
                        "env_type": spec.env_type,
                        "task": spec.task,
                        "protocol": spec.protocol,
                        "max_steps": spec.max_steps,
                        "max_commit": spec.max_commit,
                        "max_consecutive_invalid": spec.max_consecutive_invalid,
                        "agent": agent_overrides or {},
                        "seed": spec.seed,
                    },
                    sort_keys=True,
                ),
                "state_json": json.dumps(state_record, sort_keys=True),
                "extra_info": {
                    "index": index,
                    "min_global_steps": 0,
                    "max_global_steps": 0,
                },
                "reward_model": {"style": "rule"},
            }
        )
    return rows


def sft_rows_for_states(
    states: list[EvidenceState],
    *,
    targets_per_state: int = 2,
    seed: int = 1,
    allowed_target_mode_ids: set[str] | None = None,
    mode_table: ModeTable | None = None,
) -> list[dict[str, Any]]:
    table = mode_table or build_mode_table()
    rng = Random(seed)
    rows: list[dict[str, Any]] = []
    for state in states:
        valid = [
            mode_id
            for mode_id in state.valid_mode_ids
            if allowed_target_mode_ids is None or mode_id in allowed_target_mode_ids
        ]
        rng.shuffle(valid)
        for target_mode_id in valid[: min(targets_per_state, len(valid))]:
            target = table.modes_by_id[target_mode_id].canonical.render_flat_rules()
            rows.append(
                {
                    "state_id": state.state_id,
                    "prompt": build_prompt(state),
                    "response": target,
                    "target_mode_id": target_mode_id,
                    "metadata": {
                        "valid_mode_count": state.valid_mode_count,
                        "separation_bucket": state.separation_bucket,
                        "family_bucket": state.family_bucket,
                    },
                }
            )
    return rows


def oracle_group_eval(
    state: EvidenceState,
    *,
    samples: int,
    mode_table: ModeTable | None = None,
) -> dict[str, float]:
    table = mode_table or build_mode_table()
    outputs = [
        table.modes_by_id[mode_id].canonical.render_json()
        for mode_id in state.valid_mode_ids[:samples]
    ]
    results = [verify_output(output, state, mode_table=table) for output in outputs]
    return group_metrics(results, state)


def _balanced_state_cap(
    states: list[EvidenceState],
    *,
    max_rows: int,
    seed: int,
) -> list[EvidenceState]:
    if max_rows <= 0:
        return []
    grouped: dict[int, list[EvidenceState]] = {}
    for state in states:
        grouped.setdefault(state.valid_mode_count, []).append(state)
    rng = Random(seed)
    for group in grouped.values():
        rng.shuffle(group)

    selected: list[EvidenceState] = []
    buckets = sorted(grouped)
    while len(selected) < max_rows and buckets:
        next_buckets = []
        for bucket in buckets:
            group = grouped[bucket]
            if group:
                selected.append(group.pop())
                if len(selected) == max_rows:
                    break
            if group:
                next_buckets.append(bucket)
        buckets = next_buckets
    return selected


def build_split_dataset(
    *,
    output_dir: str | Path,
    target_counts: tuple[int, ...] = (4, 8, 16),
    states_per_count: dict[str, int] | None = None,
    max_rows_per_split: dict[str, int] | None = None,
    seed: int = 1,
    max_evidence: int = 8,
    beam_width: int = 256,
    targets_per_state: int = 2,
    suffix: str = "jsonl",
    separation_buckets: tuple[str, ...] = (),
    source_splits: dict[str, str] | None = None,
    verl_task_overrides: dict[str, Any] | None = None,
    verl_agent_overrides: dict[str, Any] | None = None,
    verl_agent_name: str = "causal_micro_lab_agent_loop",
    include_verl: bool = True,
    progress: ProgressFn | None = None,
    progress_every: int = 25,
    mode_table: ModeTable | None = None,
) -> dict[str, Path]:
    table = mode_table or build_mode_table()
    split_ids = split_mode_ids(seed=seed, mode_table=table)
    source_for_split = source_splits or {split: split for split in SPLIT_NAMES}
    counts = states_per_count or {"train": 128, "val": 32, "test": 32}
    row_caps = max_rows_per_split or {}
    root = Path(output_dir)
    outputs: dict[str, Path] = {}
    seen_state_ids: set[str] = set()
    if progress:
        progress(
            f"[tables] writing modes={len(table.modes)} "
            f"experiments={len(table.experiments)} to {root}"
        )
    outputs["modes"] = write_table(mode_rows(mode_table=table), root / f"modes.{suffix}")
    outputs["experiments"] = write_table(
        experiment_rows(mode_table=table),
        root / f"experiments.{suffix}",
    )
    for split in SPLIT_NAMES:
        source_split = source_for_split.get(split, split)
        if source_split not in split_ids:
            raise ValueError(
                f"unknown source split {source_split!r} for output split {split!r}"
            )
        if progress:
            progress(
                f"[{split}] hidden source={source_split} "
                f"modes={len(split_ids[source_split])} "
                f"states_per_M={counts.get(split, 0)}"
            )
        states = generate_split_states(
            split_mode_ids=split_ids[source_split],
            target_counts=target_counts,
            states_per_count=counts.get(split, 0),
            seed=seed + SPLIT_NAMES.index(split),
            max_evidence=max_evidence,
            beam_width=beam_width,
            separation_buckets=separation_buckets,
            split_name=split,
            progress=progress,
            progress_every=progress_every,
            exclude_state_ids=seen_state_ids,
            mode_table=table,
        )
        max_rows = row_caps.get(split)
        if max_rows is not None and max_rows >= 0 and len(states) > max_rows:
            states = _balanced_state_cap(
                states,
                max_rows=max_rows,
                seed=seed + 1000 + SPLIT_NAMES.index(split),
            )
        seen_state_ids.update(state.state_id for state in states)
        if progress:
            progress(f"[{split}] writing {len(states)} states")
        state_path = write_table(
            state_rows(states, mode_table=table),
            root / f"states_{split}.{suffix}",
        )
        if progress:
            progress(f"[{split}] building SFT rows")
        sft_path = write_table(
            sft_rows_for_states(
                states,
                targets_per_state=targets_per_state,
                seed=seed + SPLIT_NAMES.index(split),
                allowed_target_mode_ids=split_ids[split],
                mode_table=table,
            ),
            root / f"sft_{split}.{suffix}",
        )
        outputs[f"states_{split}"] = state_path
        outputs[f"sft_{split}"] = sft_path
        if include_verl:
            if progress:
                progress(f"[{split}] building veRL rows")
            verl_path = write_table(
                verl_rows_for_states(
                    states,
                    agent_name=verl_agent_name,
                    data_source=f"causal_micro_lab_{split}",
                    task_overrides=verl_task_overrides,
                    agent_overrides=verl_agent_overrides,
                    mode_table=table,
                ),
                root / f"verl_{split}.{suffix}",
            )
            outputs[f"verl_{split}"] = verl_path
        if progress:
            progress(f"[{split}] done")

    manifest = {
        "target_counts": list(target_counts),
        "states_per_count": counts,
        "max_rows_per_split": row_caps,
        "seed": seed,
        "max_evidence": max_evidence,
        "beam_width": beam_width,
        "targets_per_state": targets_per_state,
        "suffix": suffix,
        "separation_buckets": list(separation_buckets),
        "include_verl": include_verl,
        "verl_task_overrides": verl_task_overrides or {},
        "verl_agent_overrides": verl_agent_overrides or {},
        "verl_agent_name": verl_agent_name,
        "mode_split_counts": {
            split: len(split_ids[split])
            for split in SPLIT_NAMES
        },
        "files": {
            name: str(path.relative_to(root))
            for name, path in sorted(outputs.items())
        },
    }
    manifest_path = write_jsonl([manifest], root / "manifest.jsonl")
    outputs["manifest"] = manifest_path
    if progress:
        progress(f"[done] wrote manifest: {manifest_path}")
    return outputs
