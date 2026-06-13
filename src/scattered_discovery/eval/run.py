from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scattered_discovery.backends import (
    ChatBackend,
    ChatMessage,
    ChatOptions,
    ChatResponse,
    OllamaBackend,
)
from scattered_discovery.config import ExperimentConfig, load_config
from scattered_discovery.envs.scattered_causal import CommitScore, ScatteredDiscoveryEnv
from scattered_discovery.envs.scattered_dsl import (
    extract_action_text,
    parse_action_line,
)
from scattered_discovery.prompts.scattered import (
    SYSTEM_PROMPT,
    finalizer_prompt,
    final_commit_prompt,
    initial_user_prompt,
    observation_prompt,
    repair_prompt,
)


def build_backend(config: ExperimentConfig) -> ChatBackend:
    if config.model.provider == "ollama":
        return OllamaBackend(
            model=config.model.model,
            base_url=config.model.base_url,
            temperature=config.model.temperature,
            top_p=config.model.top_p,
            num_predict=config.model.num_predict,
            request_timeout_s=config.model.request_timeout_s,
            think=config.model.think,
        )
    raise ValueError(f"Unsupported provider: {config.model.provider}")


def run_episode(
    *,
    config: ExperimentConfig,
    backend: ChatBackend,
    dispersion: float,
    world_index: int,
    rollout_index: int,
    output_transcript: bool = False,
) -> dict[str, Any]:
    world_seed = config.eval.seed + int(dispersion * 1000) * 10_000 + world_index
    episode_seed = world_seed * 1009 + rollout_index * 9173 + 19
    budget = config.world.base_budget
    max_commit = 1
    if config.eval.protocol == "set":
        multiplier = config.eval.set_budget_multiplier or config.eval.rollouts_per_world
        budget = config.world.base_budget * multiplier
        max_commit = config.eval.rollouts_per_world

    env = ScatteredDiscoveryEnv(
        config.world,
        world_seed=world_seed,
        episode_seed=episode_seed,
        dispersion=dispersion,
        budget=budget,
        protocol=config.eval.protocol,
        max_commit=max_commit,
    )
    messages = [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=initial_user_prompt(env, config.agent)),
    ]
    transcript: list[dict[str, Any]] = [
        {"role": message.role, "content": message.content} for message in messages
    ]
    steps = 0
    parse_failures = 0
    syntax_repairs = 0
    finalizer_attempts = 0
    model_seconds = 0.0

    for _ in range(config.eval.max_steps):
        steps += 1
        started = time.monotonic()
        response = backend.chat(messages)
        model_seconds += time.monotonic() - started
        response, repaired = maybe_repair_response(
            response=response,
            messages=messages,
            backend=backend,
            env=env,
            config=config,
        )
        model_seconds += repaired["seconds"]
        syntax_repairs += repaired["attempts"]
        finalizer_attempts += repaired["finalizer_attempts"]
        result = env.step(response.content)
        if not result.parse_ok:
            parse_failures += 1
        if response.content.strip():
            messages.append(ChatMessage(role="assistant", content=response.content))
        messages.append(
            ChatMessage(
                role="user",
                content=observation_prompt(env, result.observation, config.agent),
            )
        )
        transcript.append(_assistant_transcript(response))
        transcript.append({"role": "user", "content": result.observation})
        if result.done:
            break
        if env.budget <= 0:
            break

    if not env.done and config.eval.final_commit_attempt:
        started = time.monotonic()
        messages.append(
            ChatMessage(role="user", content=final_commit_prompt(env, config.agent))
        )
        response = backend.chat(messages)
        model_seconds += time.monotonic() - started
        response, repaired = maybe_repair_response(
            response=response,
            messages=messages,
            backend=backend,
            env=env,
            config=config,
        )
        model_seconds += repaired["seconds"]
        syntax_repairs += repaired["attempts"]
        finalizer_attempts += repaired["finalizer_attempts"]
        result = env.step(response.content)
        if not result.parse_ok:
            parse_failures += 1
        transcript.append({"role": "user", "content": "FINAL_COMMIT_PROMPT"})
        transcript.append(_assistant_transcript(response))

    score: CommitScore
    if env.last_score is None:
        score = env.force_empty_commit()
    else:
        score = env.last_score

    score_dict = asdict(score)
    committed = int(score_dict["committed_count"])
    valid_committed = int(score_dict["valid_committed_count"])
    duplicates = int(score_dict["duplicate_count"])
    score_dict["validity"] = valid_committed / committed if committed else 0.0
    score_dict["uniqueness"] = (
        (committed - duplicates) / committed if committed else 0.0
    )
    score_dict["recovery"] = (
        int(score_dict["valid_unique_count"]) / env.target_count
        if env.target_count
        else 0.0
    )

    diagnostics = env.diagnostics()
    record: dict[str, Any] = {
        "protocol": config.eval.protocol,
        "model": config.model.model,
        "dispersion": dispersion,
        "world_index": world_index,
        "rollout_index": rollout_index,
        "world_seed": world_seed,
        "episode_seed": episode_seed,
        "steps": steps,
        "budget_initial": env.initial_budget,
        "budget_used": env.initial_budget - env.budget,
        "parse_failures": parse_failures,
        "syntax_repairs": syntax_repairs,
        "finalizer_attempts": finalizer_attempts,
        "invalid_actions": env.invalid_actions,
        "model_seconds": model_seconds,
        "score": score_dict,
        "target_count": env.target_count,
        "diagnostics": diagnostics,
    }
    if output_transcript:
        record["transcript"] = transcript
    return record


def maybe_repair_response(
    *,
    response: ChatResponse,
    messages: list[ChatMessage],
    backend: ChatBackend,
    env: ScatteredDiscoveryEnv,
    config: ExperimentConfig,
) -> tuple[ChatResponse, dict[str, float | int]]:
    action_text = extract_action_text(response.content)
    error = "missing ACTION line"
    if action_text is not None:
        try:
            parse_action_line(action_text)
            return response, {"attempts": 0, "seconds": 0.0, "finalizer_attempts": 0}
        except Exception as exc:
            error = str(exc)

    seconds = 0.0
    finalizer_attempts = 0
    if (
        config.model.finalize_empty_content
        and not response.content.strip()
        and response.thinking.strip()
    ):
        thinking_trace = _clip_tail(
            response.thinking,
            max_chars=config.model.finalizer_max_thinking_chars,
        )
        finalizer_messages = [
            *messages,
            ChatMessage(
                role="user",
                content=finalizer_prompt(env, thinking_trace, error, config.agent),
            ),
        ]
        started = time.monotonic()
        finalized = backend.chat(
            finalizer_messages,
            options=ChatOptions(
                think=False, num_predict=config.model.finalizer_num_predict
            ),
        )
        seconds += time.monotonic() - started
        finalizer_attempts += 1
        if response.thinking and not finalized.thinking:
            finalized = ChatResponse(
                content=finalized.content, thinking=response.thinking
            )
        action_text = extract_action_text(finalized.content)
        if action_text is not None:
            try:
                parse_action_line(action_text)
                return finalized, {
                    "attempts": 0,
                    "seconds": seconds,
                    "finalizer_attempts": finalizer_attempts,
                }
            except Exception as exc:
                error = str(exc)
        else:
            error = "missing ACTION line"
        response = finalized

    repair_messages = [*messages]
    if response.content.strip():
        repair_messages.append(ChatMessage(role="assistant", content=response.content))
    repair_messages.append(
        ChatMessage(role="user", content=repair_prompt(env, error, config.agent))
    )
    started = time.monotonic()
    repaired = backend.chat(
        repair_messages,
        options=ChatOptions(
            think=False, num_predict=config.model.finalizer_num_predict
        ),
    )
    seconds += time.monotonic() - started
    if response.thinking and not repaired.thinking:
        repaired = ChatResponse(content=repaired.content, thinking=response.thinking)
    return repaired, {
        "attempts": 1,
        "seconds": seconds,
        "finalizer_attempts": finalizer_attempts,
    }


def _assistant_transcript(response: ChatResponse) -> dict[str, Any]:
    item: dict[str, Any] = {"role": "assistant", "content": response.content}
    if response.thinking:
        item["thinking"] = response.thinking
    return item


def _clip_tail(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return "[thinking trace clipped]\n" + text[-max_chars:]


def summarize(
    records: list[dict[str, Any]], config: ExperimentConfig
) -> dict[str, Any]:
    by_dispersion: dict[str, dict[str, Any]] = {}
    for dispersion in config.sweep.dispersions:
        subset = [item for item in records if item["dispersion"] == dispersion]
        if not subset:
            continue
        world_groups: dict[int, list[dict[str, Any]]] = {}
        for item in subset:
            world_groups.setdefault(item["world_index"], []).append(item)

        recovery_values = []
        valid_unique_counts = []
        validity_values = []
        uniqueness_values = []
        for group in world_groups.values():
            recovered_branches: set[int] = set()
            committed = 0
            valid_committed = 0
            duplicates = 0
            target_count = group[0]["target_count"]
            for item in group:
                score = item["score"]
                recovered_branches.update(score["valid_branch_ids"])
                committed += score["committed_count"]
                valid_committed += score.get("valid_committed_count", 0)
                duplicates += score["duplicate_count"]
            recovery_values.append(
                len(recovered_branches) / target_count if target_count else 0.0
            )
            valid_unique_counts.append(len(recovered_branches))
            denominator = committed if committed else 1
            validity_values.append(valid_committed / denominator)
            uniqueness_values.append((committed - duplicates) / denominator)

        by_dispersion[str(dispersion)] = {
            "episodes": len(subset),
            "worlds": len(world_groups),
            "recovery_at_k_mean": _mean(recovery_values),
            "recovery_at_k_values": recovery_values,
            "valid_unique_mean": _mean(valid_unique_counts),
            "pass_at_1": _mean(
                [
                    1.0 if item["score"]["valid_unique_count"] > 0 else 0.0
                    for item in subset
                ]
            ),
            "reward_mean": _mean([item["score"]["reward"] for item in subset]),
            "validity_mean": _mean(validity_values),
            "uniqueness_mean": _mean(uniqueness_values),
            "non_final_count_mean": _mean(
                [item["score"].get("non_final_count", 0) for item in subset]
            ),
            "invalid_actions_mean": _mean([item["invalid_actions"] for item in subset]),
            "parse_failures_mean": _mean([item["parse_failures"] for item in subset]),
            "syntax_repairs_mean": _mean([item["syntax_repairs"] for item in subset]),
            "finalizer_attempts_mean": _mean(
                [item["finalizer_attempts"] for item in subset]
            ),
            "budget_used_mean": _mean([item["budget_used"] for item in subset]),
            "model_seconds_mean": _mean([item["model_seconds"] for item in subset]),
        }

    all_recoveries = [
        value
        for dispersion_summary in by_dispersion.values()
        for value in dispersion_summary["recovery_at_k_values"]
    ]
    return {
        "run_name": config.eval.run_name,
        "model": config.model.model,
        "protocol": config.eval.protocol,
        "num_worlds": config.eval.num_worlds,
        "rollouts_per_world": config.eval.rollouts_per_world,
        "max_steps": config.eval.max_steps,
        "dispersions": config.sweep.dispersions,
        "overall": {
            "episodes": len(records),
            "recovery_at_k_mean": _mean(all_recoveries),
            "pass_at_1": _mean(
                [
                    1.0 if item["score"]["valid_unique_count"] > 0 else 0.0
                    for item in records
                ]
            ),
            "reward_mean": _mean([item["score"]["reward"] for item in records]),
            "validity_mean": _mean(
                [float(item["score"].get("validity", 0.0)) for item in records]
            ),
            "uniqueness_mean": _mean(
                [float(item["score"].get("uniqueness", 0.0)) for item in records]
            ),
            "non_final_count_mean": _mean(
                [item["score"].get("non_final_count", 0) for item in records]
            ),
            "invalid_actions_mean": _mean(
                [item["invalid_actions"] for item in records]
            ),
            "parse_failures_mean": _mean([item["parse_failures"] for item in records]),
            "syntax_repairs_mean": _mean([item["syntax_repairs"] for item in records]),
            "finalizer_attempts_mean": _mean(
                [item["finalizer_attempts"] for item in records]
            ),
            "budget_used_mean": _mean([item["budget_used"] for item in records]),
            "model_seconds_mean": _mean([item["model_seconds"] for item in records]),
        },
        "by_dispersion": by_dispersion,
    }


def _mean(values: list[float | int]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def write_outputs(
    *,
    config: ExperimentConfig,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    config_path: Path,
) -> Path:
    output_dir = Path(config.eval.output_dir) / config.eval.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with (output_dir / "config_source.txt").open("w", encoding="utf-8") as handle:
        handle.write(config_path.read_text(encoding="utf-8"))
    return output_dir


def prepare_output_dir(config: ExperimentConfig, config_path: Path) -> Path:
    output_dir = Path(config.eval.output_dir) / config.eval.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "episodes.jsonl").write_text("", encoding="utf-8")
    with (output_dir / "config_source.txt").open("w", encoding="utf-8") as handle:
        handle.write(config_path.read_text(encoding="utf-8"))
    return output_dir


def append_checkpoint(output_dir: Path, record: dict[str, Any]) -> None:
    with (output_dir / "episodes.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, help="Path to YAML experiment config."
    )
    parser.add_argument("--model", default=None, help="Override model name.")
    parser.add_argument(
        "--transcripts", action="store_true", help="Store full transcripts."
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    if args.model:
        config = ExperimentConfig(
            world=config.world,
            sweep=config.sweep,
            eval=config.eval,
            model=type(config.model)(**{**asdict(config.model), "model": args.model}),
            agent=config.agent,
        )
    backend = build_backend(config)
    output_dir = prepare_output_dir(config, config_path)

    records: list[dict[str, Any]] = []
    for dispersion in config.sweep.dispersions:
        for world_index in range(config.eval.num_worlds):
            rollout_count = (
                config.eval.rollouts_per_world
                if config.eval.protocol == "single"
                else 1
            )
            for rollout_index in range(rollout_count):
                print(
                    f"running model={config.model.model} protocol={config.eval.protocol} "
                    f"dispersion={dispersion} world={world_index} rollout={rollout_index}",
                    flush=True,
                )
                record = run_episode(
                    config=config,
                    backend=backend,
                    dispersion=dispersion,
                    world_index=world_index,
                    rollout_index=rollout_index,
                    output_transcript=args.transcripts,
                )
                records.append(record)
                append_checkpoint(output_dir, record)

    summary = summarize(records, config)
    output_dir = write_outputs(
        config=config,
        records=records,
        summary=summary,
        config_path=config_path,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote results to {output_dir}")


if __name__ == "__main__":
    main()
