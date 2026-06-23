from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from scattered_discovery.backends import ChatBackend, ChatMessage, OllamaBackend
from scattered_discovery.envs.base import EnvSpec
from scattered_discovery.envs.factory import make_env


def run_local_episode(
    *,
    spec: EnvSpec | dict[str, Any],
    backend: ChatBackend,
    max_steps: int | None = None,
    max_consecutive_invalid: int | None = None,
    output_transcript: bool = False,
) -> dict[str, Any]:
    env = make_env(spec)
    step_limit = (
        spec.max_steps if isinstance(spec, EnvSpec) else int(spec.get("max_steps", 8))
    )
    if max_steps is not None:
        step_limit = max_steps
    invalid_limit = (
        spec.max_consecutive_invalid
        if isinstance(spec, EnvSpec)
        else int(spec.get("max_consecutive_invalid", 2))
    )
    if max_consecutive_invalid is not None:
        invalid_limit = max_consecutive_invalid

    messages = [
        ChatMessage(role="system", content=env.system_prompt("local")),
        ChatMessage(role="user", content=env.reset()),
    ]
    transcript: list[dict[str, Any]] = [
        {"role": message.role, "content": message.content} for message in messages
    ]
    model_seconds = 0.0
    steps = 0
    final_score = None
    consecutive_invalid = 0
    early_stop_reason: str | None = None

    for _ in range(step_limit):
        steps += 1
        started = time.monotonic()
        response = backend.chat(messages)
        model_seconds += time.monotonic() - started
        result = env.step(response.content)
        invalid_step = (not result.parse_ok) or (
            "not admissible" in result.observation.lower()
        )
        consecutive_invalid = consecutive_invalid + 1 if invalid_step else 0
        if response.content.strip():
            messages.append(ChatMessage(role="assistant", content=response.content))
        messages.append(
            ChatMessage(
                role="user",
                content=env.observation_prompt(result, "local"),
            )
        )
        if output_transcript:
            item: dict[str, Any] = {"role": "assistant", "content": response.content}
            if response.thinking:
                item["thinking"] = response.thinking
            transcript.append(item)
            transcript.append({"role": "user", "content": result.observation})
        if result.done:
            final_score = result.score
            break
        if invalid_limit > 0 and consecutive_invalid >= invalid_limit:
            early_stop_reason = "consecutive_invalid_actions"
            break

    score = final_score if final_score is not None else env.force_finalize()
    score.metrics["early_stop_reason"] = early_stop_reason
    score.metrics["max_consecutive_invalid"] = invalid_limit
    score.metrics["consecutive_invalid_at_stop"] = consecutive_invalid

    record: dict[str, Any] = {
        "steps": steps,
        "model_seconds": model_seconds,
        "score": score.as_dict() if hasattr(score, "as_dict") else score,
        "diagnostics": env.diagnostics(),
    }
    record["diagnostics"]["early_stop_reason"] = early_stop_reason
    record["diagnostics"]["max_consecutive_invalid"] = invalid_limit
    record["diagnostics"]["consecutive_invalid_at_stop"] = consecutive_invalid
    if output_transcript:
        record["transcript"] = transcript
    return record


def _load_spec(path: str | Path) -> EnvSpec:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return EnvSpec(**raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, help="Path to an EnvSpec JSON file.")
    parser.add_argument("--model", default="qwen3:1.7b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-consecutive-invalid", type=int, default=None)
    parser.add_argument("--num-predict", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--transcripts", action="store_true")
    args = parser.parse_args()

    backend = OllamaBackend(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        num_predict=args.num_predict,
        think=True,
    )
    record = run_local_episode(
        spec=_load_spec(args.spec),
        backend=backend,
        max_steps=args.max_steps,
        max_consecutive_invalid=args.max_consecutive_invalid,
        output_transcript=args.transcripts,
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
