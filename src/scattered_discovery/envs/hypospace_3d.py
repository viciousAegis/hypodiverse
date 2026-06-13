from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from random import Random
import re
from typing import Any, Literal

from scattered_discovery.envs.base import DiscoveryScore, DiscoveryStep, RewardBreakdown
from scattered_discovery.prompts.generic import (
    next_action_observation_prompt,
    system_prompt_for_runtime,
)
from scattered_discovery.prompts.hypospace import reconstruction_3d_initial_prompt
from scattered_discovery.rewards import (
    HYPO_3D_REWARD,
    RewardConfig,
    duplicate_set_zeroes_reward,
)


@dataclass(frozen=True)
class HeightStructure:
    heights: tuple[tuple[int, ...], ...]
    max_height: int

    @property
    def grid_size(self) -> int:
        return len(self.heights)

    def key(self) -> str:
        rows = ["".join(str(value) for value in row) for row in self.heights]
        return "heights:" + "/".join(rows)

    def format(self) -> str:
        rows = [" ".join(str(value) for value in row) for row in self.heights]
        return "heights([" + "; ".join(rows) + "])"

    def top_view(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(1 if value > 0 else 0 for value in row) for row in self.heights
        )

    def front_view(self) -> tuple[int, ...]:
        return tuple(
            max(self.heights[row][col] for row in range(self.grid_size))
            for col in range(self.grid_size)
        )

    def side_view(self) -> tuple[int, ...]:
        return tuple(max(row) for row in self.heights)

    def block_count(self) -> int:
        return sum(1 for row in self.heights for value in row if value > 0)


def enumerate_structures(
    *,
    grid_size: int,
    max_height: int,
    max_blocks: int | None = None,
) -> tuple[HeightStructure, ...]:
    values = range(max_height + 1)
    structures: list[HeightStructure] = []
    for flat in product(values, repeat=grid_size * grid_size):
        if all(value == 0 for value in flat):
            continue
        rows = tuple(
            tuple(flat[row * grid_size : (row + 1) * grid_size])
            for row in range(grid_size)
        )
        structure = HeightStructure(rows, max_height)
        if max_blocks is not None and structure.block_count() > max_blocks:
            continue
        structures.append(structure)
    return tuple(structures)


def _format_matrix(matrix: tuple[tuple[int, ...], ...]) -> str:
    return (
        "[" + "; ".join(" ".join(str(value) for value in row) for row in matrix) + "]"
    )


def parse_height_structure(
    text: str, *, grid_size: int, max_height: int
) -> HeightStructure:
    value = text.strip().strip("`").strip()
    if value.upper().startswith("STRUCTURE:"):
        value = value.split(":", 1)[1].strip()
    if value.lower().startswith(("heights(", "structure(")) and value.endswith(")"):
        value = value[value.find("(") + 1 : -1].strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1].strip()
    row_texts = [row.strip() for row in value.split(";") if row.strip()]
    if len(row_texts) != grid_size:
        raise ValueError(f"expected {grid_size} rows in height grid")
    rows: list[tuple[int, ...]] = []
    for row_text in row_texts:
        parts = [part for part in re.split(r"[\s,]+", row_text.strip("[] ")) if part]
        if len(parts) != grid_size:
            raise ValueError(f"expected {grid_size} values per row")
        row = tuple(int(part) for part in parts)
        if any(value < 0 or value > max_height for value in row):
            raise ValueError(f"height values must be in [0, {max_height}]")
        rows.append(row)
    structure = HeightStructure(tuple(rows), max_height)
    if structure.block_count() == 0:
        raise ValueError("empty structures are not in the hypothesis space")
    return structure


def _extract_action_text(model_text: str) -> str:
    lines = [line.strip() for line in model_text.splitlines() if line.strip()]
    for line in lines:
        cleaned = line.strip("`").strip()
        if cleaned.upper().startswith("ACTION:"):
            return cleaned.split(":", 1)[1].strip()
    return model_text.strip()


def _split_commit_payload(payload: str) -> list[str]:
    value = payload.strip()
    if value.startswith("[") or value.endswith("]"):
        if not value.startswith("[") or not value.endswith("]"):
            raise ValueError("COMMIT set payload requires [heights(...)| heights(...)]")
        parts = [part.strip() for part in value[1:-1].split("|") if part.strip()]
        if not parts:
            raise ValueError("COMMIT set payload requires at least one structure")
        return parts
    return [value]


class HypoSpace3DEnv:
    """Interactive 3D HypoSpace variant over supported height grids."""

    def __init__(
        self,
        *,
        grid_size: int = 2,
        max_height: int = 3,
        max_blocks: int | None = 3,
        target_heights: tuple[tuple[int, ...], ...] | None = None,
        seed: int = 0,
        query_budget: int | None = None,
        protocol: str = "single",
        max_commit: int = 1,
        valid_hypothesis_reward: float | None = None,
        false_penalty: float | None = None,
        unsupported_penalty: float | None = None,
        format_reward: float | None = None,
        admissible_reward: float | None = None,
        commit_format_reward: float | None = None,
        invalid_action_penalty: float | None = None,
        show_version_space_size: bool = False,
        reward_config: RewardConfig | None = None,
    ) -> None:
        rewards = reward_config or HYPO_3D_REWARD
        self.grid_size = grid_size
        self.max_height = max_height
        self.max_blocks = max_blocks
        self.hypothesis_space = enumerate_structures(
            grid_size=grid_size,
            max_height=max_height,
            max_blocks=max_blocks,
        )
        if target_heights is None:
            self.target = Random(seed).choice(self.hypothesis_space)
        else:
            self.target = HeightStructure(
                tuple(tuple(row) for row in target_heights), max_height
            )
        self.query_budget = query_budget if query_budget is not None else 2
        self.initial_budget = self.query_budget
        self.protocol = protocol
        self.max_commit = max_commit
        self.valid_hypothesis_reward = (
            rewards.valid_hypothesis_reward
            if valid_hypothesis_reward is None
            else valid_hypothesis_reward
        )
        self.false_penalty = (
            rewards.false_penalty if false_penalty is None else false_penalty
        )
        self.unsupported_penalty = (
            rewards.unsupported_penalty
            if unsupported_penalty is None
            else unsupported_penalty
        )
        self.show_version_space_size = show_version_space_size
        self._format_reward = (
            rewards.format_reward if format_reward is None else format_reward
        )
        self._admissible_reward = (
            rewards.admissible_reward
            if admissible_reward is None
            else admissible_reward
        )
        self._commit_format_reward = (
            rewards.commit_format_reward
            if commit_format_reward is None
            else commit_format_reward
        )
        self._invalid_action_penalty = (
            rewards.invalid_action_penalty
            if invalid_action_penalty is None
            else invalid_action_penalty
        )
        self._breakdown = RewardBreakdown()
        self._observations: dict[str, Any] = {}
        self._done = False
        self._last_score: DiscoveryScore | None = None
        self._parse_failures = 0
        self._invalid_actions = 0
        self._final_keys = tuple(
            structure.key() for structure in self._top_version_space()
        )

    @property
    def done(self) -> bool:
        return self._done

    def system_prompt(self, runtime: Literal["local", "verl"] = "local") -> str:
        return system_prompt_for_runtime(runtime)

    def reset(self) -> str:
        return reconstruction_3d_initial_prompt(
            grid_size=self.grid_size,
            max_height=self.max_height,
            max_blocks=self.max_blocks,
            query_budget=self.query_budget,
        )

    def observation_prompt(
        self,
        step: DiscoveryStep,
        runtime: Literal["local", "verl"] = "local",
    ) -> str:
        del runtime
        return next_action_observation_prompt(step.observation)

    def step(self, model_text_or_action: str) -> DiscoveryStep:
        if self._done:
            return DiscoveryStep(
                "Episode is already done.", True, False, score=self._last_score
            )
        action_text = _extract_action_text(model_text_or_action)
        upper = action_text.upper()
        try:
            if upper.startswith("VIEW "):
                return self._view(action_text[5:].strip().lower(), action_text)
            if upper.startswith("PROBE "):
                return self._probe(action_text[6:].strip(), action_text)
            if upper.startswith("COMMIT "):
                return self._commit(
                    [
                        self._ensure_in_space(
                            parse_height_structure(
                                part,
                                grid_size=self.grid_size,
                                max_height=self.max_height,
                            )
                        )
                        for part in _split_commit_payload(action_text[7:])
                    ],
                    action_text=action_text,
                )
            if upper.startswith("COMMIT_SET "):
                return self._commit(
                    [
                        self._ensure_in_space(
                            parse_height_structure(
                                part,
                                grid_size=self.grid_size,
                                max_height=self.max_height,
                            )
                        )
                        for part in _split_commit_payload(action_text[11:])
                    ],
                    action_text=action_text,
                )
            raise ValueError("unknown action; use VIEW, PROBE, or COMMIT")
        except Exception as exc:
            self._parse_failures += 1
            self._invalid_actions += 1
            self.query_budget = max(0, self.query_budget - 1)
            self._breakdown = self._breakdown.plus(
                invalid_action=-self._invalid_action_penalty
            )
            return DiscoveryStep(
                observation=f"Invalid action: {exc}. Return exactly one valid ACTION line.",
                done=False,
                parse_ok=False,
                action_text=action_text,
                metrics=self._metrics(),
                debug={"error": str(exc)},
            )

    def force_finalize(self) -> DiscoveryScore:
        score = self._score_commit(())
        self._done = True
        self._last_score = score
        return score

    def diagnostics(self) -> dict[str, Any]:
        return {
            "env_type": "hypospace_3d_interactive",
            "grid_size": self.grid_size,
            "max_height": self.max_height,
            "max_blocks": self.max_blocks,
            "target_structure": self.target.format(),
            "target_key": self.target.key(),
            "final_compatible_keys": list(self._final_keys),
            "current_version_space_size": len(self._current_version_space()),
            "final_version_space_size": len(self._final_keys),
            "observations": self._serializable_observations(),
            "budget_used": self.initial_budget - self.query_budget,
            "budget_remaining": self.query_budget,
            "parse_failures": self._parse_failures,
            "invalid_actions": self._invalid_actions,
            "reward_breakdown_so_far": self._breakdown.as_dict(),
        }

    def _view(self, view: str, action_text: str) -> DiscoveryStep:
        if self.query_budget <= 0:
            return DiscoveryStep(
                "No query budget remains. Submit a COMMIT action.",
                False,
                True,
                action_text=action_text,
                metrics=self._metrics(),
            )
        if view not in {"top", "front", "side"}:
            self._invalid_actions += 1
            self.query_budget = max(0, self.query_budget - 1)
            self._breakdown = self._breakdown.plus(
                format=self._format_reward, invalid_action=-self._invalid_action_penalty
            )
            return DiscoveryStep(
                f"Unknown view {view!r}. Use top, front, or side.",
                False,
                True,
                action_text=action_text,
                metrics=self._metrics(),
            )
        self.query_budget -= 1
        self._breakdown = self._breakdown.plus(
            format=self._format_reward, admissible=self._admissible_reward
        )
        value: Any
        if view == "top":
            value = self.target.top_view()
            rendered = _format_matrix(value)
        elif view == "front":
            value = self.target.front_view()
            rendered = "[" + " ".join(str(item) for item in value) + "]"
        else:
            value = self.target.side_view()
            rendered = "[" + " ".join(str(item) for item in value) + "]"
        self._observations[f"view:{view}"] = value
        observation = f"VIEW {view} -> {rendered}."
        if self.show_version_space_size:
            observation += (
                " Compatible structures remaining: "
                f"{len(self._current_version_space())}."
            )
        return DiscoveryStep(
            observation=observation,
            done=False,
            parse_ok=True,
            action_text=action_text,
            metrics=self._metrics(),
        )

    def _probe(self, payload: str, action_text: str) -> DiscoveryStep:
        if self.query_budget <= 0:
            return DiscoveryStep(
                "No query budget remains. Submit a COMMIT action.",
                False,
                True,
                action_text=action_text,
                metrics=self._metrics(),
            )
        values: dict[str, int] = {}
        for part in payload.split(","):
            if "=" not in part:
                raise ValueError("PROBE requires row=R,col=C")
            name, raw_value = [item.strip().lower() for item in part.split("=", 1)]
            values[name] = int(raw_value)
        row = values.get("row")
        col = values.get("col")
        if (
            row is None
            or col is None
            or row < 0
            or col < 0
            or row >= self.grid_size
            or col >= self.grid_size
        ):
            raise ValueError("PROBE coordinates out of range")
        self.query_budget -= 1
        self._breakdown = self._breakdown.plus(
            format=self._format_reward, admissible=self._admissible_reward
        )
        height = self.target.heights[row][col]
        self._observations[f"probe:{row},{col}"] = height
        observation = f"PROBE row={row},col={col} -> height={height}."
        if self.show_version_space_size:
            observation += (
                " Compatible structures remaining: "
                f"{len(self._current_version_space())}."
            )
        return DiscoveryStep(
            observation=observation,
            done=False,
            parse_ok=True,
            action_text=action_text,
            metrics=self._metrics(),
        )

    def _commit(
        self, structures: list[HeightStructure], *, action_text: str
    ) -> DiscoveryStep:
        if self.protocol == "single" and len(structures) != 1:
            return DiscoveryStep(
                "This episode uses single-answer protocol. Use COMMIT with exactly one structure.",
                False,
                True,
                action_text=action_text,
            )
        if len(structures) > self.max_commit:
            return DiscoveryStep(
                f"Too many committed structures: got {len(structures)}, max is {self.max_commit}.",
                False,
                True,
                action_text=action_text,
            )
        self._breakdown = self._breakdown.plus(
            format=self._format_reward,
            admissible=self._admissible_reward,
            commit_format=self._commit_format_reward,
        )
        score = self._score_commit(tuple(structures))
        self._done = True
        self._last_score = score
        valid = ", ".join(score.valid_keys) if score.valid_keys else "none"
        return DiscoveryStep(
            observation=f"Episode complete. Valid final structures: {valid}. Reward={score.reward:.3f}; valid_unique={score.valid_unique_count}; false={score.false_count}; unsupported={score.unsupported_count}; duplicates={score.duplicate_count}.",
            done=True,
            parse_ok=True,
            action_text=action_text,
            reward=score.reward,
            score=score,
            metrics=self._metrics(),
        )

    def _score_commit(self, structures: tuple[HeightStructure, ...]) -> DiscoveryScore:
        final_keys = set(self._final_keys)
        current_keys = {structure.key() for structure in self._current_version_space()}
        valid_keys: set[str] = set()
        seen: set[str] = set()
        false_count = 0
        unsupported_count = 0
        duplicate_count = 0
        valid_committed_count = 0
        for structure in structures:
            key = structure.key()
            duplicate = key in seen
            if duplicate:
                duplicate_count += 1
            else:
                seen.add(key)
            if key not in current_keys:
                unsupported_count += 1
                continue
            if key not in final_keys:
                false_count += 1
                continue
            valid_committed_count += 1
            if not duplicate:
                valid_keys.add(key)
        if duplicate_set_zeroes_reward(self.protocol, duplicate_count):
            breakdown = RewardBreakdown()
        else:
            breakdown = self._breakdown.plus(
                valid_hypothesis=self.valid_hypothesis_reward * len(valid_keys),
                false_commit=-self.false_penalty * false_count,
                unsupported_commit=-self.unsupported_penalty * unsupported_count,
            )
        final_key_list = tuple(sorted(final_keys))
        return DiscoveryScore(
            reward=breakdown.total,
            breakdown=breakdown,
            valid_keys=tuple(sorted(valid_keys)),
            valid_committed_count=valid_committed_count,
            valid_unique_count=len(valid_keys),
            committed_count=len(structures),
            false_count=false_count,
            unsupported_count=unsupported_count,
            duplicate_count=duplicate_count,
            parse_failures=self._parse_failures,
            invalid_actions=self._invalid_actions,
            metrics={
                "recovery": len(valid_keys) / len(final_keys) if final_keys else 0.0,
                "target_count": len(final_keys),
                "final_version_space_size": len(final_keys),
                "current_version_space_size": len(current_keys),
                "budget_used": self.initial_budget - self.query_budget,
            },
            reward_vector=tuple(
                1.0 if key in valid_keys else 0.0 for key in final_key_list
            ),
        )

    def _ensure_in_space(self, structure: HeightStructure) -> HeightStructure:
        if structure.grid_size != self.grid_size:
            raise ValueError("wrong grid size")
        if self.max_blocks is not None and structure.block_count() > self.max_blocks:
            raise ValueError(f"structure exceeds max occupied cells {self.max_blocks}")
        if structure.key() not in {
            candidate.key() for candidate in self.hypothesis_space
        }:
            raise ValueError("structure is outside the enumerated hypothesis space")
        return structure

    def _top_version_space(self) -> tuple[HeightStructure, ...]:
        target_top = self.target.top_view()
        return tuple(
            structure
            for structure in self.hypothesis_space
            if structure.top_view() == target_top
        )

    def _current_version_space(self) -> tuple[HeightStructure, ...]:
        return tuple(
            structure
            for structure in self.hypothesis_space
            if self._matches_observations(structure)
        )

    def _matches_observations(self, structure: HeightStructure) -> bool:
        for key, value in self._observations.items():
            if key == "view:top" and structure.top_view() != value:
                return False
            if key == "view:front" and structure.front_view() != value:
                return False
            if key == "view:side" and structure.side_view() != value:
                return False
            if key.startswith("probe:"):
                row, col = [int(part) for part in key.split(":", 1)[1].split(",")]
                if structure.heights[row][col] != value:
                    return False
        return True

    def _serializable_observations(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in self._observations.items():
            if isinstance(value, tuple):
                result[key] = [
                    list(row) if isinstance(row, tuple) else row for row in value
                ]
            else:
                result[key] = value
        return result

    def _metrics(self) -> dict[str, Any]:
        return {
            "budget_remaining": self.query_budget,
            "budget_used": self.initial_budget - self.query_budget,
            "observations": len(self._observations),
            "current_version_space_size": len(self._current_version_space()),
            "parse_failures": self._parse_failures,
            "invalid_actions": self._invalid_actions,
            "reward_breakdown_so_far": self._breakdown.as_dict(),
        }
