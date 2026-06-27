from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from typing import Any


@dataclass(frozen=True)
class RewardConfig:
    """Reward values shared by environment constructors and dataset specs."""

    valid_hypothesis_reward: float
    false_penalty: float
    non_final_penalty: float
    unsupported_penalty: float
    budget_penalty: float
    clean_invalid_final_reward: float
    format_reward: float
    admissible_reward: float
    commit_format_reward: float
    invalid_action_penalty: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def with_overrides(self, overrides: dict[str, Any] | None) -> "RewardConfig":
        if not overrides:
            return self
        unknown = sorted(set(overrides) - REWARD_FIELDS)
        if unknown:
            raise ValueError(
                f"Unknown reward field(s): {', '.join(unknown)}. "
                f"Allowed fields: {', '.join(sorted(REWARD_FIELDS))}."
            )
        return replace(self, **{key: float(value) for key, value in overrides.items()})

    def world_kwargs(self) -> dict[str, float]:
        return {
            "valid_hypothesis_reward": self.valid_hypothesis_reward,
            "false_penalty": self.false_penalty,
            "non_final_penalty": self.non_final_penalty,
            "unsupported_penalty": self.unsupported_penalty,
            "budget_penalty": self.budget_penalty,
            "clean_invalid_final_reward": self.clean_invalid_final_reward,
        }

    def shaping_kwargs(self) -> dict[str, float]:
        return {
            "format_reward": self.format_reward,
            "admissible_reward": self.admissible_reward,
            "commit_format_reward": self.commit_format_reward,
            "invalid_action_penalty": self.invalid_action_penalty,
        }


TERMINAL_ONLY_REWARD = RewardConfig(
    valid_hypothesis_reward=1.0,
    false_penalty=0.0,
    non_final_penalty=0.0,
    unsupported_penalty=0.0,
    budget_penalty=0.0,
    clean_invalid_final_reward=0.0,
    format_reward=0.0,
    admissible_reward=0.0,
    commit_format_reward=0.0,
    invalid_action_penalty=0.0,
)

TERMINAL_CLEAN_INVALID_BONUS_REWARD = RewardConfig(
    valid_hypothesis_reward=1.0,
    false_penalty=0.0,
    non_final_penalty=0.0,
    unsupported_penalty=0.0,
    budget_penalty=0.0,
    clean_invalid_final_reward=0.2,
    format_reward=0.0,
    admissible_reward=0.0,
    commit_format_reward=0.0,
    invalid_action_penalty=0.0,
)

SHAPED_REWARD = RewardConfig(
    valid_hypothesis_reward=1.0,
    false_penalty=0.5,
    non_final_penalty=0.25,
    unsupported_penalty=0.25,
    budget_penalty=0.0,
    clean_invalid_final_reward=0.0,
    format_reward=0.03,
    admissible_reward=0.02,
    commit_format_reward=0.05,
    invalid_action_penalty=0.05,
)

DEFAULT_REWARD_PROFILE = "terminal_only"

REWARD_PROFILES: dict[str, RewardConfig] = {
    "terminal_only": TERMINAL_ONLY_REWARD,
    "terminal_clean_invalid_bonus": TERMINAL_CLEAN_INVALID_BONUS_REWARD,
    "shaped": SHAPED_REWARD,
}

HYPO_CAUSAL_REWARD = TERMINAL_ONLY_REWARD
HYPO_BOOLEAN_REWARD = TERMINAL_ONLY_REWARD
HYPO_3D_REWARD = TERMINAL_ONLY_REWARD
SCATTERED_CAUSAL_REWARD = TERMINAL_ONLY_REWARD
CAUSAL_MICRO_LAB_REWARD = TERMINAL_ONLY_REWARD

REWARD_DEFAULT_PROFILES: dict[str, str] = {
    "causal_micro_lab": DEFAULT_REWARD_PROFILE,
    "hypospace_causal": DEFAULT_REWARD_PROFILE,
    "hypospace_boolean": DEFAULT_REWARD_PROFILE,
    "hypospace_3d": DEFAULT_REWARD_PROFILE,
    "scattered_causal": DEFAULT_REWARD_PROFILE,
}

REWARD_DEFAULTS: dict[str, RewardConfig] = {
    "causal_micro_lab": CAUSAL_MICRO_LAB_REWARD,
    "hypospace_causal": HYPO_CAUSAL_REWARD,
    "hypospace_boolean": HYPO_BOOLEAN_REWARD,
    "hypospace_3d": HYPO_3D_REWARD,
    "scattered_causal": SCATTERED_CAUSAL_REWARD,
}

REWARD_FIELDS = set(RewardConfig.__dataclass_fields__)


def reward_config_for_profile(profile: str) -> RewardConfig:
    try:
        return REWARD_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported reward profile: {profile}. "
            f"Allowed profiles: {', '.join(sorted(REWARD_PROFILES))}."
        ) from exc


def reward_config_for_env(env_type: str) -> RewardConfig:
    try:
        return REWARD_DEFAULTS[env_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported env_type for rewards: {env_type}") from exc


def reward_config_from_task(env_type: str, task: dict[str, Any]) -> RewardConfig:
    """Build an env-specific reward config from an EnvSpec task mapping.

    Preferred YAML shapes are:

    ```yaml
    task:
      reward_profile: terminal_only
    ```

    or:

    ```yaml
    task:
      reward:
        profile: shaped
        valid_hypothesis_reward: 1.0
    ```

    Flat task-level reward fields are also accepted for small one-off specs.
    """

    if env_type not in REWARD_DEFAULT_PROFILES:
        raise ValueError(f"Unsupported env_type for rewards: {env_type}")

    flat_overrides = {key: task[key] for key in REWARD_FIELDS if key in task}
    nested_raw = task.get("reward") or {}
    if nested_raw and not isinstance(nested_raw, dict):
        raise ValueError("task.reward must be a mapping when provided.")
    nested = dict(nested_raw)
    nested_profile = nested.pop("profile", None)
    task_profile = task.get("reward_profile")
    if (
        task_profile is not None
        and nested_profile is not None
        and str(task_profile) != str(nested_profile)
    ):
        raise ValueError("task.reward_profile and task.reward.profile disagree.")
    profile = str(
        task_profile
        or nested_profile
        or REWARD_DEFAULT_PROFILES.get(env_type, DEFAULT_REWARD_PROFILE)
    )
    base = reward_config_for_profile(profile)
    return base.with_overrides({**flat_overrides, **nested})


def reward_defaults_as_dict() -> dict[str, dict[str, float]]:
    return {name: config.as_dict() for name, config in REWARD_DEFAULTS.items()}


def reward_profiles_as_dict() -> dict[str, dict[str, float]]:
    return {name: config.as_dict() for name, config in REWARD_PROFILES.items()}


def duplicate_set_zeroes_reward(protocol: str, duplicate_count: int) -> bool:
    """Puri-style uniqueness rule for multi-answer sets.

    Duplicates are not a negative scalar penalty. In set mode, duplicate answers
    make the generated set fail the uniqueness/format condition, so the final
    reward is zeroed.
    """

    return protocol == "set" and duplicate_count > 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Print reward defaults.")
    parser.add_argument("--env-type", choices=sorted(REWARD_DEFAULTS))
    parser.add_argument("--profile", choices=sorted(REWARD_PROFILES))
    args = parser.parse_args()

    if args.env_type and args.profile:
        payload = {
            args.env_type: reward_config_for_profile(args.profile).as_dict(),
            "profile": args.profile,
        }
    elif args.env_type:
        payload: dict[str, Any] = {
            args.env_type: reward_config_for_env(args.env_type).as_dict()
        }
    elif args.profile:
        payload = {args.profile: reward_config_for_profile(args.profile).as_dict()}
    else:
        payload = {
            "default_profile": DEFAULT_REWARD_PROFILE,
            "env_default_profiles": REWARD_DEFAULT_PROFILES,
            "env_defaults": reward_defaults_as_dict(),
            "profiles": reward_profiles_as_dict(),
        }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
