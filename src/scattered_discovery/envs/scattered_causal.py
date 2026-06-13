from __future__ import annotations

from dataclasses import asdict, dataclass
from random import Random
from typing import Any, Literal

from scattered_discovery.config import WorldConfig
from scattered_discovery.envs.scattered_dsl import (
    CommitAction,
    Edge,
    Expr,
    InterveneAction,
    PathExpr,
    TestAction,
    canonical_key,
    edge_keys_for_path,
    extract_action_text,
    format_expr,
    parse_action_line,
    variables_in_expr,
)
from scattered_discovery.envs.base import DiscoveryScore, DiscoveryStep, RewardBreakdown
from scattered_discovery.envs.scattered_evidence import (
    EvidenceStore,
    EvidenceSummary,
    GaussianEvidenceModel,
)
from scattered_discovery.envs.scattered_world import (
    GeneratedWorld,
    HypothesisInfo,
    WorldGenerator,
)
from scattered_discovery.config import AgentConfig
from scattered_discovery.rewards import (
    SCATTERED_CAUSAL_REWARD,
    RewardConfig,
    duplicate_set_zeroes_reward,
)


@dataclass(frozen=True)
class CommitScore:
    reward: float
    valid_keys: tuple[str, ...]
    valid_branch_ids: tuple[int, ...]
    false_count: int
    non_final_count: int
    unsupported_count: int
    duplicate_count: int
    committed_count: int
    valid_committed_count: int
    valid_unique_count: int


@dataclass(frozen=True)
class StepResult:
    observation: str
    done: bool
    parse_ok: bool
    action_text: str | None
    score: CommitScore | None = None
    debug: dict[str, Any] | None = None


class ScatteredDiscoveryEnv:
    def __init__(
        self,
        config: WorldConfig,
        *,
        world_seed: int,
        episode_seed: int,
        dispersion: float,
        budget: int | None = None,
        protocol: str = "single",
        max_commit: int | None = None,
    ) -> None:
        self.config = config
        self.protocol = protocol
        self.max_commit = max_commit
        self.world: GeneratedWorld = WorldGenerator(
            num_branches=config.num_branches,
            branch_depth=config.branch_depth,
            distractors_per_node=config.distractors_per_node,
        ).generate(seed=world_seed, dispersion=dispersion)
        self.rng = Random(episode_seed)
        self.evidence_model = GaussianEvidenceModel(
            true_mean=config.true_mean,
            false_mean=config.false_mean,
            sigma=config.noise_sigma,
            accept_threshold=config.accept_threshold,
            reject_threshold=config.reject_threshold,
        )
        self.evidence = EvidenceStore(self.evidence_model)
        self.known_variables: set[str] = set(self.world.initial_variables)
        self.accepted_claims: set[str] = set()
        self.rejected_claims: set[str] = set()
        self.budget = config.base_budget if budget is None else budget
        self.initial_budget = self.budget
        self.done = False
        self.last_score: CommitScore | None = None
        self.invalid_actions = 0

    @property
    def terminal_count(self) -> int:
        return self.target_count

    @property
    def target_count(self) -> int:
        return len(self.world.terminal_keys)

    def public_state_text(
        self,
        max_evidence_items: int = 12,
        *,
        include_evidence_status: bool = False,
    ) -> str:
        evidence_items = sorted(
            self.evidence.summaries(),
            key=lambda item: (-item.samples, item.key),
        )[:max_evidence_items]
        evidence = "\n".join(
            self._format_evidence_summary(
                item, include_evidence_status=include_evidence_status
            )
            for item in evidence_items
        )
        if not evidence:
            evidence = "- none"
        lines = [
            f"Budget remaining: {self.budget}",
            f"Known variables: {', '.join(sorted(self.known_variables))}",
        ]
        if include_evidence_status:
            lines.extend(
                [
                    f"Accepted claims: {self._format_claim_set(self.accepted_claims)}",
                    f"Rejected claims: {self._format_claim_set(self.rejected_claims)}",
                ]
            )
        lines.append(f"Evidence summary:\n{evidence}")
        return "\n".join(lines)

    def step(self, model_text_or_action: str) -> StepResult:
        if self.done:
            return StepResult(
                observation="Episode is already done.",
                done=True,
                parse_ok=False,
                action_text=None,
                score=self.last_score,
            )

        action_text = (
            extract_action_text(model_text_or_action) or model_text_or_action.strip()
        )
        try:
            action = parse_action_line(action_text)
        except Exception as exc:
            self._charge(self.config.invalid_action_cost)
            self.invalid_actions += 1
            return StepResult(
                observation=(
                    f"Invalid action syntax: {exc}. Use exactly one ACTION line such as "
                    "ACTION: INTERVENE x00 or ACTION: TEST edge(x00,x01)."
                ),
                done=False,
                parse_ok=False,
                action_text=action_text,
                debug={"error": str(exc)},
            )

        if isinstance(action, TestAction):
            return self._test(action.expr, action_text)
        if isinstance(action, InterveneAction):
            return self._intervene(action.variable, action_text)
        if isinstance(action, CommitAction):
            return self._commit(action.exprs, action.mode, action_text)
        raise TypeError(f"Unsupported action: {action!r}")

    def force_empty_commit(self) -> CommitScore:
        score = self._score_commit(())
        self.done = True
        self.last_score = score
        return score

    def diagnostics(self) -> dict[str, Any]:
        return {
            "target_keys": sorted(self.world.terminal_keys),
            "branches": [
                {"branch_id": branch.branch_id, "path": branch.path}
                for branch in self.world.branches
            ],
            "known_variables": sorted(self.known_variables),
            "budget_used": self.initial_budget - self.budget,
            "invalid_actions": self.invalid_actions,
        }

    def _test(self, expr: Expr, action_text: str) -> StepResult:
        if not self._has_budget(self.config.test_cost):
            return self._no_budget(action_text)
        unknown = variables_in_expr(expr) - self.known_variables
        if unknown:
            self._charge(self.config.invalid_action_cost)
            return StepResult(
                observation=(
                    f"Action not admissible: unknown variables {', '.join(sorted(unknown))}. "
                    "Use INTERVENE on known variables to expose new variables."
                ),
                done=False,
                parse_ok=True,
                action_text=action_text,
                debug={"unknown_variables": sorted(unknown)},
            )

        self._charge(self.config.test_cost)
        info = self.world.classify(expr)
        key = canonical_key(expr)
        signal = self.evidence_model.sample(self.rng, info.true)
        summary = self.evidence.update(key, signal)
        self._sync_claim_status(key, summary)
        return StepResult(
            observation=self._format_test_observation(expr, signal, summary),
            done=False,
            parse_ok=True,
            action_text=action_text,
            debug={"hidden": asdict(info), "signal": signal},
        )

    def _intervene(self, variable: str, action_text: str) -> StepResult:
        if not self._has_budget(self.config.intervene_cost):
            return self._no_budget(action_text)
        if variable not in self.known_variables:
            self._charge(self.config.invalid_action_cost)
            return StepResult(
                observation=(
                    f"Action not admissible: {variable} is not known. "
                    "Intervene only on known variables."
                ),
                done=False,
                parse_ok=True,
                action_text=action_text,
                debug={"unknown_variable": variable},
            )

        self._charge(self.config.intervene_cost)
        candidates = self.world.effect_candidates(variable)
        if not candidates:
            return StepResult(
                observation=f"INTERVENE {variable}: no measurable downstream changes detected.",
                done=False,
                parse_ok=True,
                action_text=action_text,
                debug={"effects": []},
            )

        effect_lines = []
        debug_effects = []
        for target, is_true in candidates:
            signal = self.evidence_model.sample(self.rng, is_true)
            edge = Edge(variable, target)
            key = canonical_key(edge)
            summary = self.evidence.update(key, signal)
            self._sync_claim_status(key, summary)
            self.known_variables.add(target)
            effect_lines.append(f"- {target}: measured_effect={signal:.2f}")
            debug_effects.append(
                {"target": target, "true_edge": is_true, "signal": signal}
            )

        return StepResult(
            observation=(
                f"INTERVENE {variable}: observed downstream measurements:\n"
                + "\n".join(effect_lines)
                + f"\nNewly measurable variables: {', '.join(sorted(target for target, _ in candidates))}"
            ),
            done=False,
            parse_ok=True,
            action_text=action_text,
            debug={"effects": debug_effects},
        )

    def _commit(
        self, exprs: tuple[Expr, ...], mode: str, action_text: str
    ) -> StepResult:
        del mode
        if self.protocol == "single" and len(exprs) != 1:
            return StepResult(
                observation="This episode uses single-answer protocol. Use COMMIT with exactly one hypothesis.",
                done=False,
                parse_ok=True,
                action_text=action_text,
            )
        if self.max_commit is not None and len(exprs) > self.max_commit:
            return StepResult(
                observation=(
                    f"Too many committed hypotheses: got {len(exprs)}, max is {self.max_commit}. "
                    "Submit a smaller final set."
                ),
                done=False,
                parse_ok=True,
                action_text=action_text,
            )

        score = self._score_commit(exprs)
        self.done = True
        self.last_score = score
        valid = ", ".join(score.valid_keys) if score.valid_keys else "none"
        return StepResult(
            observation=(
                f"Episode complete. Valid final hypotheses: {valid}. "
                f"Reward={score.reward:.3f}; valid_unique={score.valid_unique_count}; false={score.false_count}; "
                f"non_final={score.non_final_count}; unsupported={score.unsupported_count}; "
                f"duplicates={score.duplicate_count}."
            ),
            done=True,
            parse_ok=True,
            action_text=action_text,
            score=score,
            debug={"score": asdict(score)},
        )

    def _score_commit(self, exprs: tuple[Expr, ...]) -> CommitScore:
        valid_keys: set[str] = set()
        valid_branch_ids: set[int] = set()
        seen: set[str] = set()
        false_count = 0
        non_final_count = 0
        unsupported_count = 0
        duplicate_count = 0
        valid_committed_count = 0

        for expr in exprs:
            key = canonical_key(expr)
            duplicate = key in seen
            if duplicate:
                duplicate_count += 1
            else:
                seen.add(key)
            info = self.world.classify(expr)
            if not info.true:
                false_count += 1
                continue
            if not info.terminal:
                non_final_count += 1
                continue
            if not self._evidence_backed(expr, info):
                unsupported_count += 1
                continue
            valid_committed_count += 1
            if duplicate:
                continue
            valid_keys.add(key)
            branch_id = self.world.terminal_key_to_branch.get(key)
            if branch_id is not None:
                valid_branch_ids.add(branch_id)

        budget_used = self.initial_budget - self.budget
        reward = (
            self.config.valid_hypothesis_reward * len(valid_keys)
            - self.config.false_penalty * false_count
            - self.config.non_final_penalty * non_final_count
            - self.config.unsupported_penalty * unsupported_count
            - self.config.budget_penalty * budget_used
        )
        return CommitScore(
            reward=reward,
            valid_keys=tuple(sorted(valid_keys)),
            valid_branch_ids=tuple(sorted(valid_branch_ids)),
            false_count=false_count,
            non_final_count=non_final_count,
            unsupported_count=unsupported_count,
            duplicate_count=duplicate_count,
            committed_count=len(exprs),
            valid_committed_count=valid_committed_count,
            valid_unique_count=len(valid_keys),
        )

    def _evidence_backed(self, expr: Expr, info: HypothesisInfo) -> bool:
        key = canonical_key(expr)
        if self.evidence.is_accepted(key):
            return True
        if isinstance(expr, PathExpr):
            return all(
                self.evidence.is_accepted(edge_key)
                for edge_key in edge_keys_for_path(expr)
            )
        return info.terminal and self.evidence.is_accepted(key)

    def _format_test_observation(
        self, expr: Expr, signal: float, summary: EvidenceSummary
    ) -> str:
        return (
            f"TEST {format_expr(expr)}: measurement={signal:.2f}; "
            f"samples_for_claim={summary.samples}."
        )

    def _sync_claim_status(self, key: str, summary: EvidenceSummary) -> None:
        if summary.status == "accepted":
            self.accepted_claims.add(key)
            self.rejected_claims.discard(key)
        elif summary.status == "rejected":
            self.rejected_claims.add(key)
            self.accepted_claims.discard(key)

    def _format_evidence_summary(
        self, item: EvidenceSummary, *, include_evidence_status: bool
    ) -> str:
        latest = "n/a" if item.latest_signal is None else f"{item.latest_signal:.2f}"
        mean = "n/a" if item.mean_signal is None else f"{item.mean_signal:.2f}"
        text = (
            f"- {item.key}: samples={item.samples}, "
            f"mean_measurement={mean}, latest_measurement={latest}"
        )
        if include_evidence_status:
            text += f", posterior={item.posterior:.2f}, status={item.status}"
        return text

    def _format_claim_set(self, keys: set[str]) -> str:
        if not keys:
            return "none"
        return ", ".join(sorted(keys))

    def _has_budget(self, cost: int) -> bool:
        return self.budget >= cost

    def _charge(self, cost: int) -> None:
        self.budget = max(0, self.budget - cost)

    def _no_budget(self, action_text: str) -> StepResult:
        return StepResult(
            observation="No experiment budget remains. Submit a final COMMIT action.",
            done=False,
            parse_ok=True,
            action_text=action_text,
        )


class ScatteredCausalDiscoveryEnv:
    """Generic DiscoveryEnv adapter for the synthetic scattered causal engine."""

    def __init__(
        self,
        config: WorldConfig,
        *,
        world_seed: int,
        episode_seed: int,
        dispersion: float,
        protocol: str = "single",
        max_commit: int = 1,
        budget: int | None = None,
        agent_config: AgentConfig | None = None,
        format_reward: float | None = None,
        admissible_reward: float | None = None,
        commit_format_reward: float | None = None,
        invalid_action_penalty: float | None = None,
        reward_config: RewardConfig | None = None,
    ) -> None:
        rewards = reward_config or SCATTERED_CAUSAL_REWARD
        self._env = ScatteredDiscoveryEnv(
            config,
            world_seed=world_seed,
            episode_seed=episode_seed,
            dispersion=dispersion,
            budget=budget,
            protocol=protocol,
            max_commit=max_commit,
        )
        self.config = config
        self.protocol = protocol
        self.agent_config = agent_config or AgentConfig()
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
        self._parse_failures = 0

    @property
    def done(self) -> bool:
        return self._env.done

    def system_prompt(self, runtime: Literal["local", "verl"] = "local") -> str:
        del runtime
        from scattered_discovery.prompts.scattered import SYSTEM_PROMPT

        return SYSTEM_PROMPT

    def reset(self) -> str:
        from scattered_discovery.prompts.scattered import initial_user_prompt

        return initial_user_prompt(self._env, self.agent_config)

    def observation_prompt(
        self,
        step: DiscoveryStep,
        runtime: Literal["local", "verl"] = "local",
    ) -> str:
        del runtime
        from scattered_discovery.prompts.scattered import observation_prompt

        return observation_prompt(self._env, step.observation, self.agent_config)

    def step(self, model_text_or_action: str) -> DiscoveryStep:
        result = self._env.step(model_text_or_action)
        if result.parse_ok:
            self._breakdown = self._breakdown.plus(format=self._format_reward)
            if "not admissible" not in result.observation.lower():
                self._breakdown = self._breakdown.plus(
                    admissible=self._admissible_reward
                )
        else:
            self._parse_failures += 1
            self._breakdown = self._breakdown.plus(
                invalid_action=-self._invalid_action_penalty
            )

        if result.score is not None:
            score = self._convert_score(result.score)
            return DiscoveryStep(
                observation=result.observation,
                done=result.done,
                parse_ok=result.parse_ok,
                action_text=result.action_text,
                reward=score.reward,
                score=score,
                metrics=self._step_metrics(),
                debug=result.debug or {},
            )
        return DiscoveryStep(
            observation=result.observation,
            done=result.done,
            parse_ok=result.parse_ok,
            action_text=result.action_text,
            metrics=self._step_metrics(),
            debug=result.debug or {},
        )

    def force_finalize(self) -> DiscoveryScore:
        return self._convert_score(self._env.force_empty_commit())

    def diagnostics(self) -> dict[str, Any]:
        diagnostics = self._env.diagnostics()
        diagnostics["parse_failures"] = self._parse_failures
        diagnostics["reward_breakdown_so_far"] = self._breakdown.as_dict()
        return diagnostics

    def _convert_score(self, score: CommitScore) -> DiscoveryScore:
        budget_used = self._env.initial_budget - self._env.budget
        duplicate_zeroed = duplicate_set_zeroes_reward(
            self.protocol, score.duplicate_count
        )
        if duplicate_zeroed:
            terminal_breakdown = RewardBreakdown()
        else:
            terminal_breakdown = self._breakdown.plus(
                valid_hypothesis=self.config.valid_hypothesis_reward
                * score.valid_unique_count,
                commit_format=self._commit_format_reward
                if score.committed_count > 0
                else 0.0,
                false_commit=-self.config.false_penalty * score.false_count,
                non_final_commit=-self.config.non_final_penalty * score.non_final_count,
                unsupported_commit=-self.config.unsupported_penalty
                * score.unsupported_count,
                budget=-self.config.budget_penalty * budget_used,
            )
        valid_keys = score.valid_keys
        valid_branch_ids = score.valid_branch_ids
        valid_unique_count = score.valid_unique_count
        target_count = self._env.target_count
        recovery = valid_unique_count / target_count if target_count else 0.0
        return DiscoveryScore(
            reward=terminal_breakdown.total,
            breakdown=terminal_breakdown,
            valid_keys=valid_keys,
            valid_branch_ids=valid_branch_ids,
            valid_committed_count=score.valid_committed_count,
            valid_unique_count=valid_unique_count,
            committed_count=score.committed_count,
            false_count=score.false_count,
            non_final_count=score.non_final_count,
            unsupported_count=score.unsupported_count,
            duplicate_count=score.duplicate_count,
            parse_failures=self._parse_failures,
            invalid_actions=self._env.invalid_actions,
            metrics={
                "base_task_reward": score.reward,
                "target_count": target_count,
                "budget_initial": self._env.initial_budget,
                "budget_used": budget_used,
                "recovery": recovery,
                "raw_score": asdict(score),
            },
            reward_vector=tuple(
                1.0 if branch_id in valid_branch_ids else 0.0
                for branch_id in range(target_count)
            ),
        )

    def _step_metrics(self) -> dict[str, Any]:
        return {
            "budget_remaining": self._env.budget,
            "budget_used": self._env.initial_budget - self._env.budget,
            "parse_failures": self._parse_failures,
            "invalid_actions": self._env.invalid_actions,
            "reward_breakdown_so_far": self._breakdown.as_dict(),
        }
