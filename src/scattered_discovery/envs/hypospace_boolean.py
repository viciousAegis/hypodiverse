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
from scattered_discovery.prompts.hypospace import boolean_initial_prompt
from scattered_discovery.rewards import (
    HYPO_BOOLEAN_REWARD,
    RewardConfig,
    duplicate_set_zeroes_reward,
)


class BooleanParseError(ValueError):
    pass


@dataclass(frozen=True)
class BoolExpr:
    op: str
    args: tuple["BoolExpr", ...] = ()
    name: str | None = None

    @staticmethod
    def var(name: str) -> "BoolExpr":
        return BoolExpr("VAR", name=name)

    def evaluate(self, assignment: dict[str, int]) -> int:
        if self.op == "VAR":
            if self.name is None:
                raise ValueError("variable has no name")
            return int(assignment[self.name])
        if self.op == "NOT":
            return 1 - self.args[0].evaluate(assignment)
        if self.op == "AND":
            return int(all(arg.evaluate(assignment) for arg in self.args))
        if self.op == "OR":
            return int(any(arg.evaluate(assignment) for arg in self.args))
        if self.op == "XOR":
            value = 0
            for arg in self.args:
                value ^= arg.evaluate(assignment)
            return value
        if self.op == "NOR":
            return 1 - int(any(arg.evaluate(assignment) for arg in self.args))
        raise ValueError(f"unsupported op {self.op!r}")

    def depth(self) -> int:
        if self.op == "VAR":
            return 0
        return 1 + max(arg.depth() for arg in self.args)

    def operators(self) -> set[str]:
        if self.op == "VAR":
            return set()
        ops = {self.op}
        for arg in self.args:
            ops.update(arg.operators())
        return ops

    def variables(self) -> set[str]:
        if self.op == "VAR":
            return {self.name or ""}
        values: set[str] = set()
        for arg in self.args:
            values.update(arg.variables())
        return values

    def mechanistic_key(self) -> tuple[Any, ...]:
        if self.op == "VAR":
            return ("VAR", self.name)
        if self.op == "NOT":
            return ("NOT", self.args[0].mechanistic_key())
        child_keys: list[tuple[Any, ...]] = []
        for arg in self.args:
            key = arg.mechanistic_key()
            if self.op in {"AND", "OR", "XOR"} and key and key[0] == self.op:
                child_keys.extend(key[1])
            else:
                child_keys.append(key)
        if self.op in {"AND", "OR", "XOR", "NOR"}:
            child_keys = sorted(child_keys)
        if self.op in {"AND", "OR"}:
            deduped: list[tuple[Any, ...]] = []
            for key in child_keys:
                if key not in deduped:
                    deduped.append(key)
            child_keys = deduped
        return (self.op, tuple(child_keys))

    def key(self) -> str:
        return repr(self.mechanistic_key())

    def format(self) -> str:
        if self.op == "VAR":
            return self.name or ""
        if self.op == "NOT":
            child = self.args[0]
            rendered = child.format()
            if child.op != "VAR":
                rendered = f"({rendered})"
            return f"NOT {rendered}"
        if self.op == "NOR":
            return "NOR(" + ", ".join(arg.format() for arg in self.args) + ")"
        separator = f" {self.op} "
        return separator.join(
            arg.format() if arg.op == "VAR" else f"({arg.format()})"
            for arg in self.args
        )


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[(),]", text)


class _BoolParser:
    def __init__(
        self, tokens: list[str], variables: tuple[str, ...], operators: set[str]
    ) -> None:
        self.tokens = tokens
        self.variables = set(variables)
        self.operators = operators
        self.pos = 0

    def parse(self) -> BoolExpr:
        expr = self._parse_or()
        if self.pos != len(self.tokens):
            raise BooleanParseError(f"unexpected token {self.tokens[self.pos]!r}")
        return expr

    def _peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _consume(self) -> str:
        token = self._peek()
        if token is None:
            raise BooleanParseError("unexpected end of expression")
        self.pos += 1
        return token

    def _parse_or(self) -> BoolExpr:
        expr = self._parse_xor()
        while (self._peek() or "").upper() == "OR":
            self._consume()
            if "OR" not in self.operators:
                raise BooleanParseError("OR is not allowed")
            expr = BoolExpr("OR", (expr, self._parse_xor()))
        return expr

    def _parse_xor(self) -> BoolExpr:
        expr = self._parse_and()
        while (self._peek() or "").upper() == "XOR":
            self._consume()
            if "XOR" not in self.operators:
                raise BooleanParseError("XOR is not allowed")
            expr = BoolExpr("XOR", (expr, self._parse_and()))
        return expr

    def _parse_and(self) -> BoolExpr:
        expr = self._parse_unary()
        while (self._peek() or "").upper() == "AND":
            self._consume()
            if "AND" not in self.operators:
                raise BooleanParseError("AND is not allowed")
            expr = BoolExpr("AND", (expr, self._parse_unary()))
        return expr

    def _parse_unary(self) -> BoolExpr:
        token = self._peek()
        if token is None:
            raise BooleanParseError("unexpected end of expression")
        upper = token.upper()
        if upper == "NOT":
            self._consume()
            if "NOT" not in self.operators:
                raise BooleanParseError("NOT is not allowed")
            return BoolExpr("NOT", (self._parse_unary(),))
        if upper == "NOR":
            self._consume()
            if "NOR" not in self.operators:
                raise BooleanParseError("NOR is not allowed")
            if self._consume() != "(":
                raise BooleanParseError("NOR requires NOR(expr, expr)")
            first = self._parse_or()
            if self._consume() != ",":
                raise BooleanParseError("NOR requires two comma-separated arguments")
            second = self._parse_or()
            if self._consume() != ")":
                raise BooleanParseError("NOR call missing closing parenthesis")
            return BoolExpr("NOR", (first, second))
        if token == "(":
            self._consume()
            expr = self._parse_or()
            if self._consume() != ")":
                raise BooleanParseError("missing closing parenthesis")
            return expr
        name = self._consume()
        if name not in self.variables:
            raise BooleanParseError(f"unknown variable {name!r}")
        return BoolExpr.var(name)


def parse_boolean_expr(
    text: str, variables: tuple[str, ...], operators: set[str]
) -> BoolExpr:
    value = text.strip().strip("`").strip()
    if value.upper().startswith("EXPRESSION:"):
        value = value.split(":", 1)[1].strip()
    if value.lower().startswith("expr(") and value.endswith(")"):
        value = value[value.find("(") + 1 : -1].strip()
    tokens = _tokenize(value)
    if not tokens:
        raise BooleanParseError("empty expression")
    return _BoolParser(tokens, variables, operators).parse()


def truth_table(expr: BoolExpr, variables: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(
        expr.evaluate(dict(zip(variables, values, strict=True)))
        for values in product([0, 1], repeat=len(variables))
    )


def enumerate_expressions(
    variables: tuple[str, ...],
    operators: set[str],
    max_depth: int,
) -> tuple[BoolExpr, ...]:
    by_depth: dict[int, list[BoolExpr]] = {0: [BoolExpr.var(var) for var in variables]}
    seen: set[tuple[Any, ...]] = {expr.mechanistic_key() for expr in by_depth[0]}
    all_exprs = list(by_depth[0])
    for depth in range(1, max_depth + 1):
        candidates: list[BoolExpr] = []
        previous = [expr for exprs in by_depth.values() for expr in exprs]
        if "NOT" in operators:
            candidates.extend(BoolExpr("NOT", (expr,)) for expr in previous)
        for op in ("AND", "OR", "XOR", "NOR"):
            if op in operators:
                candidates.extend(
                    BoolExpr(op, (left, right))
                    for left in previous
                    for right in previous
                )
        unique_for_depth: list[BoolExpr] = []
        for expr in candidates:
            if expr.depth() != depth:
                continue
            key = expr.mechanistic_key()
            if key in seen:
                continue
            seen.add(key)
            unique_for_depth.append(expr)
            all_exprs.append(expr)
        by_depth[depth] = unique_for_depth
    return tuple(all_exprs)


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
            raise ValueError("COMMIT set payload requires [expr(...); expr(...)]")
        parts = [part.strip() for part in value[1:-1].split(";") if part.strip()]
        if not parts:
            raise ValueError("COMMIT set payload requires at least one expression")
        return parts
    return [value]


def _parse_assignment(text: str, variables: tuple[str, ...]) -> dict[str, int]:
    assignment: dict[str, int] = {}
    for part in text.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"invalid assignment {part!r}; expected x=0")
        name, raw_value = [item.strip() for item in part.split("=", 1)]
        if name not in variables:
            raise ValueError(f"unknown variable {name!r}")
        if raw_value not in {"0", "1"}:
            raise ValueError(f"assignment for {name} must be 0 or 1")
        assignment[name] = int(raw_value)
    missing = set(variables) - set(assignment)
    if missing:
        raise ValueError(f"missing assignments for {', '.join(sorted(missing))}")
    return assignment


class HypoSpaceBooleanEnv:
    """Interactive Boolean HypoSpace variant with active input-output queries."""

    def __init__(
        self,
        *,
        variables: tuple[str, ...] = ("x", "y"),
        operators: tuple[str, ...] = ("AND", "OR", "NOT", "XOR", "NOR"),
        max_depth: int = 2,
        target_expression: str | None = None,
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
        rewards = reward_config or HYPO_BOOLEAN_REWARD
        self.variables = tuple(variables)
        self.operators = {op.upper() for op in operators}
        self.max_depth = max_depth
        self.hypothesis_space = enumerate_expressions(
            self.variables, self.operators, max_depth
        )
        if target_expression is None:
            self.target = Random(seed).choice(self.hypothesis_space)
        else:
            self.target = parse_boolean_expr(
                target_expression, self.variables, self.operators
            )
            self._ensure_in_space(self.target)
        self.query_budget = (
            query_budget
            if query_budget is not None
            else max(1, 2 ** len(self.variables) - 1)
        )
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
        self._observations: dict[tuple[int, ...], int] = {}
        self._done = False
        self._last_score: DiscoveryScore | None = None
        self._parse_failures = 0
        self._invalid_actions = 0
        self._target_truth_table = truth_table(self.target, self.variables)
        self._final_keys = tuple(expr.key() for expr in self._full_version_space())

    @property
    def done(self) -> bool:
        return self._done

    def system_prompt(self, runtime: Literal["local", "verl"] = "local") -> str:
        return system_prompt_for_runtime(runtime)

    def reset(self) -> str:
        return boolean_initial_prompt(
            variables=self.variables,
            operators=self.operators,
            max_depth=self.max_depth,
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
            if upper.startswith("QUERY "):
                return self._query(
                    _parse_assignment(action_text[6:].strip(), self.variables),
                    action_text,
                )
            if upper.startswith("COMMIT "):
                return self._commit(
                    [
                        self._ensure_in_space(
                            parse_boolean_expr(part, self.variables, self.operators)
                        )
                        for part in _split_commit_payload(action_text[7:])
                    ],
                    action_text=action_text,
                )
            if upper.startswith("COMMIT_SET "):
                return self._commit(
                    [
                        self._ensure_in_space(
                            parse_boolean_expr(part, self.variables, self.operators)
                        )
                        for part in _split_commit_payload(action_text[11:])
                    ],
                    action_text=action_text,
                )
            raise ValueError("unknown action; use QUERY or COMMIT")
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
            "env_type": "hypospace_boolean_interactive",
            "variables": list(self.variables),
            "operators": sorted(self.operators),
            "max_depth": self.max_depth,
            "target_expression": self.target.format(),
            "target_key": self.target.key(),
            "final_compatible_keys": list(self._final_keys),
            "current_version_space_size": len(self._current_version_space()),
            "final_version_space_size": len(self._final_keys),
            "observed_assignments": {
                ",".join(map(str, key)): value
                for key, value in sorted(self._observations.items())
            },
            "budget_used": self.initial_budget - self.query_budget,
            "budget_remaining": self.query_budget,
            "parse_failures": self._parse_failures,
            "invalid_actions": self._invalid_actions,
            "reward_breakdown_so_far": self._breakdown.as_dict(),
        }

    def _query(self, assignment: dict[str, int], action_text: str) -> DiscoveryStep:
        if self.query_budget <= 0:
            return DiscoveryStep(
                observation="No query budget remains. Submit a COMMIT action.",
                done=False,
                parse_ok=True,
                action_text=action_text,
                metrics=self._metrics(),
            )
        self.query_budget -= 1
        self._breakdown = self._breakdown.plus(
            format=self._format_reward,
            admissible=self._admissible_reward,
        )
        key = tuple(assignment[var] for var in self.variables)
        output = self.target.evaluate(assignment)
        self._observations[key] = output
        assignment_text = ", ".join(
            f"{var}={assignment[var]}" for var in self.variables
        )
        observation = f"QUERY {assignment_text} -> {output}."
        if self.show_version_space_size:
            observation += (
                " Compatible expressions remaining: "
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
        self, expressions: list[BoolExpr], *, action_text: str
    ) -> DiscoveryStep:
        if self.protocol == "single" and len(expressions) != 1:
            return DiscoveryStep(
                "This episode uses single-answer protocol. Use COMMIT with exactly one expression.",
                False,
                True,
                action_text=action_text,
                metrics=self._metrics(),
            )
        if len(expressions) > self.max_commit:
            return DiscoveryStep(
                f"Too many committed expressions: got {len(expressions)}, max is {self.max_commit}.",
                False,
                True,
                action_text=action_text,
                metrics=self._metrics(),
            )
        self._breakdown = self._breakdown.plus(
            format=self._format_reward,
            admissible=self._admissible_reward,
            commit_format=self._commit_format_reward,
        )
        score = self._score_commit(tuple(expressions))
        self._done = True
        self._last_score = score
        valid = ", ".join(score.valid_keys) if score.valid_keys else "none"
        return DiscoveryStep(
            observation=(
                f"Episode complete. Valid final expressions: {valid}. "
                f"Reward={score.reward:.3f}; valid_unique={score.valid_unique_count}; "
                f"false={score.false_count}; unsupported={score.unsupported_count}; "
                f"duplicates={score.duplicate_count}."
            ),
            done=True,
            parse_ok=True,
            action_text=action_text,
            reward=score.reward,
            score=score,
            metrics=self._metrics(),
        )

    def _score_commit(self, expressions: tuple[BoolExpr, ...]) -> DiscoveryScore:
        final_keys = set(self._final_keys)
        current_keys = {expr.key() for expr in self._current_version_space()}
        valid_keys: set[str] = set()
        seen: set[str] = set()
        false_count = 0
        unsupported_count = 0
        duplicate_count = 0
        valid_committed_count = 0

        for expr in expressions:
            key = expr.key()
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
        reward_vector = tuple(
            1.0 if key in valid_keys else 0.0 for key in final_key_list
        )
        return DiscoveryScore(
            reward=breakdown.total,
            breakdown=breakdown,
            valid_keys=tuple(sorted(valid_keys)),
            valid_committed_count=valid_committed_count,
            valid_unique_count=len(valid_keys),
            committed_count=len(expressions),
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
            reward_vector=reward_vector,
        )

    def _ensure_in_space(self, expr: BoolExpr) -> BoolExpr:
        unknown = expr.variables() - set(self.variables)
        if unknown:
            raise BooleanParseError(f"unknown variables: {', '.join(sorted(unknown))}")
        disallowed = expr.operators() - self.operators
        if disallowed:
            raise BooleanParseError(
                f"disallowed operators: {', '.join(sorted(disallowed))}"
            )
        if expr.depth() > self.max_depth:
            raise BooleanParseError(
                f"expression depth {expr.depth()} exceeds max {self.max_depth}"
            )
        if expr.key() not in {candidate.key() for candidate in self.hypothesis_space}:
            raise BooleanParseError(
                "expression is outside the enumerated hypothesis space"
            )
        return expr

    def _full_version_space(self) -> tuple[BoolExpr, ...]:
        return tuple(
            expr
            for expr in self.hypothesis_space
            if truth_table(expr, self.variables) == self._target_truth_table
        )

    def _current_version_space(self) -> tuple[BoolExpr, ...]:
        return tuple(
            expr
            for expr in self.hypothesis_space
            if all(
                expr.evaluate(dict(zip(self.variables, assignment, strict=True)))
                == output
                for assignment, output in self._observations.items()
            )
        )

    def _metrics(self) -> dict[str, Any]:
        return {
            "budget_remaining": self.query_budget,
            "budget_used": self.initial_budget - self.query_budget,
            "observed_assignments": len(self._observations),
            "current_version_space_size": len(self._current_version_space()),
            "parse_failures": self._parse_failures,
            "invalid_actions": self._invalid_actions,
            "reward_breakdown_so_far": self._breakdown.as_dict(),
        }
