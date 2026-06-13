from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from random import Random
import re
from typing import Any, Literal

from scattered_discovery.envs.base import DiscoveryScore, DiscoveryStep, RewardBreakdown
from scattered_discovery.prompts.hypospace import causal_initial_prompt
from scattered_discovery.prompts.generic import (
    next_action_observation_prompt,
    system_prompt_for_runtime,
)
from scattered_discovery.rewards import (
    HYPO_CAUSAL_REWARD,
    RewardConfig,
    duplicate_set_zeroes_reward,
)


@dataclass(frozen=True)
class CausalGraph:
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]

    @classmethod
    def from_edges(
        cls, nodes: tuple[str, ...], edges: list[tuple[str, str]]
    ) -> "CausalGraph":
        normalized = tuple(sorted(dict.fromkeys(edges)))
        graph = cls(nodes=tuple(nodes), edges=normalized)
        if not graph.is_dag():
            raise ValueError("graph contains a directed cycle")
        return graph

    def key(self) -> str:
        if not self.edges:
            return "graph:no_edges"
        return "graph:" + ",".join(f"{src}->{dst}" for src, dst in self.edges)

    def format(self) -> str:
        if not self.edges:
            return "Graph: No edges"
        return "Graph: " + ", ".join(f"{src}->{dst}" for src, dst in self.edges)

    def children(self) -> dict[str, set[str]]:
        mapping = {node: set() for node in self.nodes}
        for src, dst in self.edges:
            mapping[src].add(dst)
        return mapping

    def descendants(self, node: str) -> set[str]:
        children = self.children()
        seen: set[str] = set()
        stack = list(children[node])
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(children[current])
        return seen

    def intervention_effects(self, perturbed_node: str) -> dict[str, int]:
        downstream = self.descendants(perturbed_node)
        return {node: (1 if node in downstream else 0) for node in self.nodes}

    def is_dag(self) -> bool:
        children = self.children()
        temporary: set[str] = set()
        permanent: set[str] = set()

        def visit(node: str) -> bool:
            if node in permanent:
                return True
            if node in temporary:
                return False
            temporary.add(node)
            for child in children[node]:
                if not visit(child):
                    return False
            temporary.remove(node)
            permanent.add(node)
            return True

        return all(visit(node) for node in self.nodes)


def enumerate_dags(
    nodes: tuple[str, ...], max_edges: int | None = None
) -> tuple[CausalGraph, ...]:
    possible_edges = [(src, dst) for src in nodes for dst in nodes if src != dst]
    max_edge_count = (
        len(possible_edges)
        if max_edges is None
        else min(max_edges, len(possible_edges))
    )
    dags: list[CausalGraph] = []
    for edge_count in range(max_edge_count + 1):
        for combo in combinations(possible_edges, edge_count):
            graph = CausalGraph(nodes=nodes, edges=tuple(sorted(combo)))
            if graph.is_dag():
                dags.append(graph)
    return tuple(dags)


def parse_graph_payload(text: str, nodes: tuple[str, ...]) -> CausalGraph:
    value = text.strip().strip("`").strip()
    if value.upper().startswith("GRAPH:"):
        value = value.split(":", 1)[1].strip()
    if value.lower().startswith("graph(") and value.endswith(")"):
        value = value[value.find("(") + 1 : -1].strip()
    if re.search(r"\b(no\s+edges?|empty|none|null)\b", value, re.IGNORECASE):
        return CausalGraph(nodes=nodes, edges=())
    parts = [part.strip().rstrip(".") for part in value.split(",") if part.strip()]
    edges: list[tuple[str, str]] = []
    for part in parts:
        match = re.fullmatch(r"([A-Za-z0-9_]+)\s*->\s*([A-Za-z0-9_]+)", part)
        if match is None:
            raise ValueError(f"invalid edge {part!r}; expected A->B")
        src, dst = match.group(1), match.group(2)
        if src not in nodes or dst not in nodes:
            raise ValueError(f"unknown node in edge {src}->{dst}")
        if src == dst:
            raise ValueError("self-loops are not allowed")
        edges.append((src, dst))
    if not edges:
        raise ValueError("graph requires at least one edge or 'No edges'")
    return CausalGraph.from_edges(nodes, edges)


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
            raise ValueError("COMMIT set payload requires [graph(...); graph(...)]")
        parts = [part.strip() for part in value[1:-1].split(";") if part.strip()]
        if not parts:
            raise ValueError("COMMIT set payload requires at least one graph")
        return parts
    return [value]


class HypoSpaceCausalEnv:
    """Interactive causal HypoSpace variant with intervention queries."""

    def __init__(
        self,
        *,
        nodes: tuple[str, ...] = ("A", "B", "C"),
        max_edges: int | None = 2,
        target_edges: tuple[tuple[str, str], ...] | None = None,
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
        rewards = reward_config or HYPO_CAUSAL_REWARD
        self.nodes = tuple(nodes)
        self.max_edges = max_edges
        self.hypothesis_space = enumerate_dags(self.nodes, max_edges)
        if target_edges is None:
            self.target = Random(seed).choice(self.hypothesis_space)
        else:
            self.target = CausalGraph.from_edges(self.nodes, list(target_edges))
        self.query_budget = (
            query_budget if query_budget is not None else len(self.nodes)
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
        self._observations: dict[str, dict[str, int]] = {}
        self._done = False
        self._last_score: DiscoveryScore | None = None
        self._parse_failures = 0
        self._invalid_actions = 0
        self._final_keys = tuple(graph.key() for graph in self._full_version_space())

    @property
    def done(self) -> bool:
        return self._done

    def system_prompt(self, runtime: Literal["local", "verl"] = "local") -> str:
        return system_prompt_for_runtime(runtime)

    def reset(self) -> str:
        return causal_initial_prompt(
            nodes=self.nodes,
            max_edges=self.max_edges,
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
            if upper.startswith("INTERVENE "):
                return self._intervene(action_text[10:].strip(), action_text)
            if upper.startswith("COMMIT "):
                return self._commit(
                    [
                        parse_graph_payload(part, self.nodes)
                        for part in _split_commit_payload(action_text[7:])
                    ],
                    action_text=action_text,
                )
            if upper.startswith("COMMIT_SET "):
                return self._commit(
                    [
                        parse_graph_payload(part, self.nodes)
                        for part in _split_commit_payload(action_text[11:])
                    ],
                    action_text=action_text,
                )
            raise ValueError("unknown action; use INTERVENE or COMMIT")
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
            "env_type": "hypospace_causal_interactive",
            "nodes": list(self.nodes),
            "max_edges": self.max_edges,
            "target_graph": self.target.format(),
            "target_key": self.target.key(),
            "final_compatible_keys": list(self._final_keys),
            "current_version_space_size": len(self._current_version_space()),
            "final_version_space_size": len(self._final_keys),
            "queried_nodes": sorted(self._observations),
            "budget_used": self.initial_budget - self.query_budget,
            "budget_remaining": self.query_budget,
            "parse_failures": self._parse_failures,
            "invalid_actions": self._invalid_actions,
            "reward_breakdown_so_far": self._breakdown.as_dict(),
        }

    def _intervene(self, node: str, action_text: str) -> DiscoveryStep:
        if node not in self.nodes:
            self._invalid_actions += 1
            self.query_budget = max(0, self.query_budget - 1)
            self._breakdown = self._breakdown.plus(
                format=self._format_reward,
                invalid_action=-self._invalid_action_penalty,
            )
            return DiscoveryStep(
                observation=f"Action not admissible: unknown node {node!r}.",
                done=False,
                parse_ok=True,
                action_text=action_text,
                metrics=self._metrics(),
            )
        if self.query_budget <= 0:
            return DiscoveryStep(
                observation="No intervention budget remains. Submit a COMMIT action.",
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
        effects = self.target.intervention_effects(node)
        self._observations[node] = effects
        changed = [target for target, value in sorted(effects.items()) if value]
        unchanged = [target for target, value in sorted(effects.items()) if not value]
        observation = (
            f"INTERVENE {node} -> downstream_changed: "
            f"{', '.join(changed) if changed else 'none'}; "
            f"not_downstream_or_unchanged: "
            f"{', '.join(unchanged) if unchanged else 'none'}."
        )
        if self.show_version_space_size:
            observation += (
                f" Compatible graphs remaining: {len(self._current_version_space())}."
            )
        return DiscoveryStep(
            observation=observation,
            done=False,
            parse_ok=True,
            action_text=action_text,
            metrics=self._metrics(),
        )

    def _commit(self, graphs: list[CausalGraph], *, action_text: str) -> DiscoveryStep:
        if self.protocol == "single" and len(graphs) != 1:
            return DiscoveryStep(
                "This episode uses single-answer protocol. Use COMMIT with exactly one graph.",
                False,
                True,
                action_text=action_text,
                metrics=self._metrics(),
            )
        if len(graphs) > self.max_commit:
            return DiscoveryStep(
                f"Too many committed graphs: got {len(graphs)}, max is {self.max_commit}.",
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
        score = self._score_commit(tuple(graphs))
        self._done = True
        self._last_score = score
        valid = ", ".join(score.valid_keys) if score.valid_keys else "none"
        return DiscoveryStep(
            observation=(
                f"Episode complete. Valid final graphs: {valid}. "
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

    def _score_commit(self, graphs: tuple[CausalGraph, ...]) -> DiscoveryScore:
        final_keys = set(self._final_keys)
        current_keys = {graph.key() for graph in self._current_version_space()}
        valid_keys: set[str] = set()
        seen: set[str] = set()
        false_count = 0
        unsupported_count = 0
        duplicate_count = 0
        valid_committed_count = 0

        for graph in graphs:
            key = graph.key()
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
            committed_count=len(graphs),
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

    def _full_version_space(self) -> tuple[CausalGraph, ...]:
        target_effects = {
            node: self.target.intervention_effects(node) for node in self.nodes
        }
        return tuple(
            graph
            for graph in self.hypothesis_space
            if all(
                graph.intervention_effects(node) == effects
                for node, effects in target_effects.items()
            )
        )

    def _current_version_space(self) -> tuple[CausalGraph, ...]:
        return tuple(
            graph
            for graph in self.hypothesis_space
            if all(
                graph.intervention_effects(node) == effects
                for node, effects in self._observations.items()
            )
        )

    def _metrics(self) -> dict[str, Any]:
        return {
            "budget_remaining": self.query_budget,
            "budget_used": self.initial_budget - self.query_budget,
            "queried_nodes": sorted(self._observations),
            "current_version_space_size": len(self._current_version_space()),
            "parse_failures": self._parse_failures,
            "invalid_actions": self._invalid_actions,
            "reward_breakdown_so_far": self._breakdown.as_dict(),
        }
