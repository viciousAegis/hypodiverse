from __future__ import annotations

from dataclasses import dataclass
from random import Random

from scattered_discovery.envs.scattered_dsl import (
    Collider,
    Edge,
    Expr,
    Fork,
    PathExpr,
    canonical_key,
    variables_in_expr,
)


@dataclass(frozen=True)
class Branch:
    branch_id: int
    path: tuple[str, ...]

    @property
    def terminal_key(self) -> str:
        return canonical_key(PathExpr(self.path))


@dataclass(frozen=True)
class HypothesisInfo:
    true: bool
    terminal: bool
    variables: frozenset[str]
    branch_ids: frozenset[int]
    role: str


@dataclass(frozen=True)
class GeneratedWorld:
    branches: tuple[Branch, ...]
    initial_variables: frozenset[str]
    true_edges: frozenset[tuple[str, str]]
    outgoing_candidates: dict[str, tuple[str, ...]]
    terminal_keys: frozenset[str]
    terminal_key_to_branch: dict[str, int]
    dispersion: float

    def effect_candidates(self, variable: str) -> tuple[tuple[str, bool], ...]:
        targets = self.outgoing_candidates.get(variable, ())
        return tuple(
            (target, (variable, target) in self.true_edges) for target in targets
        )

    def classify(self, expr: Expr) -> HypothesisInfo:
        if isinstance(expr, Edge):
            is_true = (expr.src, expr.dst) in self.true_edges
            return HypothesisInfo(
                true=is_true,
                terminal=False,
                variables=frozenset({expr.src, expr.dst}),
                branch_ids=self._edge_branch_ids(expr.src, expr.dst),
                role="true_edge" if is_true else "distractor",
            )

        if isinstance(expr, PathExpr):
            key = canonical_key(expr)
            branch_ids = self._path_branch_ids(expr.nodes)
            is_true = bool(branch_ids)
            terminal = key in self.terminal_keys
            role = (
                "terminal" if terminal else "intermediate" if is_true else "distractor"
            )
            return HypothesisInfo(
                true=is_true,
                terminal=terminal,
                variables=frozenset(expr.nodes),
                branch_ids=frozenset(branch_ids),
                role=role,
            )

        if isinstance(expr, Fork):
            edges_true = all(
                (expr.cause, effect) in self.true_edges for effect in expr.effects
            )
            return HypothesisInfo(
                true=edges_true,
                terminal=False,
                variables=frozenset(variables_in_expr(expr)),
                branch_ids=self._multi_edge_branch_ids(
                    [(expr.cause, effect) for effect in expr.effects]
                ),
                role="motif" if edges_true else "distractor",
            )

        if isinstance(expr, Collider):
            edges = [(cause, expr.effect) for cause in expr.causes]
            edges_true = all(edge in self.true_edges for edge in edges)
            return HypothesisInfo(
                true=edges_true,
                terminal=False,
                variables=frozenset(variables_in_expr(expr)),
                branch_ids=self._multi_edge_branch_ids(edges),
                role="motif" if edges_true else "distractor",
            )

        raise TypeError(f"Unsupported expression: {expr!r}")

    def _edge_branch_ids(self, src: str, dst: str) -> frozenset[int]:
        ids = {
            branch.branch_id
            for branch in self.branches
            if any(
                a == src and b == dst
                for a, b in zip(branch.path[:-1], branch.path[1:], strict=True)
            )
        }
        return frozenset(ids)

    def _multi_edge_branch_ids(self, edges: list[tuple[str, str]]) -> frozenset[int]:
        ids = set[int]()
        for src, dst in edges:
            ids.update(self._edge_branch_ids(src, dst))
        return frozenset(ids)

    def _path_branch_ids(self, nodes: tuple[str, ...]) -> frozenset[int]:
        ids = set[int]()
        for branch in self.branches:
            path = branch.path
            for start in range(0, len(path) - len(nodes) + 1):
                if tuple(path[start : start + len(nodes)]) == nodes:
                    ids.add(branch.branch_id)
        return frozenset(ids)


class WorldGenerator:
    def __init__(
        self, num_branches: int, branch_depth: int, distractors_per_node: int
    ) -> None:
        if num_branches < 1:
            raise ValueError("num_branches must be positive.")
        if branch_depth < 1:
            raise ValueError("branch_depth must be positive.")
        self.num_branches = num_branches
        self.branch_depth = branch_depth
        self.distractors_per_node = distractors_per_node

    def generate(self, seed: int, dispersion: float) -> GeneratedWorld:
        rng = Random(seed)
        dispersion = min(1.0, max(0.0, dispersion))
        name_counter = 0

        def fresh() -> str:
            nonlocal name_counter
            value = f"x{name_counter:02d}"
            name_counter += 1
            return value

        shared_nodes_count = self._shared_nodes_count(dispersion)
        shared_prefix = tuple(fresh() for _ in range(shared_nodes_count))

        branches: list[Branch] = []
        for branch_id in range(self.num_branches):
            remaining = self.branch_depth + 1 - len(shared_prefix)
            branch_nodes = list(shared_prefix)
            branch_nodes.extend(fresh() for _ in range(remaining))
            branches.append(Branch(branch_id=branch_id, path=tuple(branch_nodes)))

        true_edges: set[tuple[str, str]] = set()
        parent_nodes: set[str] = set()
        for branch in branches:
            for src, dst in zip(branch.path[:-1], branch.path[1:], strict=True):
                true_edges.add((src, dst))
                parent_nodes.add(src)

        outgoing: dict[str, set[str]] = {parent: set() for parent in parent_nodes}
        for src, dst in true_edges:
            outgoing.setdefault(src, set()).add(dst)

        for parent in sorted(parent_nodes):
            for _ in range(self.distractors_per_node):
                outgoing[parent].add(fresh())

        outgoing_candidates = {
            parent: tuple(rng.sample(sorted(targets), len(targets)))
            for parent, targets in outgoing.items()
        }

        initial_variables = frozenset({branch.path[0] for branch in branches})
        terminal_keys = frozenset(branch.terminal_key for branch in branches)
        terminal_key_to_branch = {
            branch.terminal_key: branch.branch_id for branch in branches
        }

        return GeneratedWorld(
            branches=tuple(branches),
            initial_variables=initial_variables,
            true_edges=frozenset(true_edges),
            outgoing_candidates=outgoing_candidates,
            terminal_keys=terminal_keys,
            terminal_key_to_branch=terminal_key_to_branch,
            dispersion=dispersion,
        )

    def _shared_nodes_count(self, dispersion: float) -> int:
        if dispersion >= 0.95:
            return 0
        max_shared = self.branch_depth
        return int(round((1.0 - dispersion) * max_shared))
