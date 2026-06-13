from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


class DSLParseError(ValueError):
    pass


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str


@dataclass(frozen=True)
class PathExpr:
    nodes: tuple[str, ...]


@dataclass(frozen=True)
class Fork:
    cause: str
    effects: tuple[str, ...]


@dataclass(frozen=True)
class Collider:
    causes: tuple[str, ...]
    effect: str


Expr = Edge | PathExpr | Fork | Collider


@dataclass(frozen=True)
class TestAction:
    expr: Expr


@dataclass(frozen=True)
class InterveneAction:
    variable: str


@dataclass(frozen=True)
class CommitAction:
    exprs: tuple[Expr, ...]
    mode: Literal["single", "set"]


Action = TestAction | InterveneAction | CommitAction


def canonical_key(expr: Expr) -> str:
    if isinstance(expr, Edge):
        return f"edge:{expr.src}->{expr.dst}"
    if isinstance(expr, PathExpr):
        return "path:" + "->".join(expr.nodes)
    if isinstance(expr, Fork):
        return f"fork:{expr.cause}->[{','.join(sorted(expr.effects))}]"
    if isinstance(expr, Collider):
        return f"collider:[{','.join(sorted(expr.causes))}]->{expr.effect}"
    raise TypeError(f"Unsupported expression: {expr!r}")


def format_expr(expr: Expr) -> str:
    if isinstance(expr, Edge):
        return f"edge({expr.src},{expr.dst})"
    if isinstance(expr, PathExpr):
        return f"path({','.join(expr.nodes)})"
    if isinstance(expr, Fork):
        return f"fork({expr.cause},[{','.join(expr.effects)}])"
    if isinstance(expr, Collider):
        return f"collider([{','.join(expr.causes)}],{expr.effect})"
    raise TypeError(f"Unsupported expression: {expr!r}")


def variables_in_expr(expr: Expr) -> set[str]:
    if isinstance(expr, Edge):
        return {expr.src, expr.dst}
    if isinstance(expr, PathExpr):
        return set(expr.nodes)
    if isinstance(expr, Fork):
        return {expr.cause, *expr.effects}
    if isinstance(expr, Collider):
        return {*expr.causes, expr.effect}
    raise TypeError(f"Unsupported expression: {expr!r}")


def edge_keys_for_path(expr: PathExpr) -> list[str]:
    return [
        canonical_key(Edge(src, dst))
        for src, dst in zip(expr.nodes[:-1], expr.nodes[1:], strict=True)
    ]


def _parse_variable(token: str) -> str:
    value = token.strip()
    if not value.startswith("x") or not value[1:].isdigit():
        raise DSLParseError(f"Invalid variable {token!r}; expected xNN.")
    return value


def _inside_call(text: str, name: str) -> str:
    prefix = f"{name}("
    if not text.startswith(prefix) or not text.endswith(")"):
        raise DSLParseError(f"Expected {name}(...), got {text!r}.")
    return text[len(prefix) : -1].strip()


def _parse_var_list(text: str) -> tuple[str, ...]:
    value = text.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    items = tuple(_parse_variable(item) for item in value.split(",") if item.strip())
    if not items:
        raise DSLParseError("Expected at least one variable.")
    return items


def _split_top_level_once(text: str) -> tuple[str, str]:
    depth = 0
    for idx, char in enumerate(text):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        elif char == "," and depth == 0:
            return text[:idx], text[idx + 1 :]
    raise DSLParseError(f"Expected top-level comma in {text!r}.")


def parse_expr(text: str) -> Expr:
    value = text.strip().rstrip(".")
    if value.startswith("edge("):
        inside = _inside_call(value, "edge")
        left, right = _split_top_level_once(inside)
        return Edge(_parse_variable(left), _parse_variable(right))
    if value.startswith("path("):
        inside = _inside_call(value, "path")
        nodes = tuple(
            _parse_variable(item) for item in inside.split(",") if item.strip()
        )
        if len(nodes) < 2:
            raise DSLParseError("path(...) requires at least two variables.")
        return PathExpr(nodes)
    if value.startswith("fork("):
        inside = _inside_call(value, "fork")
        cause, effects = _split_top_level_once(inside)
        return Fork(_parse_variable(cause), _parse_var_list(effects))
    if value.startswith("collider("):
        inside = _inside_call(value, "collider")
        causes, effect = _split_top_level_once(inside)
        return Collider(_parse_var_list(causes), _parse_variable(effect))
    raise DSLParseError(f"Unknown expression {text!r}.")


def parse_action_line(text: str) -> Action:
    value = text.strip().strip("`").strip()
    if value.upper().startswith("ACTION:"):
        value = value.split(":", 1)[1].strip()
    upper = value.upper()
    if upper.startswith("TEST "):
        return TestAction(parse_expr(value[5:].strip()))
    if upper.startswith("INTERVENE "):
        return InterveneAction(_parse_variable(value[10:].strip()))
    if upper.startswith("COMMIT "):
        payload = value[7:].strip()
        if payload.startswith("[") or payload.endswith("]"):
            if not payload.startswith("[") or not payload.endswith("]"):
                raise DSLParseError("COMMIT set payload requires [expr; expr; ...].")
            parts = [part.strip() for part in payload[1:-1].split(";") if part.strip()]
            if not parts:
                raise DSLParseError(
                    "COMMIT set payload requires at least one expression."
                )
            return CommitAction(tuple(parse_expr(part) for part in parts), mode="set")
        return CommitAction((parse_expr(payload),), mode="single")
    if upper.startswith("COMMIT_SET "):
        payload = value[11:].strip()
        if not payload.startswith("[") or not payload.endswith("]"):
            raise DSLParseError("COMMIT set payload requires [expr; expr; ...].")
        parts = [part.strip() for part in payload[1:-1].split(";") if part.strip()]
        if not parts:
            raise DSLParseError("COMMIT set payload requires at least one expression.")
        return CommitAction(tuple(parse_expr(part) for part in parts), mode="set")
    raise DSLParseError(f"Unknown action {text!r}.")


def extract_action_text(model_text: str) -> str | None:
    lines = [line.strip() for line in model_text.splitlines() if line.strip()]
    candidates: list[str] = []
    for line in lines:
        cleaned = line.strip("`").strip()
        if cleaned.upper().startswith("ACTION:"):
            candidates.append(cleaned)
        elif cleaned.upper().startswith(("TEST ", "INTERVENE ", "COMMIT ")):
            candidates.append(cleaned)
    return candidates[0] if candidates else None
