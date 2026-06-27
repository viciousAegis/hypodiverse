from __future__ import annotations

import json
import re
from typing import Any

from scattered_discovery.envs.causal_micro_lab.dsl import (
    CausalMicroLabError,
    Rule,
    make_hypothesis,
)


class HypothesisParseError(ValueError):
    pass


_RULE_LINE_RE = re.compile(
    r"^\s*(Z1|Z2|Y)\s*(?:=|:=)\s*(COPY|NOT|AND|OR|XOR)\s*\(([^()]*)\)\s*$",
    re.IGNORECASE,
)


def _extract_json(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    return value


def _strip_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    return value


def parse_hypothesis_rules(text: str):
    value = _strip_fence(text)
    rules = []
    for raw_line in value.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _RULE_LINE_RE.match(line)
        if match is None:
            raise HypothesisParseError(f"invalid rule line: {raw_line!r}")
        target, operator, raw_inputs = match.groups()
        inputs = tuple(
            item.strip().upper()
            for item in raw_inputs.split(",")
            if item.strip()
        )
        try:
            rules.append(
                Rule(
                    target=target.upper(),
                    operator=operator.upper(),
                    inputs=inputs,
                )
            )
        except CausalMicroLabError as exc:
            raise HypothesisParseError(str(exc)) from exc
    try:
        return make_hypothesis(rules)
    except CausalMicroLabError as exc:
        raise HypothesisParseError(str(exc)) from exc


def parse_hypothesis(text: str, *, strict: bool = True):
    value = _strip_fence(text)
    if value.startswith("{") or value.startswith("["):
        return parse_hypothesis_json(value, strict=strict)
    return parse_hypothesis_rules(value)


def parse_hypothesis_json(text: str, *, strict: bool = True):
    try:
        payload = json.loads(_extract_json(text))
    except json.JSONDecodeError as exc:
        raise HypothesisParseError(f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise HypothesisParseError("hypothesis must be a JSON object")
    allowed_top = {"rules"}
    if strict and set(payload) != allowed_top:
        raise HypothesisParseError("top-level object must contain only 'rules'")
    rules_raw = payload.get("rules")
    if not isinstance(rules_raw, list):
        raise HypothesisParseError("'rules' must be a list")
    rules = []
    for index, item in enumerate(rules_raw):
        if not isinstance(item, dict):
            raise HypothesisParseError(f"rule {index} must be an object")
        allowed_rule = {"target", "operator", "inputs"}
        if strict and set(item) != allowed_rule:
            raise HypothesisParseError(
                f"rule {index} must contain only target/operator/inputs"
            )
        target = item.get("target")
        operator = item.get("operator")
        inputs = item.get("inputs")
        if not isinstance(target, str) or not isinstance(operator, str):
            raise HypothesisParseError("rule target and operator must be strings")
        if not isinstance(inputs, list) or not all(
            isinstance(input_name, str) for input_name in inputs
        ):
            raise HypothesisParseError("rule inputs must be a string list")
        try:
            rules.append(Rule(target=target, operator=operator, inputs=tuple(inputs)))
        except CausalMicroLabError as exc:
            raise HypothesisParseError(str(exc)) from exc
    try:
        return make_hypothesis(rules)
    except CausalMicroLabError as exc:
        raise HypothesisParseError(str(exc)) from exc


def parse_record_state(raw: dict[str, Any]):
    from scattered_discovery.envs.causal_micro_lab.state_generator import (
        EvidenceItem,
        EvidenceState,
    )

    private = raw.get("private") or {}
    metadata = raw.get("metadata") or {}
    evidence = []
    for item in raw.get("visible_experiments") or []:
        observation = item["observation"]
        evidence.append(
            EvidenceItem(
                experiment_id=int(item["experiment_id"]),
                outcome=(
                    int(observation["Z1"]),
                    int(observation["Z2"]),
                    int(observation["Y"]),
                ),
            )
        )
    return EvidenceState(
        state_id=str(raw["state_id"]),
        hidden_mode_id=str(private.get("hidden_mode_id", "")),
        evidence=tuple(evidence),
        valid_mode_ids=tuple(str(item) for item in private.get("valid_mode_ids", ())),
        mean_separation=float(metadata.get("mean_separation", 0.0)),
        minimum_separation=float(metadata.get("minimum_separation", 0.0)),
        maximum_separation=float(metadata.get("maximum_separation", 0.0)),
        separation_bucket=str(metadata.get("separation_bucket", "unassigned")),
        family_bucket=str(metadata.get("family_bucket", "unknown")),
    )
