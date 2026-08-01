from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from scattered_discovery.envs.causal_micro_lab.dsl import (
    BINARY_OPERATORS,
    CausalMicroLabError,
    PARENTS,
    Rule,
    make_hypothesis,
)


class HypothesisParseError(ValueError):
    pass


@dataclass(frozen=True)
class HypothesisCandidate:
    index: int
    text: str


@dataclass(frozen=True)
class VerbalizedHypothesisCandidate:
    index: int
    text: str
    probability: float | None


_RULE_LINE_RE = re.compile(
    r"^\s*(Z1|Z2|Y)\s*(?:=|:=)\s*(COPY|NOT|AND|OR|XOR)\s*\(([^()]*)\)\s*$",
    re.IGNORECASE,
)
_FLAT_RULE_LINE_RE = re.compile(
    r"^\s*(Z1|Z2|Y)\s*(?::|=|:=)?\s+(COPY|NOT|AND|OR|XOR)\s+([A-Za-z0-9_\s,]+?)\s*$",
    re.IGNORECASE,
)
_INFIX_RULE_LINE_RE = re.compile(
    r"^\s*(Z1|Z2|Y)\s*(?::|=|:=)\s*"
    r"([A-Za-z][A-Za-z0-9_]*)\s+(AND|OR|XOR)\s+"
    r"([A-Za-z][A-Za-z0-9_]*)\s*$",
    re.IGNORECASE,
)
_DIRECT_RULE_LINE_RE = re.compile(
    r"^\s*(Z1|Z2|Y)\s*(?::|=|:=)\s*([A-Za-z][A-Za-z0-9_]*)\s*$",
    re.IGNORECASE,
)
_ANSWER_TAG_RE = re.compile(
    r"<answer(?P<index>\d+)>\s*(?P<body>.*?)\s*</answer(?P=index)>",
    re.IGNORECASE | re.DOTALL,
)
_PLAIN_ANSWER_HEADING_RE = re.compile(
    r"^\s*ANSWER\s+(?P<index>\d+)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_PROBABILITY_LINE_RE = re.compile(
    r"^\s*PROBABILITY\s*:\s*(?P<value>"
    r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    r")\s*$",
    re.IGNORECASE | re.MULTILINE,
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


def _canonicalize_inputs(
    target: str, operator: str, inputs: tuple[str, ...]
) -> tuple[str, ...]:
    if operator.upper() not in BINARY_OPERATORS:
        return inputs
    allowed = PARENTS[target.upper()]
    if all(input_name in allowed for input_name in inputs):
        return tuple(sorted(inputs, key=allowed.index))
    return inputs


def parse_hypothesis_rules(text: str):
    value = _strip_fence(text)
    rules = []
    for raw_line in value.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _RULE_LINE_RE.match(line)
        if match is not None:
            target, operator, raw_inputs = match.groups()
            inputs = tuple(
                item.strip().upper() for item in raw_inputs.split(",") if item.strip()
            )
        else:
            match = _FLAT_RULE_LINE_RE.match(line)
            if match is not None:
                target, operator, raw_inputs = match.groups()
                inputs = tuple(
                    item.strip().upper()
                    for item in re.split(r"[\s,]+", raw_inputs)
                    if item.strip()
                )
            else:
                infix_match = _INFIX_RULE_LINE_RE.match(line)
                if infix_match is not None:
                    target, left_input, operator, right_input = infix_match.groups()
                    inputs = (left_input.upper(), right_input.upper())
                else:
                    direct_match = _DIRECT_RULE_LINE_RE.match(line)
                    if direct_match is None:
                        raise HypothesisParseError(f"invalid rule line: {raw_line!r}")
                    target, direct_input = direct_match.groups()
                    operator = "COPY"
                    inputs = (direct_input.upper(),)
        target = target.upper()
        operator = operator.upper()
        inputs = _canonicalize_inputs(target, operator, inputs)
        if operator in BINARY_OPERATORS and len(set(inputs)) != len(inputs):
            raise HypothesisParseError(f"invalid rule line: {raw_line!r}")
        try:
            rules.append(
                Rule(
                    target=target,
                    operator=operator,
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


def parse_hypothesis_set(
    text: str, *, expected_count: int | None = None
) -> list[HypothesisCandidate]:
    value = _strip_fence(text)
    candidates = [
        HypothesisCandidate(
            index=int(match.group("index")), text=match.group("body").strip()
        )
        for match in _ANSWER_TAG_RE.finditer(value)
    ]
    candidates.sort(key=lambda item: item.index)
    if expected_count is not None:
        return [item for item in candidates if 1 <= item.index <= expected_count]
    return candidates


def parse_verbalized_hypothesis_set(
    text: str,
    *,
    expected_count: int | None = None,
) -> list[VerbalizedHypothesisCandidate]:
    value = _strip_fence(text)
    headings = list(_PLAIN_ANSWER_HEADING_RE.finditer(value))
    candidates = []
    for offset, heading in enumerate(headings):
        body_end = (
            headings[offset + 1].start() if offset + 1 < len(headings) else len(value)
        )
        body = value[heading.end() : body_end].strip()
        probability_matches = list(_PROBABILITY_LINE_RE.finditer(body))
        probability = (
            float(probability_matches[0].group("value"))
            if len(probability_matches) == 1
            else None
        )
        hypothesis_text = _PROBABILITY_LINE_RE.sub("", body).strip()
        candidates.append(
            VerbalizedHypothesisCandidate(
                index=int(heading.group("index")),
                text=hypothesis_text,
                probability=probability,
            )
        )
    candidates.sort(key=lambda item: item.index)
    if expected_count is not None:
        return [item for item in candidates if 1 <= item.index <= expected_count]
    return candidates


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
            target_text = target.upper()
            operator_text = operator.upper()
            input_tuple = tuple(input_name.upper() for input_name in inputs)
            rules.append(
                Rule(
                    target=target_text,
                    operator=operator_text,
                    inputs=_canonicalize_inputs(
                        target_text,
                        operator_text,
                        input_tuple,
                    ),
                )
            )
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
    valid_mode_ids = tuple(str(item) for item in private.get("valid_mode_ids", ()))
    observed_experiment_ids = tuple(item.experiment_id for item in evidence)
    raw_separation_definition = metadata.get("separation_definition")
    separation_definition = str(
        raw_separation_definition or "predictive_target_disagreement_v2"
    )
    if raw_separation_definition in {
        "predictive_target_disagreement_v2",
        "full_outcome_disagreement_v3",
    }:
        mean_separation = float(metadata.get("mean_separation", 0.0))
        minimum_separation = float(metadata.get("minimum_separation", 0.0))
        maximum_separation = float(metadata.get("maximum_separation", 0.0))
        separation_bucket = str(metadata.get("separation_bucket", "unassigned"))
    elif valid_mode_ids:
        from scattered_discovery.envs.causal_micro_lab.state_generator import (
            absolute_separation_bucket,
            separation_for_modes,
        )

        mean_separation, minimum_separation, maximum_separation = separation_for_modes(
            valid_mode_ids, observed_experiment_ids
        )
        separation_bucket = absolute_separation_bucket(mean_separation)
    else:
        mean_separation = minimum_separation = maximum_separation = 0.0
        separation_bucket = "unassigned"
    return EvidenceState(
        state_id=str(raw["state_id"]),
        hidden_mode_id=str(private.get("hidden_mode_id", "")),
        evidence=tuple(evidence),
        valid_mode_ids=valid_mode_ids,
        mean_separation=mean_separation,
        minimum_separation=minimum_separation,
        maximum_separation=maximum_separation,
        separation_bucket=separation_bucket,
        family_bucket=str(metadata.get("family_bucket", "unknown")),
        separation_definition=separation_definition,
        separation_targets=tuple(
            str(item) for item in metadata.get("separation_targets", ("Y",))
        ),
        representative_budget=(
            int(metadata["representative_budget"])
            if metadata.get("representative_budget") is not None
            else None
        ),
        oracle_singleton_representation_error=(
            float(metadata["oracle_singleton_representation_error"])
            if metadata.get("oracle_singleton_representation_error") is not None
            else None
        ),
        oracle_budget_representation_error=(
            float(metadata["oracle_budget_representation_error"])
            if metadata.get("oracle_budget_representation_error") is not None
            else None
        ),
        representative_coverage_opportunity=(
            float(metadata["representative_coverage_opportunity"])
            if metadata.get("representative_coverage_opportunity") is not None
            else None
        ),
    )
