from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Iterable, Literal

from scattered_discovery.envs.causal_micro_lab.consequence_reward import (
    CandidateStatus,
)

Variant = Literal["logdet", "count"]


@dataclass(frozen=True)
class DiversityCandidate:
    state_id: str
    status: CandidateStatus
    consequence_signature: str | None
    behavior_key: str | None

    @property
    def valid(self) -> bool:
        return (
            self.status is CandidateStatus.VALID
            and self.consequence_signature is not None
            and self.behavior_key is not None
        )


@dataclass
class BehaviorArchive:
    counts: dict[tuple[str, str], float] = field(default_factory=dict)
    last_epoch: int = 0
    validity_history: list[float] = field(default_factory=list)
    max_running_validity: float = 0.0
    beta_multiplier: float = 1.0

    def count(self, state_id: str, behavior_key: str) -> float:
        return self.counts.get((state_id, behavior_key), 0.0)

    def add(self, state_id: str, behavior_key: str, amount: float = 1.0) -> None:
        key = (state_id, behavior_key)
        self.counts[key] = self.counts.get(key, 0.0) + amount

    def decay(self, gamma: float) -> None:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        self.counts = {
            key: value * gamma
            for key, value in self.counts.items()
            if value * gamma > 1e-12
        }

    def update_validity(self, value: float, *, window: int) -> float:
        self.validity_history.append(float(value))
        if len(self.validity_history) > window:
            self.validity_history = self.validity_history[-window:]
        running = sum(self.validity_history) / len(self.validity_history)
        self.max_running_validity = max(self.max_running_validity, running)
        return running

    def to_dict(self) -> dict[str, object]:
        return {
            "counts": [
                [state_id, behavior_key, value]
                for (state_id, behavior_key), value in sorted(self.counts.items())
            ],
            "last_epoch": self.last_epoch,
            "validity_history": self.validity_history,
            "max_running_validity": self.max_running_validity,
            "beta_multiplier": self.beta_multiplier,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "BehaviorArchive":
        raw_counts = data.get("counts") or []
        counts = {
            (str(state_id), str(behavior_key)): float(value)
            for state_id, behavior_key, value in raw_counts  # type: ignore[misc]
        }
        return cls(
            counts=counts,
            last_epoch=int(data.get("last_epoch", 0)),
            validity_history=[
                float(item) for item in data.get("validity_history", [])  # type: ignore[arg-type]
            ],
            max_running_validity=float(data.get("max_running_validity", 0.0)),
            beta_multiplier=float(data.get("beta_multiplier", 1.0)),
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(target)

    @classmethod
    def load(cls, path: str | Path) -> "BehaviorArchive":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class DiversityDiagnostics:
    valid_completions: int
    unique_behaviors: int
    duplicate_valid_completions: int
    skipped: bool
    mean_raw_reward: float
    mean_archive_scale: float


def hamming_disagreement(left: str, right: str) -> float:
    if len(left) != len(right):
        raise ValueError("consequence signatures must have equal length")
    if not left:
        return 0.0
    return sum(a != b for a, b in zip(left, right, strict=True)) / len(left)


def _cholesky_logdet(matrix: list[list[float]]) -> float:
    size = len(matrix)
    if size == 0:
        return 0.0
    lower = [[0.0] * size for _ in range(size)]
    for row in range(size):
        for column in range(row + 1):
            value = matrix[row][column]
            for inner in range(column):
                value -= lower[row][inner] * lower[column][inner]
            if row == column:
                if value <= 0.0:
                    raise ValueError("kernel is not positive definite")
                lower[row][column] = math.sqrt(value)
            else:
                lower[row][column] = value / lower[column][column]
    return 2.0 * sum(math.log(lower[index][index]) for index in range(size))


def _kernel(
    signatures: list[str],
    *,
    ell: float,
    jitter: float,
) -> list[list[float]]:
    if ell <= 0.0:
        raise ValueError("ell must be positive")
    matrix: list[list[float]] = []
    for row, left in enumerate(signatures):
        values = []
        for column, right in enumerate(signatures):
            value = math.exp(-hamming_disagreement(left, right) / ell)
            if row == column:
                value += jitter
            values.append(value)
        matrix.append(values)
    return matrix


def _without(matrix: list[list[float]], omitted: int) -> list[list[float]]:
    return [
        [value for column, value in enumerate(row) if column != omitted]
        for index, row in enumerate(matrix)
        if index != omitted
    ]


def diversity_rewards(
    candidates: Iterable[DiversityCandidate],
    *,
    variant: Variant = "logdet",
    ell: float = 0.25,
    jitter: float = 1e-6,
    archive: BehaviorArchive | None = None,
    update_archive: bool = True,
) -> tuple[list[float], DiversityDiagnostics]:
    items = list(candidates)
    rewards = [0.0] * len(items)
    valid_indices = [index for index, item in enumerate(items) if item.valid]
    if len(valid_indices) < 2:
        if archive is not None and update_archive:
            for index in valid_indices:
                item = items[index]
                archive.add(item.state_id, item.behavior_key or "")
        return rewards, DiversityDiagnostics(
            valid_completions=len(valid_indices),
            unique_behaviors=len(valid_indices),
            duplicate_valid_completions=0,
            skipped=True,
            mean_raw_reward=0.0,
            mean_archive_scale=1.0,
        )

    signatures_by_key: dict[str, str] = {}
    counts: Counter[str] = Counter()
    state_by_key: dict[str, str] = {}
    for index in valid_indices:
        item = items[index]
        key = item.behavior_key or ""
        signature = item.consequence_signature or ""
        existing = signatures_by_key.setdefault(key, signature)
        if existing != signature:
            raise ValueError("behavior-key collision with different signatures")
        counts[key] += 1
        state_by_key[key] = item.state_id

    keys = sorted(signatures_by_key)
    raw_by_key: dict[str, float] = {}
    if variant == "logdet":
        matrix = _kernel(
            [signatures_by_key[key] for key in keys],
            ell=ell,
            jitter=jitter,
        )
        full_logdet = _cholesky_logdet(matrix)
        for index, key in enumerate(keys):
            # The log-det difference is the log conditional variance and is
            # non-positive for this unit-diagonal kernel. Exponentiating gives
            # a positive marginal contribution in (0, 1], so splitting credit
            # among duplicates remains a genuine duplicate tax.
            raw_by_key[key] = math.exp(
                full_logdet - _cholesky_logdet(_without(matrix, index))
            )
    elif variant == "count":
        raw_by_key = {
            key: counts[key] ** -0.5
            for key in keys
        }
    else:
        raise ValueError(f"unknown diversity variant {variant!r}")

    scales: dict[str, float] = {}
    for key in keys:
        previous = (
            archive.count(state_by_key[key], key)
            if archive is not None
            else 0.0
        )
        scales[key] = (1.0 + previous) ** -0.5

    for index in valid_indices:
        item = items[index]
        key = item.behavior_key or ""
        rewards[index] = raw_by_key[key] * scales[key] / counts[key]

    if archive is not None and update_archive:
        for index in valid_indices:
            item = items[index]
            archive.add(item.state_id, item.behavior_key or "")

    return rewards, DiversityDiagnostics(
        valid_completions=len(valid_indices),
        unique_behaviors=len(keys),
        duplicate_valid_completions=len(valid_indices) - len(keys),
        skipped=False,
        mean_raw_reward=sum(rewards[index] for index in valid_indices)
        / len(valid_indices),
        mean_archive_scale=sum(scales.values()) / len(scales),
    )
