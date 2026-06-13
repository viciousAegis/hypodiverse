from __future__ import annotations

from scattered_discovery.algos.base import VerlAlgorithm
from scattered_discovery.algos.echo import EchoGRPOAlgorithm
from scattered_discovery.algos.grpo import GRPOAlgorithm, SetRewardGRPOAlgorithm


_ALGORITHMS: dict[str, VerlAlgorithm] = {
    algorithm.name: algorithm
    for algorithm in (
        GRPOAlgorithm(),
        SetRewardGRPOAlgorithm(),
        EchoGRPOAlgorithm(),
    )
}


def get_algorithm(name: str) -> VerlAlgorithm:
    try:
        return _ALGORITHMS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_ALGORITHMS))
        raise ValueError(
            f"Unknown algorithm {name!r}. Available: {available}."
        ) from exc


def list_algorithms() -> tuple[VerlAlgorithm, ...]:
    return tuple(_ALGORITHMS[name] for name in sorted(_ALGORITHMS))
