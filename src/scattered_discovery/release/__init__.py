"""Release tooling for reproducible Scattered Discovery artifacts."""

from importlib import import_module
from typing import Any

__all__ = [
    "EXACT_MODEL_SPECS",
    "ReleaseError",
    "build_dataset_release",
    "build_model_release",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    module = import_module(".causal_micro_lab", __name__)
    return getattr(module, name)
