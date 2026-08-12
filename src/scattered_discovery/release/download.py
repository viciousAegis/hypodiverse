"""Download the frozen HypoDiverse splits into their canonical local paths."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


DEFAULT_REPO_ID = "viciousa3gis/hypodiverse"
DEFAULT_REVISION = "d16867cc49836f72ace9e3667164fa6e4ae76eda"

FILES = {
    "data/train.jsonl": (
        Path("data/causal_micro_lab/trainable/verl_train.jsonl"),
        "90b21e2532757ad7661f3f61d3e14901d20e167ad0525318a7e03378151cc2f4",
    ),
    "data/validation.jsonl": (
        Path("data/causal_micro_lab/trainable/verl_val.jsonl"),
        "5865f9d985becd6d33cdbbe62090c8baac450282716760c838ba40064d0d5e6f",
    ),
    "data/test.jsonl": (
        Path("eval_sets/causal_micro_lab/canonical_eval/verl_test.jsonl"),
        "4e6ee134a105276b91ebdedab55afe3f5af53689f7c3f666b037cfa0aae36967",
    ),
    "source/final_v3/states.jsonl": (
        Path("eval_sets/causal_micro_lab/canonical_eval/states.jsonl"),
        "21360ecc53b19e75b92e00e8586795205d0b11a14044392cbab7067a063080e3",
    ),
    "source/final_v3/manifest.json": (
        Path("eval_sets/causal_micro_lab/canonical_eval/manifest.json"),
        "3c0109b0f7f565f4b903353f29cb719c615dd525e1d7078e4a7ab8028988f7e8",
    ),
    "source/trainable/experiments.jsonl": (
        Path("data/causal_micro_lab/trainable/experiments.jsonl"),
        "aea1f302f5230d015b17c89fb4210c400716226414637ee903c7558cc6997e5a",
    ),
    "source/trainable/manifest.jsonl": (
        Path("data/causal_micro_lab/trainable/manifest.jsonl"),
        "0b477f54e12f98f6351e16455f3065e96ee576cb96e8dcf8033d813bfb3e92c0",
    ),
    "source/trainable/modes.jsonl": (
        Path("data/causal_micro_lab/trainable/modes.jsonl"),
        "0cd909dbb82ebf3135950852f8af505a9fed3b1affffc36e14eda0ff523e996a",
    ),
    "source/trainable/states_train.jsonl": (
        Path("data/causal_micro_lab/trainable/states_train.jsonl"),
        "c964dbbed9b706486e631fb6d4b81e677ad3550fd563ca2f66d38ad3107d3351",
    ),
    "source/trainable/states_val.jsonl": (
        Path("data/causal_micro_lab/trainable/states_val.jsonl"),
        "7ebc33830ae2127cc931d1733448ba7e9570ac245fc85c82542c392674e4975a",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_frozen_splits(
    *,
    repo_root: str | Path = ".",
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
    force: bool = False,
) -> dict[str, Path]:
    """Materialize exact split files, rejecting unexpected local contents."""

    root = Path(repo_root).expanduser().resolve()
    resolved: dict[str, Path] = {}
    for hub_path, (relative_destination, expected_hash) in FILES.items():
        destination = root / relative_destination
        if destination.is_file() and _sha256(destination) == expected_hash:
            resolved[hub_path] = destination
            continue
        if destination.exists() and not force:
            raise RuntimeError(
                f"Refusing to replace non-canonical file: {destination}. "
                "Pass --force to restore the frozen split."
            )

        cached = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=hub_path,
                repo_type="dataset",
                revision=revision,
            )
        )
        actual_hash = _sha256(cached)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"SHA256 mismatch for {repo_id}@{revision}:{hub_path}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        shutil.copyfile(cached, temporary)
        temporary.replace(destination)
        resolved[hub_path] = destination
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and verify the frozen HypoDiverse JSONL splits."
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    paths = download_frozen_splits(
        repo_root=args.repo_root,
        repo_id=args.repo_id,
        revision=args.revision,
        force=args.force,
    )
    for hub_path, destination in paths.items():
        print(f"{hub_path} -> {destination}")


if __name__ == "__main__":
    main()
