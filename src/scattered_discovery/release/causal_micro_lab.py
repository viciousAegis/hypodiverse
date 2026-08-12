"""Package and publish exact Boolean Causal Micro-Lab release artifacts.

This module deliberately has no dataset-generation path. It only validates and
copies existing files, so a release cannot silently differ from the data used by
the recorded training and evaluation runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
BASE_MODEL = "Qwen/Qwen3-4B"
DEFAULT_TRAIN_FILE = Path("data/causal_micro_lab/trainable/verl_train.jsonl")
DEFAULT_VALIDATION_FILE = Path("data/causal_micro_lab/trainable/verl_val.jsonl")
DEFAULT_TEST_FILE = Path("eval_sets/causal_micro_lab/canonical_eval/verl_test.jsonl")
DEFAULT_DATASET_OUTPUT = Path("artifacts/hf_release/hypodiverse")
DEFAULT_MODEL_OUTPUT_ROOT = Path("artifacts/hf_release/models")

DATASET_PROVENANCE_CONFIGS = (
    Path("configs/verl/runs/causal_micro_lab_cluster_grpo.yaml"),
    Path("configs/verl/runs/causal_micro_lab_cluster_lifpo.yaml"),
    Path("configs/verl/eval/hypodiverse_base.yaml"),
    Path("configs/verl/eval/hypodiverse_grpo.yaml"),
    Path("configs/verl/eval/hypodiverse_lifpo.yaml"),
)


@dataclass(frozen=True)
class ModelSpec:
    method: str
    display_name: str
    release_directory: str
    checkpoint_step: int
    training_config: Path
    evaluation_config: Path
    evaluation_protocol: str

    @property
    def merged_directory_name(self) -> str:
        return self.release_directory


EXACT_MODEL_SPECS: Mapping[str, ModelSpec] = {
    "grpo": ModelSpec(
        method="grpo",
        display_name="GRPO",
        release_directory="hypodiverse-grpo-step-90",
        checkpoint_step=90,
        training_config=Path("configs/verl/runs/causal_micro_lab_cluster_grpo.yaml"),
        evaluation_config=Path("configs/verl/eval/hypodiverse_grpo.yaml"),
        evaluation_protocol="standard",
    ),
    "lifpo": ModelSpec(
        method="lifpo",
        display_name="LIFPO",
        release_directory="hypodiverse-lifpo-step-55",
        checkpoint_step=55,
        training_config=Path("configs/verl/runs/causal_micro_lab_cluster_lifpo.yaml"),
        evaluation_config=Path("configs/verl/eval/hypodiverse_lifpo.yaml"),
        evaluation_protocol="latent-conditioned generation (8 latent identities)",
    ),
}


class ReleaseError(RuntimeError):
    """Raised when an artifact cannot be released without losing provenance."""


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _path_from_root(repo_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise ReleaseError(f"Missing {label}: {path}")


def _json_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None
    return decoded if isinstance(decoded, dict) else None


def _extract_state_id(row: Mapping[str, Any]) -> str | None:
    direct = row.get("state_id")
    if direct is not None:
        return str(direct)

    state = _json_object(row.get("state_json"))
    if state and state.get("state_id") is not None:
        return str(state["state_id"])

    env_spec = _json_object(row.get("env_spec_json"))
    if env_spec:
        task = env_spec.get("task")
        if isinstance(task, dict):
            nested_state = task.get("state")
            if (
                isinstance(nested_state, dict)
                and nested_state.get("state_id") is not None
            ):
                return str(nested_state["state_id"])

    extra = row.get("extra_info")
    if isinstance(extra, dict) and extra.get("state_id") is not None:
        return str(extra["state_id"])
    return None


def _extract_prompt_hash(row: Mapping[str, Any]) -> str | None:
    for key in ("prompt", "raw_prompt"):
        value = row.get(key)
        if value is not None:
            return _canonical_hash(value)
    return None


def inspect_jsonl(path: Path, *, require_rows: bool = False) -> dict[str, Any]:
    """Strictly validate JSONL and collect identifiers used for split checks."""

    rows = 0
    state_ids: list[str] = []
    prompt_hashes: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ReleaseError(f"Blank JSONL row in {path} at line {line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReleaseError(
                    f"Invalid JSON in {path} at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise ReleaseError(
                    f"JSONL row must be an object in {path} at line {line_number}"
                )
            rows += 1
            state_id = _extract_state_id(row)
            if state_id is not None:
                state_ids.append(state_id)
            prompt_hash = _extract_prompt_hash(row)
            if prompt_hash is not None:
                prompt_hashes.append(prompt_hash)

    if require_rows and rows == 0:
        raise ReleaseError(f"Required split is empty: {path}")

    state_id_set = set(state_ids)
    prompt_hash_set = set(prompt_hashes)
    return {
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "state_ids_found": len(state_ids),
        "unique_state_ids": len(state_id_set),
        "duplicate_state_id_rows": len(state_ids) - len(state_id_set),
        "prompt_hashes_found": len(prompt_hashes),
        "unique_prompt_hashes": len(prompt_hash_set),
        "duplicate_prompt_rows": len(prompt_hashes) - len(prompt_hash_set),
        "_state_ids": state_id_set,
        "_prompt_hashes": prompt_hash_set,
    }


def _public_jsonl_info(info: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in info.items() if not key.startswith("_")}


def _pairwise_overlap(
    split_info: Mapping[str, Mapping[str, Any]], field: str
) -> dict[str, dict[str, Any]]:
    names = list(split_info)
    result: dict[str, dict[str, Any]] = {}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = sorted(split_info[left][field] & split_info[right][field])
            result[f"{left}__{right}"] = {
                "count": len(overlap),
                "examples": overlap[:20],
            }
    return result


def _overlap_report(
    split_info: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    state_ids = _pairwise_overlap(split_info, "_state_ids")
    prompt_hashes = _pairwise_overlap(split_info, "_prompt_hashes")
    return {
        "state_ids": state_ids,
        "prompt_sha256": prompt_hashes,
        "any_state_id_overlap": any(item["count"] for item in state_ids.values()),
        "any_prompt_overlap": any(item["count"] for item in prompt_hashes.values()),
    }


def _git_provenance(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "working_tree_dirty": None if status is None else bool(status),
    }


def _file_entry(path: Path, root: Path) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix == ".jsonl":
        entry["rows"] = inspect_jsonl(path)["rows"]
    return entry


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _prepare_output(path: Path, *, force: bool) -> Path:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not path.is_dir():
        raise ReleaseError(f"Release output exists and is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        known_release = (path / "release_manifest.json").is_file()
        if not (known_release or force):
            raise ReleaseError(
                f"Refusing to replace non-release directory {path}; pass --force"
            )
    return Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))


def _install_staging(temp_path: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    temp_path.replace(destination)


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_auxiliaries(
    source_dir: Path,
    destination: Path,
    *,
    excluded: set[Path],
) -> list[str]:
    copied: list[str] = []
    for source in sorted(path for path in source_dir.rglob("*") if path.is_file()):
        if source.resolve() in excluded:
            continue
        relative = source.relative_to(source_dir)
        target = destination / relative
        _copy_file(source, target)
        copied.append(target.as_posix())
    return copied


def _dataset_card(split_info: Mapping[str, Mapping[str, Any]]) -> str:
    counts = {name: info["rows"] for name, info in split_info.items()}
    return f"""---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
  - split: validation
    path: data/validation.jsonl
  - split: test
    path: data/test.jsonl
task_categories:
- text-generation
language:
- en
license: cc-by-4.0
pretty_name: HypoDiverse
---

# HypoDiverse

This release contains the exact JSONL rows used to train and evaluate the
HypoDiverse models. HypoDiverse is an enumerable synthetic benchmark for
measuring the validity, uniqueness, repetition, and predictive diversity of
sets of scientific hypotheses. The release tool copies the frozen files and
never regenerates examples during publishing.

## Splits

| Split | Rows | Provenance |
|---|---:|---|
| train | {counts["train"]} | Exact `verl_train.jsonl` used by GRPO and LIFPO |
| validation | {counts["validation"]} | Exact `verl_val.jsonl` used during training |
| test | {counts["test"]} | Frozen canonical evaluation set |

The files under `data/` can be loaded with `datasets.load_dataset("json",
data_files=...)`. Each row retains the veRL-compatible prompt, environment
state, and verifier metadata required by the original training or evaluation
pipeline. Source tables, state files, and manifests are preserved under
`source/` when they exist. Exact run and evaluation configurations are under
`provenance/configs/`.

`release_manifest.json` records row counts, SHA256 hashes, source paths, Git
provenance, and state-ID/prompt overlap checks. The manifest intentionally does
not hash itself; every other packaged file is hashed there.
"""


def build_dataset_release(
    *,
    repo_root: str | Path = ".",
    train_file: str | Path = DEFAULT_TRAIN_FILE,
    validation_file: str | Path = DEFAULT_VALIDATION_FILE,
    test_file: str | Path = DEFAULT_TEST_FILE,
    output_dir: str | Path = DEFAULT_DATASET_OUTPUT,
    force: bool = False,
) -> dict[str, Any]:
    """Build an upload-ready dataset folder from exact, existing JSONL files."""

    root = Path(repo_root).expanduser().resolve()
    train = _path_from_root(root, train_file)
    validation = _path_from_root(root, validation_file)
    test = _path_from_root(root, test_file)
    output = _path_from_root(root, output_dir)

    split_paths = {"train": train, "validation": validation, "test": test}
    for split, path in split_paths.items():
        _require_file(path, f"{split} JSONL")

    config_paths = [_path_from_root(root, path) for path in DATASET_PROVENANCE_CONFIGS]
    for path in config_paths:
        _require_file(path, "provenance config")

    split_info = {
        split: inspect_jsonl(path, require_rows=True)
        for split, path in split_paths.items()
    }
    overlaps = _overlap_report(split_info)
    git = _git_provenance(root)
    temp = _prepare_output(output, force=force)

    try:
        packaged_splits = {
            "train": temp / "data/train.jsonl",
            "validation": temp / "data/validation.jsonl",
            "test": temp / "data/test.jsonl",
        }
        for split, source in split_paths.items():
            _copy_file(source, packaged_splits[split])

        excluded_train = {train.resolve(), validation.resolve()}
        for name in (
            "sft_train.jsonl",
            "sft_val.jsonl",
            "sft_test.jsonl",
            "verl_test.jsonl",
            "states_test.jsonl",
        ):
            excluded_train.add((train.parent / name).resolve())
        _copy_auxiliaries(
            train.parent,
            temp / "source/trainable",
            excluded=excluded_train,
        )
        _copy_auxiliaries(
            test.parent,
            temp / "source/final_v3",
            excluded={test.resolve()},
        )

        config_entries: list[dict[str, Any]] = []
        for source in config_paths:
            relative = source.relative_to(root)
            target = temp / "provenance" / relative
            _copy_file(source, target)
            config_entries.append(
                {
                    "source_path": relative.as_posix(),
                    "packaged_path": target.relative_to(temp).as_posix(),
                    "bytes": source.stat().st_size,
                    "sha256": sha256_file(source),
                }
            )

        (temp / "README.md").write_text(_dataset_card(split_info), encoding="utf-8")

        # Validate every copied JSONL, including optional source auxiliaries.
        for jsonl in temp.rglob("*.jsonl"):
            inspect_jsonl(jsonl, require_rows=jsonl.parent.name == "data")

        payload_files = sorted(
            (
                _file_entry(path, temp)
                for path in temp.rglob("*")
                if path.is_file() and path.name != "release_manifest.json"
            ),
            key=lambda item: item["path"],
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "dataset",
            "name": "HypoDiverse",
            "created_at_utc": _utc_now(),
            "generation_policy": "copy_existing_only; no regeneration",
            "git": git,
            "source_splits": {
                split: {
                    "source_path": _display_path(path, root),
                    **_public_jsonl_info(split_info[split]),
                    "packaged_path": packaged_splits[split]
                    .relative_to(temp)
                    .as_posix(),
                }
                for split, path in split_paths.items()
            },
            "split_overlap": overlaps,
            "exact_configs": config_entries,
            "files": payload_files,
            "manifest_self_hash_omitted": True,
        }
        _write_json(temp / "release_manifest.json", manifest)
        _install_staging(temp, output)
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise

    return {
        "artifact_type": "dataset",
        "output_dir": str(output),
        "manifest_path": str(output / "release_manifest.json"),
        "rows": {split: info["rows"] for split, info in split_info.items()},
        "split_overlap": overlaps,
    }


def _iter_model_files(model_dir: Path) -> Iterable[Path]:
    for path in sorted(item for item in model_dir.rglob("*") if item.is_file()):
        relative = path.relative_to(model_dir)
        if ".cache" in relative.parts or path.name == ".DS_Store":
            continue
        yield path


def validate_merged_model(model_dir: str | Path) -> dict[str, Any]:
    model_path = Path(model_dir).expanduser().resolve()
    if not model_path.is_dir():
        raise ReleaseError(
            f"Merged Hugging Face model directory not found: {model_path}"
        )

    config_path = model_path / "config.json"
    tokenizer_config = model_path / "tokenizer_config.json"
    _require_file(config_path, "model config")
    _require_file(tokenizer_config, "tokenizer config")
    if not any(
        (model_path / name).is_file()
        for name in ("tokenizer.json", "tokenizer.model", "vocab.json")
    ):
        raise ReleaseError(f"No tokenizer vocabulary found in {model_path}")

    try:
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReleaseError(f"Invalid model config: {config_path}") from exc
    if not isinstance(model_config, dict):
        raise ReleaseError(f"Model config is not a JSON object: {config_path}")

    single_weights = model_path / "model.safetensors"
    index_path = model_path / "model.safetensors.index.json"
    if not single_weights.is_file() and not index_path.is_file():
        raise ReleaseError(f"No safetensors model weights found in {model_path}")

    weight_files: set[str] = set()
    if single_weights.is_file():
        weight_files.add(single_weights.name)
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index["weight_map"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ReleaseError(f"Invalid safetensors index: {index_path}") from exc
        if not isinstance(weight_map, dict) or not weight_map:
            raise ReleaseError(f"Empty safetensors weight map: {index_path}")
        weight_files.update(str(name) for name in weight_map.values())

    for name in sorted(weight_files):
        path = model_path / name
        _require_file(path, "safetensors shard")
        if path.stat().st_size == 0:
            raise ReleaseError(f"Empty safetensors shard: {path}")

    files = [_file_entry(path, model_path) for path in _iter_model_files(model_path)]
    return {
        "model_dir": str(model_path),
        "architecture": model_config.get("architectures"),
        "model_type": model_config.get("model_type"),
        "weight_files": sorted(weight_files),
        "files": files,
        "total_bytes": sum(item["bytes"] for item in files),
    }


def _model_card(
    spec: ModelSpec,
    *,
    dataset_repo_id: str,
    dataset_revision: str,
) -> str:
    if spec.method == "lifpo":
        method_summary = """LIFPO starts from the finite-budget objective that
different rollout identities should cover different valid hypotheses. It uses
latent rollout identities and inverse-frequency credit while retaining the same
exact validity verifier as GRPO."""
    else:
        method_summary = """This checkpoint is the validity-reward GRPO baseline.
Each completion is rewarded for producing a hypothesis that is consistent with
the visible evidence; the reward has no explicit set-diversity term."""
    return f"""---
base_model: {BASE_MODEL}
library_name: transformers
pipeline_tag: text-generation
license: apache-2.0
datasets:
- {dataset_repo_id}
tags:
- causal-reasoning
- reinforcement-learning
- grpo
- hypodiverse
---

# HypoDiverse: {spec.display_name}

This is the exact merged Hugging Face checkpoint evaluated on HypoDiverse.

## Provenance

| Field | Value |
|---|---|
| Thesis-facing method | {spec.display_name} |
| Evaluated checkpoint | `global_step_{spec.checkpoint_step}` |
| Base model | `{BASE_MODEL}` |
| Dataset | [`{dataset_repo_id}`](https://huggingface.co/datasets/{dataset_repo_id}/tree/{dataset_revision}) |
| Pinned dataset revision | `{dataset_revision}` |
| Evaluation protocol | {spec.evaluation_protocol} |

{method_summary}

Exact training and evaluation configurations, per-file model hashes, and the
pinned dataset revision are recorded in `release_manifest.json` and
`provenance/configs/`.
"""


def build_model_release(
    *,
    method: str,
    model_dir: str | Path,
    dataset_repo_id: str,
    dataset_revision: str,
    repo_root: str | Path = ".",
    output_dir: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Validate a merged checkpoint and build its upload metadata folder."""

    if method not in EXACT_MODEL_SPECS:
        raise ReleaseError(f"Unknown method {method!r}; choose grpo or lifpo")
    if not dataset_repo_id or "/" not in dataset_repo_id:
        raise ReleaseError("dataset_repo_id must be a Hugging Face owner/repo ID")
    is_commit_sha = 7 <= len(dataset_revision) <= 64 and all(
        character in "0123456789abcdefABCDEF" for character in dataset_revision
    )
    if not is_commit_sha:
        raise ReleaseError(
            "dataset_revision must be an immutable hexadecimal commit SHA"
        )

    spec = EXACT_MODEL_SPECS[method]
    root = Path(repo_root).expanduser().resolve()
    model_path = _path_from_root(root, model_dir)
    metadata_output = _path_from_root(
        root,
        output_dir or (DEFAULT_MODEL_OUTPUT_ROOT / method),
    )
    validation = validate_merged_model(model_path)

    config_sources = [
        _path_from_root(root, spec.training_config),
        _path_from_root(root, spec.evaluation_config),
    ]
    for path in config_sources:
        _require_file(path, "model provenance config")

    temp = _prepare_output(metadata_output, force=force)
    try:
        config_entries: list[dict[str, Any]] = []
        for source in config_sources:
            relative = source.relative_to(root)
            target = temp / "provenance" / relative
            _copy_file(source, target)
            config_entries.append(
                {
                    "source_path": relative.as_posix(),
                    "packaged_path": target.relative_to(temp).as_posix(),
                    "bytes": source.stat().st_size,
                    "sha256": sha256_file(source),
                }
            )

        (temp / "README.md").write_text(
            _model_card(
                spec,
                dataset_repo_id=dataset_repo_id,
                dataset_revision=dataset_revision,
            ),
            encoding="utf-8",
        )
        metadata_files = sorted(
            (
                _file_entry(path, temp)
                for path in temp.rglob("*")
                if path.is_file() and path.name != "release_manifest.json"
            ),
            key=lambda item: item["path"],
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "model",
            "method": spec.method,
            "display_name": spec.display_name,
            "base_model": BASE_MODEL,
            "created_at_utc": _utc_now(),
            "source_artifact": f"{spec.display_name} evaluated checkpoint",
            "checkpoint_step": spec.checkpoint_step,
            "dataset": {
                "repo_id": dataset_repo_id,
                "revision": dataset_revision,
            },
            "evaluation_protocol": spec.evaluation_protocol,
            "git": _git_provenance(root),
            "exact_configs": config_entries,
            "model_validation": {
                key: value for key, value in validation.items() if key != "model_dir"
            },
            "metadata_files": metadata_files,
            "manifest_self_hash_omitted": True,
        }
        _write_json(temp / "release_manifest.json", manifest)
        _install_staging(temp, metadata_output)
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise

    return {
        "artifact_type": "model",
        "method": method,
        "checkpoint_step": spec.checkpoint_step,
        "_model_dir": str(model_path),
        "metadata_dir": str(metadata_output),
        "dataset_repo_id": dataset_repo_id,
        "dataset_revision": dataset_revision,
        "total_bytes": validation["total_bytes"],
    }


def _hf_api() -> Any:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ReleaseError(
            "Publishing requires huggingface_hub in the active environment"
        ) from exc
    return HfApi()


def _commit_revision(commit: Any, api: Any, repo_id: str, repo_type: str) -> str:
    revision = getattr(commit, "oid", None)
    if revision:
        return str(revision)
    info = api.repo_info(repo_id=repo_id, repo_type=repo_type)
    revision = getattr(info, "sha", None)
    if not revision:
        raise ReleaseError(f"Hugging Face did not return a revision for {repo_id}")
    return str(revision)


def upload_dataset_release(
    package_dir: str | Path,
    *,
    repo_id: str,
    private: bool = False,
    api: Any | None = None,
) -> str:
    package = Path(package_dir).expanduser().resolve()
    _require_file(package / "release_manifest.json", "dataset release manifest")
    client = api or _hf_api()
    try:
        client.create_repo(
            repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True
        )
        commit = client.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(package),
            commit_message="Publish exact HypoDiverse dataset",
            ignore_patterns=["**/.DS_Store", "**/.cache/**"],
        )
        return _commit_revision(commit, client, repo_id, "dataset")
    except ReleaseError:
        raise
    except Exception as exc:
        raise ReleaseError(f"Hugging Face dataset upload failed: {exc}") from exc


def upload_model_release(
    model_dir: str | Path,
    metadata_dir: str | Path,
    *,
    repo_id: str,
    private: bool = False,
    api: Any | None = None,
) -> str:
    model = Path(model_dir).expanduser().resolve()
    metadata = Path(metadata_dir).expanduser().resolve()
    validate_merged_model(model)
    _require_file(metadata / "release_manifest.json", "model release manifest")
    client = api or _hf_api()
    try:
        client.create_repo(
            repo_id=repo_id, repo_type="model", private=private, exist_ok=True
        )
        client.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(model),
            commit_message="Upload exact merged model checkpoint",
            ignore_patterns=["**/.DS_Store", "**/.cache/**"],
        )
        commit = client.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(metadata),
            commit_message="Add reproducibility card and provenance manifest",
            ignore_patterns=["**/.DS_Store", "**/.cache/**"],
        )
        return _commit_revision(commit, client, repo_id, "model")
    except ReleaseError:
        raise
    except Exception as exc:
        raise ReleaseError(f"Hugging Face model upload failed: {exc}") from exc


def _default_model_dir(method: str, model_root: str | Path) -> Path:
    return (
        Path(model_root)
        / "eval_checkpoints"
        / EXACT_MODEL_SPECS[method].merged_directory_name
    )


def _require_push_repo(value: str | None, label: str) -> str:
    if not value:
        raise ReleaseError(f"{label} is required with --push")
    return value


def _add_dataset_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-file", default=str(DEFAULT_TRAIN_FILE))
    parser.add_argument("--validation-file", default=str(DEFAULT_VALIDATION_FILE))
    parser.add_argument("--test-file", default=str(DEFAULT_TEST_FILE))


def _add_publish_flags(parser: argparse.ArgumentParser) -> None:
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--push", action="store_true", help="Upload after packaging")
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Package and validate without uploading (the default)",
    )
    parser.add_argument(
        "--private", action="store_true", help="Create private Hub repos"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace a non-release staging directory",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Package and publish exact HypoDiverse artifacts"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset = subparsers.add_parser("dataset", help="Package the exact dataset")
    dataset.add_argument("--repo-root", default=".")
    _add_dataset_paths(dataset)
    dataset.add_argument("--output-dir", default=str(DEFAULT_DATASET_OUTPUT))
    dataset.add_argument("--repo-id", help="Hugging Face dataset owner/repo")
    _add_publish_flags(dataset)

    model = subparsers.add_parser("model", help="Package one exact model release")
    model.add_argument("--repo-root", default=".")
    model.add_argument("--method", choices=sorted(EXACT_MODEL_SPECS), required=True)
    model.add_argument("--model-dir")
    model.add_argument(
        "--model-root", default=os.environ.get("MODEL_ROOT", ".cache/models")
    )
    model.add_argument("--output-dir")
    model.add_argument("--repo-id", help="Hugging Face model owner/repo")
    model.add_argument("--dataset-repo-id", required=True)
    model.add_argument("--dataset-revision", required=True)
    _add_publish_flags(model)

    all_parser = subparsers.add_parser(
        "all", help="Publish dataset first, then both pinned models"
    )
    all_parser.add_argument("--repo-root", default=".")
    _add_dataset_paths(all_parser)
    all_parser.add_argument("--dataset-output-dir", default=str(DEFAULT_DATASET_OUTPUT))
    all_parser.add_argument("--dataset-repo-id", required=True)
    all_parser.add_argument(
        "--dataset-revision",
        help="Pinned existing revision required for an all-command dry run",
    )
    all_parser.add_argument(
        "--model-root", default=os.environ.get("MODEL_ROOT", ".cache/models")
    )
    all_parser.add_argument("--grpo-model-dir")
    all_parser.add_argument("--lifpo-model-dir")
    all_parser.add_argument("--grpo-output-dir")
    all_parser.add_argument("--lifpo-output-dir")
    all_parser.add_argument("--grpo-repo-id")
    all_parser.add_argument("--lifpo-repo-id")
    _add_publish_flags(all_parser)
    return parser


def _run_dataset(args: argparse.Namespace) -> dict[str, Any]:
    result = build_dataset_release(
        repo_root=args.repo_root,
        train_file=args.train_file,
        validation_file=args.validation_file,
        test_file=args.test_file,
        output_dir=args.output_dir,
        force=args.force,
    )
    if args.push:
        repo_id = _require_push_repo(args.repo_id, "--repo-id")
        result["repo_id"] = repo_id
        result["revision"] = upload_dataset_release(
            result["output_dir"], repo_id=repo_id, private=args.private
        )
    return result


def _run_model(args: argparse.Namespace) -> dict[str, Any]:
    model_dir = args.model_dir or _default_model_dir(args.method, args.model_root)
    result = build_model_release(
        method=args.method,
        model_dir=model_dir,
        dataset_repo_id=args.dataset_repo_id,
        dataset_revision=args.dataset_revision,
        repo_root=args.repo_root,
        output_dir=args.output_dir,
        force=args.force,
    )
    if args.push:
        repo_id = _require_push_repo(args.repo_id, "--repo-id")
        result["repo_id"] = repo_id
        result["revision"] = upload_model_release(
            result["_model_dir"],
            result["metadata_dir"],
            repo_id=repo_id,
            private=args.private,
        )
    return result


def _run_all(args: argparse.Namespace) -> dict[str, Any]:
    if not args.push and not args.dataset_revision:
        raise ReleaseError(
            "An all-command dry run requires --dataset-revision so model cards "
            "still reference an immutable Hub revision"
        )
    if args.push:
        _require_push_repo(args.grpo_repo_id, "--grpo-repo-id")
        _require_push_repo(args.lifpo_repo_id, "--lifpo-repo-id")

    dataset = build_dataset_release(
        repo_root=args.repo_root,
        train_file=args.train_file,
        validation_file=args.validation_file,
        test_file=args.test_file,
        output_dir=args.dataset_output_dir,
        force=args.force,
    )
    if args.push:
        dataset_revision = upload_dataset_release(
            dataset["output_dir"],
            repo_id=args.dataset_repo_id,
            private=args.private,
        )
        dataset["repo_id"] = args.dataset_repo_id
        dataset["revision"] = dataset_revision
    else:
        dataset_revision = args.dataset_revision

    models: dict[str, Any] = {}
    for method in ("grpo", "lifpo"):
        supplied_model_dir = getattr(args, f"{method}_model_dir")
        model_dir = supplied_model_dir or _default_model_dir(method, args.model_root)
        result = build_model_release(
            method=method,
            model_dir=model_dir,
            dataset_repo_id=args.dataset_repo_id,
            dataset_revision=dataset_revision,
            repo_root=args.repo_root,
            output_dir=getattr(args, f"{method}_output_dir"),
            force=args.force,
        )
        if args.push:
            repo_id = getattr(args, f"{method}_repo_id")
            result["repo_id"] = repo_id
            result["revision"] = upload_model_release(
                result["_model_dir"],
                result["metadata_dir"],
                repo_id=repo_id,
                private=args.private,
            )
        models[method] = result
    return {"dataset": dataset, "models": models}


def _public_result(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _public_result(item)
            for key, item in value.items()
            if not key.startswith("_")
        }
    if isinstance(value, list):
        return [_public_result(item) for item in value]
    return value


def _public_error(message: str) -> str:
    for spec in EXACT_MODEL_SPECS.values():
        message = message.replace(
            spec.release_directory, f"{spec.display_name}-checkpoint"
        )
        message = message.replace(
            spec.training_config.name, f"{spec.method}-training-config"
        )
    return message


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "dataset":
            result = _run_dataset(args)
        elif args.command == "model":
            result = _run_model(args)
        else:
            result = _run_all(args)
    except ReleaseError as exc:
        parser.exit(2, f"error: {_public_error(str(exc))}\n")
    print(json.dumps(_public_result(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
