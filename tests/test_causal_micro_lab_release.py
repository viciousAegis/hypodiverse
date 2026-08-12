from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scattered_discovery.release.causal_micro_lab import (
    DATASET_PROVENANCE_CONFIGS,
    EXACT_MODEL_SPECS,
    ReleaseError,
    _run_all,
    build_parser,
    build_dataset_release,
    build_model_release,
    inspect_jsonl,
    sha256_file,
    upload_dataset_release,
    upload_model_release,
)


def write_jsonl(path: Path, rows: list[dict]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    path.write_bytes(content)
    return content


def make_row(state_id: str, prompt: str) -> dict:
    state = {"state_id": state_id, "visible_experiments": []}
    return {
        "prompt": prompt,
        "raw_prompt": prompt,
        "state_json": json.dumps(state, sort_keys=True),
        "env_spec_json": json.dumps({"task": {"state": state}}, sort_keys=True),
    }


class FakeCommit:
    def __init__(self, oid: str):
        self.oid = oid


class FakeApi:
    def __init__(self):
        self.created: list[dict] = []
        self.uploaded: list[dict] = []

    def create_repo(self, **kwargs):
        self.created.append(kwargs)

    def upload_folder(self, **kwargs):
        self.uploaded.append(kwargs)
        return FakeCommit(f"{len(self.uploaded):040x}")


class CausalMicroLabReleaseTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for relative in DATASET_PROVENANCE_CONFIGS:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"config: {relative.name}\n", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def _make_dataset(self, *, overlap: bool = False) -> dict[str, bytes]:
        train_rows = [make_row("train-1", "train prompt")]
        validation_state = "train-1" if overlap else "val-1"
        validation_prompt = "train prompt" if overlap else "validation prompt"
        validation_rows = [make_row(validation_state, validation_prompt)]
        test_rows = [make_row("test-1", "test prompt")]
        files = {
            "train": write_jsonl(
                self.root / "data/causal_micro_lab/trainable/verl_train.jsonl",
                train_rows,
            ),
            "validation": write_jsonl(
                self.root / "data/causal_micro_lab/trainable/verl_val.jsonl",
                validation_rows,
            ),
            "test": write_jsonl(
                self.root / "eval_sets/causal_micro_lab/final_v3/verl_test.jsonl",
                test_rows,
            ),
        }
        write_jsonl(
            self.root / "data/causal_micro_lab/trainable/modes.jsonl",
            [{"mode_id": "m1"}],
        )
        write_jsonl(
            self.root / "eval_sets/causal_micro_lab/final_v3/states.jsonl",
            [{"state_id": "test-1"}],
        )
        (self.root / "eval_sets/causal_micro_lab/final_v3/manifest.json").write_text(
            '{"version": 3}\n', encoding="utf-8"
        )
        return files

    def _make_model(self, method: str) -> Path:
        spec = EXACT_MODEL_SPECS[method]
        model = self.root / "models" / spec.merged_directory_name
        model.mkdir(parents=True)
        (model / "config.json").write_text(
            json.dumps({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}),
            encoding="utf-8",
        )
        (model / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
        (model / "tokenizer.json").write_text("{}\n", encoding="utf-8")
        (model / "model-00001-of-00002.safetensors").write_bytes(b"weights-1")
        (model / "model-00002-of-00002.safetensors").write_bytes(b"weights-2")
        (model / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "a": "model-00001-of-00002.safetensors",
                        "b": "model-00002-of-00002.safetensors",
                    }
                }
            ),
            encoding="utf-8",
        )
        return model

    def test_dataset_release_copies_exact_splits_and_auxiliaries(self):
        original = self._make_dataset()
        output = self.root / "release/dataset"
        result = build_dataset_release(repo_root=self.root, output_dir=output)

        self.assertEqual((output / "data/train.jsonl").read_bytes(), original["train"])
        self.assertEqual(
            (output / "data/validation.jsonl").read_bytes(), original["validation"]
        )
        self.assertEqual((output / "data/test.jsonl").read_bytes(), original["test"])
        self.assertTrue((output / "source/trainable/modes.jsonl").is_file())
        self.assertTrue((output / "source/final_v3/states.jsonl").is_file())
        self.assertTrue((output / "source/final_v3/manifest.json").is_file())

        manifest = json.loads((output / "release_manifest.json").read_text())
        self.assertEqual(result["rows"], {"train": 1, "validation": 1, "test": 1})
        self.assertEqual(len(manifest["exact_configs"]), 4)
        self.assertFalse(manifest["split_overlap"]["any_state_id_overlap"])
        self.assertFalse(manifest["split_overlap"]["any_prompt_overlap"])
        file_hashes = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
        self.assertEqual(
            file_hashes["data/train.jsonl"], sha256_file(output / "data/train.jsonl")
        )
        self.assertIn("configs:", (output / "README.md").read_text())

    def test_dataset_release_reports_state_and_prompt_overlap(self):
        self._make_dataset(overlap=True)
        output = self.root / "release/dataset"
        build_dataset_release(repo_root=self.root, output_dir=output)
        manifest = json.loads((output / "release_manifest.json").read_text())
        overlap = manifest["split_overlap"]
        self.assertTrue(overlap["any_state_id_overlap"])
        self.assertTrue(overlap["any_prompt_overlap"])
        self.assertEqual(overlap["state_ids"]["train__validation"]["count"], 1)
        self.assertEqual(overlap["prompt_sha256"]["train__validation"]["count"], 1)

    def test_invalid_jsonl_is_rejected(self):
        path = self.root / "broken.jsonl"
        path.write_text('{"ok": true}\nnot-json\n', encoding="utf-8")
        with self.assertRaisesRegex(ReleaseError, "Invalid JSON"):
            inspect_jsonl(path)

    def test_model_release_pins_dataset_and_records_exact_checkpoint(self):
        model = self._make_model("lifpo")
        output = self.root / "release/lifpo"
        result = build_model_release(
            method="lifpo",
            model_dir=model,
            dataset_repo_id="owner/hypodiverse",
            dataset_revision="0123456789abcdef",
            repo_root=self.root,
            output_dir=output,
        )
        manifest = json.loads((output / "release_manifest.json").read_text())
        self.assertEqual(result["checkpoint_step"], 55)
        self.assertEqual(manifest["source_artifact"], "LIFPO evaluated checkpoint")
        self.assertNotIn("ips", json.dumps(manifest).lower())
        self.assertEqual(manifest["dataset"]["revision"], "0123456789abcdef")
        self.assertEqual(len(manifest["model_validation"]["weight_files"]), 2)
        card = (output / "README.md").read_text()
        self.assertIn("# HypoDiverse: LIFPO", card)
        self.assertIn("0123456789abcdef", card)
        self.assertIn("finite-budget objective", card)
        self.assertNotIn("ips", card.lower())

    def test_model_release_rejects_moving_dataset_revision(self):
        model = self._make_model("grpo")
        with self.assertRaisesRegex(ReleaseError, "immutable hexadecimal commit SHA"):
            build_model_release(
                method="grpo",
                model_dir=model,
                dataset_repo_id="owner/dataset",
                dataset_revision="main",
                repo_root=self.root,
            )

    def test_upload_helpers_use_upload_folder_without_network(self):
        self._make_dataset()
        dataset_output = self.root / "release/dataset"
        build_dataset_release(repo_root=self.root, output_dir=dataset_output)
        api = FakeApi()
        revision = upload_dataset_release(
            dataset_output, repo_id="owner/dataset", api=api
        )
        self.assertEqual(revision, f"{1:040x}")
        self.assertEqual(api.created[0]["repo_type"], "dataset")
        self.assertEqual(api.uploaded[0]["repo_type"], "dataset")

        model = self._make_model("grpo")
        metadata = self.root / "release/grpo"
        build_model_release(
            method="grpo",
            model_dir=model,
            dataset_repo_id="owner/dataset",
            dataset_revision=f"{1:040x}",
            repo_root=self.root,
            output_dir=metadata,
        )
        model_api = FakeApi()
        model_revision = upload_model_release(
            model, metadata, repo_id="owner/grpo", api=model_api
        )
        self.assertEqual(model_revision, f"{2:040x}")
        self.assertEqual(len(model_api.uploaded), 2)
        self.assertEqual(model_api.created[0]["repo_type"], "model")

    def test_all_push_pins_uploaded_dataset_revision_before_models(self):
        self._make_dataset()
        grpo_model = self._make_model("grpo")
        lifpo_model = self._make_model("lifpo")
        dataset_revision = "a" * 40
        events: list[tuple[str, str]] = []

        def fake_dataset_upload(*args, **kwargs):
            events.append(("dataset", kwargs["repo_id"]))
            return dataset_revision

        def fake_model_upload(model_dir, metadata_dir, **kwargs):
            manifest = json.loads(
                (Path(metadata_dir) / "release_manifest.json").read_text()
            )
            self.assertEqual(manifest["dataset"]["revision"], dataset_revision)
            events.append(("model", kwargs["repo_id"]))
            return "b" * 40

        args = build_parser().parse_args(
            [
                "all",
                "--repo-root",
                str(self.root),
                "--dataset-repo-id",
                "owner/dataset",
                "--grpo-repo-id",
                "owner/grpo",
                "--lifpo-repo-id",
                "owner/lifpo",
                "--grpo-model-dir",
                str(grpo_model),
                "--lifpo-model-dir",
                str(lifpo_model),
                "--push",
            ]
        )
        with (
            patch(
                "scattered_discovery.release.causal_micro_lab.upload_dataset_release",
                side_effect=fake_dataset_upload,
            ),
            patch(
                "scattered_discovery.release.causal_micro_lab.upload_model_release",
                side_effect=fake_model_upload,
            ),
        ):
            result = _run_all(args)

        self.assertEqual(
            events,
            [
                ("dataset", "owner/dataset"),
                ("model", "owner/grpo"),
                ("model", "owner/lifpo"),
            ],
        )
        self.assertEqual(result["dataset"]["revision"], dataset_revision)


if __name__ == "__main__":
    unittest.main()
