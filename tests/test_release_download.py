from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scattered_discovery.release import download


class FrozenDatasetDownloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cache_file = self.root / "cache.jsonl"
        self.cache_file.write_bytes(b'{"row": 1}\n')
        self.digest = hashlib.sha256(self.cache_file.read_bytes()).hexdigest()
        self.files = {
            "data/train.jsonl": (Path("data/train.jsonl"), self.digest),
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_downloads_and_verifies_pinned_file(self) -> None:
        with (
            patch.object(download, "FILES", self.files),
            patch.object(
                download,
                "hf_hub_download",
                return_value=str(self.cache_file),
            ) as hub_download,
        ):
            result = download.download_frozen_splits(repo_root=self.root)

        destination = self.root / "data/train.jsonl"
        self.assertEqual(destination.read_bytes(), self.cache_file.read_bytes())
        self.assertEqual(result["data/train.jsonl"], destination.resolve())
        hub_download.assert_called_once_with(
            repo_id=download.DEFAULT_REPO_ID,
            filename="data/train.jsonl",
            repo_type="dataset",
            revision=download.DEFAULT_REVISION,
        )

    def test_reuses_matching_local_file_without_network(self) -> None:
        destination = self.root / "data/train.jsonl"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(self.cache_file.read_bytes())
        with (
            patch.object(download, "FILES", self.files),
            patch.object(download, "hf_hub_download") as hub_download,
        ):
            download.download_frozen_splits(repo_root=self.root)
        hub_download.assert_not_called()

    def test_rejects_noncanonical_local_file_without_force(self) -> None:
        destination = self.root / "data/train.jsonl"
        destination.parent.mkdir(parents=True)
        destination.write_text("different\n", encoding="utf-8")
        with patch.object(download, "FILES", self.files):
            with self.assertRaisesRegex(RuntimeError, "Refusing to replace"):
                download.download_frozen_splits(repo_root=self.root)

    def test_rejects_download_with_wrong_hash(self) -> None:
        files = {"data/train.jsonl": (Path("data/train.jsonl"), "0" * 64)}
        with (
            patch.object(download, "FILES", files),
            patch.object(
                download,
                "hf_hub_download",
                return_value=str(self.cache_file),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                download.download_frozen_splits(repo_root=self.root)


if __name__ == "__main__":
    unittest.main()
