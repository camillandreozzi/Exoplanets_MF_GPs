"""Integrity tests for the research inputs and their loader."""

from __future__ import annotations

import hashlib
import unittest

from exoplanets_mf.data import load_all
from exoplanets_mf.paths import PROJECT_ROOT


class DataIntegrityTests(unittest.TestCase):
    def test_dataset_shapes_and_alignment(self) -> None:
        data = load_all()

        self.assertEqual(data["XHF"].shape, (97, 9))
        self.assertEqual(data["YHF"].shape, (97, 195))
        self.assertEqual(data["YLF"].shape, (97, 195))
        self.assertEqual(data["XLF_10k"].shape, (10_000, 9))
        self.assertEqual(data["YLF_10k"].shape, (10_000, 195))
        self.assertEqual(data["observed"].shape, (195, 4))
        self.assertEqual(data["wavelengths"].shape, (195,))

    def test_input_files_match_recorded_checksums(self) -> None:
        manifest = PROJECT_ROOT / "data" / "checksums.sha256"

        for line in manifest.read_text(encoding="utf-8").splitlines():
            expected, relative_path = line.split(maxsplit=1)
            path = PROJECT_ROOT / relative_path
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, expected, relative_path)


if __name__ == "__main__":
    unittest.main()
