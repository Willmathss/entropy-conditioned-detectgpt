"""CPU-only checks for the self-contained paper data and result summaries."""

from __future__ import annotations

import csv
import json
import sys
import unittest
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_paper_data import audit  # noqa: E402
from build_paper_tables import rows as auroc_rows  # noqa: E402
from build_table_8 import build as table_8_rows  # noqa: E402


class PaperArtifactsTest(unittest.TestCase):
    def test_data_inventory_and_split_integrity(self) -> None:
        actual = audit(ROOT)
        expected = json.loads((ROOT / "data/manifests/paper_data_manifest.json").read_text())
        self.assertEqual(actual, expected)
        self.assertEqual(actual["n_files"], 191)

    def test_tables_1_to_4(self) -> None:
        actual = list(auroc_rows(ROOT))
        with (ROOT / "results/paper_auroc.csv").open(newline="") as handle:
            recorded = list(csv.DictReader(handle))
        self.assertEqual(len(actual), len(recorded))
        self.assertEqual(len(actual), 320)
        for item, stored in zip(actual, recorded):
            self.assertEqual((item["setting"], item["model"], item["dataset"], item["method"]),
                             (stored["setting"], stored["model"], stored["dataset"], stored["method"]))
            self.assertAlmostEqual(item["auroc"], float(stored["auroc"]))
        groups = defaultdict(list)
        for row in actual:
            groups[row["setting"], row["method"]].append(row["auroc"])
        self.assertAlmostEqual(sum(groups["whitebox", "Ours"]) / 20, 0.8374, places=4)
        self.assertAlmostEqual(sum(groups["blackbox", "Ours"]) / 20, 0.7701, places=4)

    def test_table_8(self) -> None:
        actual = table_8_rows(ROOT)
        with (ROOT / "results/deployment_ablation/table_8.csv").open(newline="") as handle:
            recorded = list(csv.DictReader(handle))
        self.assertEqual(len(actual), len(recorded))
        self.assertEqual(len(actual), 4)
        for item, stored in zip(actual, recorded):
            self.assertEqual((item["setting"], item["split"]),
                             (stored["setting"], stored["split"]))
            for metric in ("auroc", "pr_auc", "pauc_005", "power_001", "power_005"):
                self.assertAlmostEqual(item[metric], float(stored[metric]))


if __name__ == "__main__":
    unittest.main()
