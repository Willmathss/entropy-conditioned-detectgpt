#!/usr/bin/env python3
"""Rebuild paper Table 8 from the 20 per-cell outputs in each setting/split."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean


MODELS = {"llama-2-7b", "opt-2.7b", "opt-6.7b", "falcon-7b"}
DATASETS = {"c4", "pubmed", "squad", "writing", "xsum"}
METRICS = ("auroc", "pr_auc", "pauc_005", "power_001", "power_005")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def settings(root: Path):
    base = root / "results/deployment_ablation"
    yield "White", "Independent", (
        base / "whitebox_independent/nonllama/cell_comparison.csv",
        base / "whitebox_independent/llama_fixed_fulltext/cell_comparison.csv",
    ), "white_independent"
    yield "White", "Fresh", (base / "fresh/whitebox/cell_results.csv",), "standard"
    yield "Black", "Independent", (base / "blackbox_independent/cell_comparison.csv",), "standard"
    yield "Black", "Fresh", (base / "fresh/blackbox/cell_results.csv",), "standard"


def build(root: Path) -> list[dict[str, str | float]]:
    output = []
    for setting, split, paths, schema in settings(root):
        rows = [row for path in paths for row in read_rows(path)]
        cells = {(row.get("source_model") or row.get("model"), row["dataset"]) for row in rows}
        expected = {(model, dataset) for model in MODELS for dataset in DATASETS}
        assert len(rows) == 20 and cells == expected, (setting, split, len(rows), cells ^ expected)
        prefix = "delta_l_minus_s_"
        fields = (
            ("auc", "pr_auc", "pauc_at_0.05", "power_at_0.01", "power_at_0.05")
            if schema == "white_independent" else
            ("auroc", "pr_auc", "pauc_005", "conformal_power_001", "conformal_power_005")
        )
        values = [mean(float(row[prefix + field]) for row in rows) for field in fields]
        output.append({"setting": setting, "split": split, **dict(zip(METRICS, values))})
    return output


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    rows = build(root)
    path = root / "results/deployment_ablation/table_8.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("setting", "split", *METRICS))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row["setting"], row["split"], *(f"{row[key]:+.4f}" for key in METRICS))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
