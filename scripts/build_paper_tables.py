#!/usr/bin/env python3
"""Build the per-configuration AUROC values underlying paper Tables 1 and 2."""

from __future__ import annotations

import csv
import json
from pathlib import Path


MODELS = ("llama-2-7b", "opt-2.7b", "opt-6.7b", "falcon-7b")
DATASETS = ("c4", "pubmed", "squad", "writing", "xsum")


def load(path: Path):
    with path.open() as handle:
        return json.load(handle)


def rows(root: Path):
    for setting in ("whitebox", "blackbox"):
        for model in MODELS:
            for dataset in DATASETS:
                if setting == "whitebox":
                    result = load(root / "results/whitebox/factorial" / model / dataset / "results.json")
                    full = result["variants"]["full"] if "variants" in result else result["full"]
                    metrics = full.get("metrics", full)
                    ours = metrics.get("auroc", metrics.get("auc"))
                    baseline = load(root / "results/whitebox/conditional_baselines" / model / dataset / "results.json")
                    fast = baseline["methods"]["Fast-DetectGPT-10000"]["auc"]
                    ada = baseline["methods"]["AdaDetectGPT-exact"]["auc"]
                    probability = load(root / "results/whitebox/probability_rank" / model / dataset / "power_result.json")
                    extra = [(method, probability["methods"][method]["auroc"]) for method in
                             ("Likelihood", "Entropy", "Rank", "LogRank", "LRR")]
                else:
                    result = load(root / "results/blackbox/factorial" / model / dataset / "full" / "results.json")
                    metrics = result.get("metrics", result)
                    ours = metrics.get("auroc", metrics.get("auc"))
                    base = root / "results/blackbox/conditional_baselines"
                    prefix = f"{dataset}_{model}.gpt-j-6B_gpt-neo-2.7B."
                    fast = load(base / f"{prefix}sampling_discrepancy_analytic.json")["metrics"]["roc_auc"]
                    ada = load(base / f"{prefix}AdaDetectGPT.bspline.json")["metrics"]["roc_auc"]
                    probability = load(root / "results/blackbox/probability_rank" / model / dataset / "result.json")
                    extra = [(display, probability["methods"][key]["auroc"]) for display, key in
                             (("Likelihood", "likelihood"), ("Entropy", "entropy"),
                              ("Rank", "rank"), ("LogRank", "logrank"), ("LRR", "lrr"))]
                for method, value in (*extra, ("Fast-DetectGPT", fast), ("AdaDetectGPT", ada), ("Ours", ours)):
                    yield {"setting": setting, "model": model, "dataset": dataset,
                           "method": method, "auroc": float(value)}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "results/paper_auroc.csv"
    data = list(rows(root))
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=data[0])
        writer.writeheader()
        writer.writerows(data)
    print(f"wrote {len(data)} rows to {output}")


if __name__ == "__main__":
    main()
