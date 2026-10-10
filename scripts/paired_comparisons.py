#!/usr/bin/env python3
"""Paired cell bootstrap and one-sided Wilcoxon tests for Ours versus AdaDetectGPT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


MODELS = ("llama-2-7b", "opt-2.7b", "opt-6.7b", "falcon-7b")
DATASETS = ("c4", "pubmed", "squad", "writing", "xsum")


def whitebox(root: Path) -> tuple[np.ndarray, np.ndarray]:
    ours, ada = [], []
    for model in MODELS:
        for dataset in DATASETS:
            with (root / "results/whitebox/factorial" / model / dataset / "results.json").open() as f:
                result = json.load(f)
            full = result["variants"]["full"] if "variants" in result else result["full"]
            metrics = full.get("metrics", full)
            ours.append(float(metrics.get("auroc", metrics.get("auc"))))
            with (root / "results/whitebox/conditional_baselines" / model / dataset / "results.json").open() as f:
                baseline = json.load(f)
            ada.append(float(baseline["methods"]["AdaDetectGPT-exact"]["auc"]))
    return np.asarray(ours), np.asarray(ada)


def _blackbox_auc(path: Path) -> float:
    with path.open() as f:
        result = json.load(f)
    return float(result["metrics"]["roc_auc"])


def blackbox(root: Path) -> tuple[np.ndarray, np.ndarray]:
    ours, ada = [], []
    base = root / "results/blackbox/conditional_baselines"
    for model in MODELS:
        for dataset in DATASETS:
            with (root / "results/blackbox/factorial" / model / dataset / "full" / "results.json").open() as f:
                result = json.load(f)
            metrics = result.get("metrics", result)
            ours.append(float(metrics.get("auroc", metrics.get("auc"))))
            name = f"{dataset}_{model}.gpt-j-6B_gpt-neo-2.7B.AdaDetectGPT.bspline.json"
            ada.append(_blackbox_auc(base / name))
    return np.asarray(ours), np.asarray(ada)


def compare(ours: np.ndarray, ada: np.ndarray, seed: int, reps: int) -> dict:
    differences = ours - ada
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(reps, differences.size))
    boot = differences[indices].mean(axis=1)
    return {
        "n": int(differences.size),
        "ours_macro": float(ours.mean()),
        "ada_macro": float(ada.mean()),
        "mean_difference": float(differences.mean()),
        "wins": int((differences > 0).sum()),
        "bootstrap_95_ci": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "wilcoxon_one_sided_p": float(wilcoxon(differences, alternative="greater").pvalue),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--bootstrap-reps", type=int, default=100_000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = {
        "whitebox": compare(*whitebox(root), args.seed, args.bootstrap_reps),
        "blackbox": compare(*blackbox(root), args.seed, args.bootstrap_reps),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

