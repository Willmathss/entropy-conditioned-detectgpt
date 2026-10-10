#!/usr/bin/env python3
"""Rebuild Figure 2 and 3 source CSVs, optionally rendering figures outside the repo."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_curve


MODELS = ("llama-2-7b", "opt-2.7b", "opt-6.7b", "falcon-7b")
DATASETS = ("c4", "pubmed", "squad", "writing", "xsum")
METHODS = {
    "Fast-DetectGPT": ("conditional_baselines", "Fast-DetectGPT-10000"),
    "AdaDetectGPT": ("conditional_baselines", "AdaDetectGPT-exact"),
    "Ours": ("factorial", "full"),
}
COLORS = {"Fast-DetectGPT": "#777777", "AdaDetectGPT": "#367a9b", "Ours": "#c26a2b"}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def scores(root: Path, model: str, dataset: str, method: str) -> tuple[np.ndarray, ...]:
    directory, prefix = METHODS[method]
    path = root / "results/whitebox" / directory / model / dataset / "scores.npz"
    with np.load(path, allow_pickle=False) as archive:
        return tuple(np.asarray(archive[f"{prefix}__{split}"], dtype=float)
                     for split in ("human", "machine", "calibration"))


def empirical_roc(human: np.ndarray, machine: np.ndarray, grid: np.ndarray) -> np.ndarray:
    fpr, tpr, _ = roc_curve(
        np.r_[np.zeros(len(human)), np.ones(len(machine))],
        np.r_[human, machine], drop_intermediate=False,
    )
    # Match the paper's diagnostic convention at vertical ROC segments.
    unique_fpr = np.unique(fpr)
    upper_tpr = np.array([np.max(tpr[fpr == value]) for value in unique_fpr])
    return np.interp(grid, unique_fpr, upper_tpr)


def conformal_rate(calibration: np.ndarray, query: np.ndarray, alpha: float) -> float:
    p_values = (1 + (calibration[:, None] >= query).sum(axis=0)) / (len(calibration) + 1)
    return float(np.mean(p_values <= alpha))


def figure_two(root: Path, data_dir: Path, plot_dir: Path | None = None) -> None:
    grid = np.linspace(0, 0.05, 501)
    curve_rows, point_rows = [], []
    figure, axis = plt.subplots(figsize=(7.2, 3.5), constrained_layout=True)
    for method in METHODS:
        curves = []
        points = {alpha: [] for alpha in (0.01, 0.05)}
        for model in MODELS:
            for dataset in DATASETS:
                human, machine, calibration = scores(root, model, dataset, method)
                curves.append(empirical_roc(human, machine, grid))
                for alpha in points:
                    points[alpha].append((conformal_rate(calibration, human, alpha),
                                          conformal_rate(calibration, machine, alpha)))
        mean_curve = np.mean(curves, axis=0)
        for fpr, power in zip(grid, mean_curve):
            curve_rows.append({"method": method, "fpr": float(fpr), "power": float(power)})
        axis.plot(grid * 100, mean_curve * 100, label=method, color=COLORS[method], linewidth=2)
        for alpha, values in points.items():
            fpr, power = np.mean(values, axis=0)
            point_rows.append({"method": method, "nominal_alpha": alpha,
                               "realized_fpr": float(fpr), "power": float(power)})
            axis.scatter([fpr * 100], [power * 100], marker="D", s=42, facecolors="white",
                         edgecolors=COLORS[method], linewidths=1.5, zorder=3)
    axis.set(xlim=(0, 5), ylim=(0, None), xlabel="Human false-positive rate (%)",
             ylabel="Machine detection power (%)")
    axis.legend(frameon=False, loc="upper left", ncol=3, fontsize=8)
    axis.grid(alpha=0.18)
    write_csv(data_dir / "figure_2_roc.csv", curve_rows)
    write_csv(data_dir / "figure_2_conformal_points.csv", point_rows)
    if plot_dir is not None:
        figure.savefig(plot_dir / "figure_2_roc.png", dpi=240)
        figure.savefig(plot_dir / "figure_2_roc.pdf")
    plt.close(figure)


def figure_three(root: Path, data_dir: Path, plot_dir: Path | None = None) -> None:
    examples = (("opt-6.7b", "pubmed"), ("opt-2.7b", "c4"))
    qq_rows, power_rows = [], []
    grid = np.linspace(0.01, 0.05, 41)
    figure, axes = plt.subplots(2, 2, figsize=(8.2, 5.6), constrained_layout=True)
    for row_number, (model, dataset) in enumerate(examples):
        human, machine, _ = scores(root, model, dataset, "Ours")
        ax_qq, ax_power = axes[row_number]
        for label, values, color in (("Human", human, "#35769a"),
                                     ("Machine", machine, "#bb6a35")):
            reference = norm.ppf((np.arange(1, len(values) + 1) - 0.5) / len(values))
            empirical = np.sort((values - values.mean()) / values.std(ddof=1))
            qq_rows.extend({"model": model, "dataset": dataset, "class": label.lower(),
                            "normal_quantile": float(x), "score_quantile": float(y)}
                           for x, y in zip(reference, empirical))
            ax_qq.plot(reference, empirical, color=color, linewidth=1.5, label=label)
        ax_qq.plot([-3, 3], [-3, 3], color="#777777", linestyle="--", linewidth=1)
        ax_qq.set(xlim=(-3, 3), ylim=(-3, 3), xlabel="Standard normal quantile",
                  ylabel="Standardized score quantile", title=f"{model} / {dataset}")
        ax_qq.legend(frameon=False, fontsize=8)

        diagnostic = root / "results/gaussian_diagnostics" / model / dataset
        partitions = json.loads((diagnostic / "diagnostic_partitions.json").read_text())
        with (diagnostic / "gaussian_prediction_partitions.csv").open(newline="") as handle:
            records = [item for item in csv.DictReader(handle)
                       if item["family"] == "shared" and float(item["alpha"]) == 0.01]
        indexed = {(int(item["repeat"]), int(item["fold"])): item for item in records}
        predicted, empirical = [], []
        for split in partitions:
            fit = indexed[(split["repeat"], split["fold"])]
            mu_h, mu_m, sd = (float(fit[key]) for key in ("mu_h", "mu_m", "sd_shared"))
            threshold = mu_h + sd * norm.ppf(1 - grid)
            predicted.append(norm.sf((threshold - mu_m) / sd))
            eval_human = human[split["eval_human_indices"]]
            eval_machine = machine[split["eval_machine_indices"]]
            empirical.append(empirical_roc(eval_human, eval_machine, grid))
        mean_predicted, mean_empirical = np.mean(predicted, axis=0), np.mean(empirical, axis=0)
        power_rows.extend({"model": model, "dataset": dataset, "fpr": float(fpr),
                           "gaussian_predicted_power": float(pred),
                           "heldout_empirical_power": float(obs)}
                          for fpr, pred, obs in zip(grid, mean_predicted, mean_empirical))
        ax_power.plot(grid * 100, mean_empirical * 100, color="#35769a", linewidth=2,
                      label="Empirical")
        ax_power.plot(grid * 100, mean_predicted * 100, color="#bb6a35", linestyle="--",
                      linewidth=2, label="Gaussian prediction")
        ax_power.set(xlim=(1, 5), xlabel="Human false-positive rate (%)",
                     ylabel="Detection power (%)", title=f"{model} / {dataset}")
        ax_power.legend(frameon=False, fontsize=8)
        ax_power.grid(alpha=0.18)
    write_csv(data_dir / "figure_3_qq.csv", qq_rows)
    write_csv(data_dir / "figure_3_power.csv", power_rows)
    if plot_dir is not None:
        figure.savefig(plot_dir / "figure_3_gaussian.png", dpi=240)
        figure.savefig(plot_dir / "figure_3_gaussian.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-dir", type=Path, help="Optional directory for reconstructed plots")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "results/figure_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_dir is not None:
        args.plot_dir.mkdir(parents=True, exist_ok=True)
    figure_two(root, data_dir, args.plot_dir)
    figure_three(root, data_dir, args.plot_dir)
    print("Rebuilt Figure 2 and 3 data from the final score artifacts")


if __name__ == "__main__":
    main()
