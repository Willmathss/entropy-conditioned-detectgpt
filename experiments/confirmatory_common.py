#!/usr/bin/env python3
"""Shared utilities for the frozen-L_theta confirmatory experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import (
    auc as sklearn_auc,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


SCHEMA_VERSION = 1
PROTOCOL_NAME = "fresh-frozen-ltheta-confirmation-20260902"
BOOTSTRAP_SEED = 20260902
BOOTSTRAP_REPETITIONS = 20_000
METRIC_NAMES = (
    "auroc",
    "pr_auc",
    "pauc_001",
    "pauc_005",
    "tpr_at_fpr_001",
    "tpr_at_fpr_005",
    "conformal_fpr_001",
    "conformal_power_001",
    "conformal_fpr_005",
    "conformal_power_005",
)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def text_sha256(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def prefix_sha256(text: str, words: int = 48) -> str:
    prefix = " ".join(normalize_text(text).split()[:words])
    return hashlib.sha256(prefix.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def validate_score_inputs(
    u: np.ndarray,
    theta: np.ndarray,
    v: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    u_array = np.asarray(u, dtype=np.float64)
    flat = np.asarray(theta, dtype=np.float64).reshape(-1)
    if u_array.ndim != 2 or u_array.shape[1] != len(flat):
        raise ValueError(f"U/theta mismatch: U={u_array.shape}, theta={flat.shape}")
    if not np.isfinite(u_array).all() or not np.isfinite(flat).all():
        raise FloatingPointError("non-finite U or theta")
    if v is None:
        return u_array, flat, None
    v_array = np.asarray(v, dtype=np.float64)
    expected = (len(u_array), len(flat), len(flat))
    if v_array.shape != expected:
        raise ValueError(f"V mismatch: got {v_array.shape}, expected {expected}")
    if not np.isfinite(v_array).all():
        raise FloatingPointError("non-finite V")
    return u_array, flat, v_array


def score_ltheta(u: np.ndarray, theta: np.ndarray) -> np.ndarray:
    u_array, flat, _ = validate_score_inputs(u, theta)
    result = u_array @ flat
    if not np.isfinite(result).all():
        raise FloatingPointError("non-finite L_theta score")
    return result


def score_stheta(
    u: np.ndarray,
    v: np.ndarray,
    theta: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    u_array, flat, v_array = validate_score_inputs(u, theta, v)
    assert v_array is not None
    numerator = u_array @ flat
    denominator2 = np.einsum("d,nde,e->n", flat, v_array, flat, optimize=True)
    denominator = np.sqrt(np.clip(denominator2, epsilon, None))
    result = numerator / denominator
    if not np.isfinite(result).all() or not np.isfinite(denominator).all():
        raise FloatingPointError("non-finite S_theta score")
    return result, denominator


def labels_scores(human: np.ndarray, machine: np.ndarray):
    human = np.asarray(human, dtype=np.float64)
    machine = np.asarray(machine, dtype=np.float64)
    if human.ndim != 1 or machine.ndim != 1 or not len(human) or not len(machine):
        raise ValueError("human and machine scores must be non-empty vectors")
    return (
        np.concatenate([np.zeros(len(human)), np.ones(len(machine))]),
        np.concatenate([human, machine]),
    )


def normalized_partial_auc(
    human: np.ndarray, machine: np.ndarray, max_fpr: float
) -> float:
    if not 0 < max_fpr <= 1:
        raise ValueError("max_fpr must be in (0, 1]")
    labels, scores = labels_scores(human, machine)
    fpr, tpr, _ = roc_curve(labels, scores)
    if not np.any(np.isclose(fpr, max_fpr, rtol=0.0, atol=1e-15)):
        index = int(np.searchsorted(fpr, max_fpr))
        left = max(index - 1, 0)
        right = min(index, len(fpr) - 1)
        if fpr[right] == fpr[left]:
            tpr_at = tpr[right]
        else:
            weight = (max_fpr - fpr[left]) / (fpr[right] - fpr[left])
            tpr_at = tpr[left] + weight * (tpr[right] - tpr[left])
        fpr = np.insert(fpr, index, max_fpr)
        tpr = np.insert(tpr, index, tpr_at)
    keep = fpr <= max_fpr + 1e-15
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(tpr[keep], fpr[keep]) / max_fpr)


def tpr_at_fpr(human: np.ndarray, machine: np.ndarray, max_fpr: float) -> float:
    labels, scores = labels_scores(human, machine)
    fpr, tpr, _ = roc_curve(labels, scores)
    eligible = tpr[fpr <= max_fpr + 1e-15]
    return float(eligible.max()) if len(eligible) else 0.0


def conformal_pvalues(calibration: np.ndarray, scores: np.ndarray) -> np.ndarray:
    calibration = np.sort(np.asarray(calibration, dtype=np.float64))
    scores = np.asarray(scores, dtype=np.float64)
    if calibration.ndim != 1 or scores.ndim != 1 or not len(calibration):
        raise ValueError("calibration and test scores must be vectors")
    counts_ge = len(calibration) - np.searchsorted(calibration, scores, side="left")
    return (1.0 + counts_ge) / float(len(calibration) + 1)


def metrics_from_scores(
    human: np.ndarray,
    machine: np.ndarray,
    calibration: np.ndarray,
    alphas: Sequence[float] = (0.01, 0.05),
) -> dict[str, float]:
    labels, scores = labels_scores(human, machine)
    precision, recall, _ = precision_recall_curve(labels, scores)
    result: dict[str, float] = {
        "auroc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(sklearn_auc(recall, precision)),
    }
    for level in (0.01, 0.05):
        suffix = f"{int(level * 100):03d}"
        result[f"pauc_{suffix}"] = normalized_partial_auc(human, machine, level)
        result[f"tpr_at_fpr_{suffix}"] = tpr_at_fpr(human, machine, level)
    human_p = conformal_pvalues(calibration, human)
    machine_p = conformal_pvalues(calibration, machine)
    for alpha in alphas:
        if not 0 < alpha < 1:
            raise ValueError("all alpha values must be in (0, 1)")
        suffix = f"{int(round(alpha * 100)):03d}"
        result[f"conformal_fpr_{suffix}"] = float(np.mean(human_p <= alpha))
        result[f"conformal_power_{suffix}"] = float(np.mean(machine_p <= alpha))
    return result


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and ordered[stop] == ordered[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or len(left) < 2:
        return float("nan")
    if float(left.std()) == 0.0 or float(right.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def score_diagnostics(
    human_l: np.ndarray,
    machine_l: np.ndarray,
    human_s: np.ndarray,
    machine_s: np.ndarray,
    human_d: np.ndarray,
    machine_d: np.ndarray,
    human_lengths: np.ndarray,
    machine_lengths: np.ndarray,
) -> dict[str, float]:
    ltheta = np.concatenate([human_l, machine_l])
    stheta = np.concatenate([human_s, machine_s])
    denominator = np.concatenate([human_d, machine_d])
    lengths = np.concatenate([human_lengths, machine_lengths]).astype(float)
    mean_denominator = float(denominator.mean())
    return {
        "spearman_ltheta_stheta": safe_correlation(
            average_ranks(ltheta), average_ranks(stheta)
        ),
        "pearson_ltheta_length": safe_correlation(ltheta, lengths),
        "pearson_stheta_length": safe_correlation(stheta, lengths),
        "pearson_denominator_length": safe_correlation(denominator, lengths),
        "denominator_mean": mean_denominator,
        "denominator_cv": (
            float(denominator.std() / mean_denominator)
            if mean_denominator != 0
            else float("nan")
        ),
        "denominator_human_mean": float(np.mean(human_d)),
        "denominator_machine_mean": float(np.mean(machine_d)),
    }


def flatten_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def macro_rows(cell_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if not cell_rows:
        raise ValueError("cannot summarize zero cells")
    row: dict[str, Any] = {"n_cells": len(cell_rows), "primary_score": "L_theta"}
    for name in METRIC_NAMES:
        l_values = np.asarray([item[f"ltheta_{name}"] for item in cell_rows], dtype=float)
        s_values = np.asarray([item[f"stheta_{name}"] for item in cell_rows], dtype=float)
        row[f"mean_ltheta_{name}"] = float(l_values.mean())
        row[f"mean_stheta_{name}"] = float(s_values.mean())
        row[f"mean_delta_l_minus_s_{name}"] = float((l_values - s_values).mean())
        row[f"ltheta_better_cells_{name}"] = int(np.sum(l_values > s_values))
        row[f"stheta_better_cells_{name}"] = int(np.sum(s_values > l_values))
    return [row]


def two_way_bootstrap_rows(
    cell_rows: Sequence[dict[str, Any]],
    *,
    seed: int = BOOTSTRAP_SEED,
    repetitions: int = BOOTSTRAP_REPETITIONS,
) -> list[dict[str, Any]]:
    """Descriptive model-by-domain bootstrap for paired L-minus-S deltas."""
    if repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    if not cell_rows:
        raise ValueError("cannot bootstrap zero cells")
    model_field = "model" if all("model" in row for row in cell_rows) else "source_model"
    if not all(model_field in row and "dataset" in row for row in cell_rows):
        raise ValueError("cell rows must identify model and dataset")
    models = sorted({str(row[model_field]) for row in cell_rows})
    datasets = sorted({str(row["dataset"]) for row in cell_rows})
    if len(models) < 2 or len(datasets) < 2:
        raise ValueError("two-way bootstrap needs at least two models and domains")
    indexed = {(str(row[model_field]), str(row["dataset"])): row for row in cell_rows}
    if len(indexed) != len(cell_rows) or len(indexed) != len(models) * len(datasets):
        raise ValueError("two-way bootstrap requires one complete model-domain grid")

    rng = np.random.default_rng(seed)
    model_draws = rng.integers(0, len(models), size=(repetitions, len(models)))
    dataset_draws = rng.integers(
        0, len(datasets), size=(repetitions, len(datasets))
    )
    output: list[dict[str, Any]] = []
    for metric in METRIC_NAMES:
        matrix = np.asarray(
            [
                [
                    float(indexed[(model, dataset)][f"delta_l_minus_s_{metric}"])
                    for dataset in datasets
                ]
                for model in models
            ],
            dtype=np.float64,
        )
        samples = np.empty(repetitions, dtype=np.float64)
        for index in range(repetitions):
            samples[index] = matrix[model_draws[index]][:, dataset_draws[index]].mean()
        lower, upper = np.quantile(samples, [0.025, 0.975])
        output.append(
            {
                "metric": metric,
                "primary_endpoint": metric == "auroc",
                "observed_mean_delta_l_minus_s": float(matrix.mean()),
                "two_way_bootstrap_ci95_lower": float(lower),
                "two_way_bootstrap_ci95_upper": float(upper),
                "bootstrap_probability_delta_le_zero": float(
                    (1 + np.count_nonzero(samples <= 0.0)) / (repetitions + 1)
                ),
                "models": len(models),
                "datasets": len(datasets),
                "repetitions": repetitions,
                "seed": seed,
                "interpretation": "descriptive_model_by_domain_resampling",
            }
        )
    return output


def require_validation_lock(
    lock_path: Path,
    generated_root: Path,
    models: Sequence[str],
    datasets: Sequence[str],
) -> dict[str, Any]:
    lock = read_json(lock_path)
    if lock.get("status") != "passed_and_frozen":
        raise ValueError(f"{lock_path}: validation lock is not passed_and_frozen")
    if lock.get("protocol_name") != PROTOCOL_NAME:
        raise ValueError(f"{lock_path}: unexpected protocol name")
    files = lock.get("generated_files")
    if not isinstance(files, dict):
        raise ValueError(f"{lock_path}: missing generated_files")
    for model in models:
        for dataset in datasets:
            key = f"{model}/{dataset}"
            expected = files.get(key)
            path = generated_root / model / f"{dataset}.jsonl"
            if not isinstance(expected, dict) or not path.is_file():
                raise FileNotFoundError(f"validation lock does not cover {path}")
            actual = file_sha256(path)
            if actual != expected.get("sha256"):
                raise ValueError(f"{path}: SHA-256 differs from validation lock")
    return lock
