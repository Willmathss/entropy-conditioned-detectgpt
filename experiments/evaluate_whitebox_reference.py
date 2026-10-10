#!/usr/bin/env python3
"""Evaluate Fast/Ada exact baselines and Water-Full.

Run ``--mode development`` first.  It performs paired five-fold evaluation,
fits final Ada and Water witnesses on the complete development cell, and
stores them. Then run ``--mode independent`` to apply those frozen witnesses
and the development calibration humans to Water's frozen independent test set.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from build_whitebox_reference_cache import (
    AdaSplineSpec,
    ReferenceSplitCache,
    WaterSplineSpec,
    parse_csv,
    parse_structures,
)
from ada_spline_reference import (
    Ada128Features,
    OfficialAdaSpline,
    fit_witness,
    score_from_features,
)


DEFAULT_MODELS = ["llama-2-7b", "opt-2.7b", "opt-6.7b", "falcon-7b"]
DEFAULT_DATASETS = ["c4", "squad", "xsum", "writing", "pubmed"]


@dataclass
class WaterFeatures:
    u_human: np.ndarray
    v_human: np.ndarray
    u_machine: np.ndarray
    v_machine: np.ndarray
    u_calibration: np.ndarray
    v_calibration: np.ndarray


@dataclass(frozen=True)
class WaterRayleighConfig:
    j_bases: int
    k_bases: int
    smooth_lambda: float
    ridge_lambda: float
    covariance_shrinkage: float

    @property
    def dimension(self) -> int:
        return self.j_bases * self.k_bases

    @property
    def name(self) -> str:
        return (
            f"J{self.j_bases}_K{self.k_bases}"
            f"_smooth{self.smooth_lambda:g}"
            f"_ridge{self.ridge_lambda:g}"
            f"_shrink{self.covariance_shrinkage:g}"
        )


@dataclass
class WaterRayleighFit:
    theta: np.ndarray
    train_auc: float
    condition_number: float
    orientation_flipped: bool


def infer_water_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent if script_dir.name == "scripts" else script_dir.parent


def parse_float_csv(value: str) -> list[float]:
    return [float(item) for item in parse_csv(value)]


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def conformal_p_values(calibration, tests) -> np.ndarray:
    calibration = np.sort(np.asarray(calibration, dtype=np.float64))
    tests = np.asarray(tests, dtype=np.float64)
    smaller = np.searchsorted(calibration, tests, side="left")
    greater_or_equal = len(calibration) - smaller
    return (1.0 + greater_or_equal) / float(len(calibration) + 1)


def full_auc(human, machine) -> float:
    human = np.asarray(human, dtype=np.float64)
    machine = np.asarray(machine, dtype=np.float64)
    labels = np.concatenate(
        [np.zeros(len(human), dtype=np.int8), np.ones(len(machine), dtype=np.int8)]
    )
    return float(roc_auc_score(labels, np.concatenate([human, machine])))


def partial_auc(human, machine, max_fpr: float) -> float:
    human = np.asarray(human, dtype=np.float64)
    machine = np.asarray(machine, dtype=np.float64)
    labels = np.concatenate(
        [np.zeros(len(human), dtype=np.int8), np.ones(len(machine), dtype=np.int8)]
    )
    return float(
        roc_auc_score(
            labels,
            np.concatenate([human, machine]),
            max_fpr=max_fpr,
        )
    )


def fast_scores(cache: ReferenceSplitCache, variant: str, epsilon: float) -> np.ndarray:
    if variant == "mc10000":
        numerator = cache.fast_mc_numerator
        variance = cache.fast_mc_variance
    elif variant == "analytic":
        numerator = cache.fast_exact_numerator
        variance = cache.fast_exact_variance
    else:
        raise ValueError(f"unknown Fast variant {variant!r}")
    return np.asarray(numerator, dtype=np.float64) / np.sqrt(
        np.clip(np.asarray(variance, dtype=np.float64), epsilon, None)
    )


def second_difference_matrix(size: int) -> np.ndarray:
    if size < 3:
        return np.zeros((0, size), dtype=np.float64)
    matrix = np.zeros((size - 2, size), dtype=np.float64)
    rows = np.arange(size - 2)
    matrix[rows, rows] = 1.0
    matrix[rows, rows + 1] = -2.0
    matrix[rows, rows + 2] = 1.0
    return matrix


def tensor_smoothness_matrix(spec: WaterSplineSpec) -> np.ndarray:
    j, k = spec.j_bases, spec.k_bases
    dz = second_difference_matrix(j)
    dh = second_difference_matrix(k)
    penalty = np.zeros((j * k, j * k), dtype=np.float64)
    if len(dz):
        penalty += np.kron(dz.T @ dz, np.eye(k)) / float((j - 2) * k)
    if len(dh):
        penalty += np.kron(np.eye(j), dh.T @ dh) / float(j * (k - 2))
    return 0.5 * (penalty + penalty.T)


def within_class_covariance(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("at least two Water feature rows are required")
    centered = values - values.mean(axis=0, keepdims=True)
    return centered.T @ centered / float(len(values) - 1)


def make_water_features(
    caches: dict[str, ReferenceSplitCache], spec: WaterSplineSpec
) -> WaterFeatures:
    name = spec.name
    return WaterFeatures(
        u_human=np.asarray(caches["human"].water_u[name], dtype=np.float64),
        v_human=np.asarray(caches["human"].water_v[name], dtype=np.float64),
        u_machine=np.asarray(caches["machine"].water_u[name], dtype=np.float64),
        v_machine=np.asarray(caches["machine"].water_v[name], dtype=np.float64),
        u_calibration=np.asarray(
            caches["calibration"].water_u[name], dtype=np.float64
        ),
        v_calibration=np.asarray(
            caches["calibration"].water_v[name], dtype=np.float64
        ),
    )


def water_scores(
    u: np.ndarray, v: np.ndarray, theta: np.ndarray, epsilon: float
) -> np.ndarray:
    flat = np.asarray(theta, dtype=np.float64).reshape(-1)
    numerator = np.asarray(u, dtype=np.float64) @ flat
    denominator2 = np.einsum(
        "d,nde,e->n", flat, np.asarray(v, dtype=np.float64), flat, optimize=True
    )
    return numerator / np.sqrt(np.clip(denominator2, epsilon, None))


def fit_water_rayleigh(
    features: WaterFeatures,
    train_indices,
    spec: WaterSplineSpec,
    config: WaterRayleighConfig,
    *,
    epsilon: float,
) -> WaterRayleighFit:
    indices = np.asarray(train_indices, dtype=np.int64)
    u_human = features.u_human[indices]
    u_machine = features.u_machine[indices]
    delta = u_machine.mean(axis=0) - u_human.mean(axis=0)
    covariance = within_class_covariance(u_human) + within_class_covariance(
        u_machine
    )
    shrinkage = config.covariance_shrinkage
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("Water covariance shrinkage must be in [0, 1]")
    covariance = (1.0 - shrinkage) * covariance + shrinkage * np.diag(
        np.diag(covariance)
    )
    covariance = 0.5 * (covariance + covariance.T)
    dimension = covariance.shape[0]
    covariance_scale = float(np.trace(covariance) / dimension)
    if not np.isfinite(covariance_scale) or covariance_scale <= 0:
        covariance_scale = 1.0

    penalty = tensor_smoothness_matrix(spec)
    positive = np.linalg.eigvalsh(penalty)
    positive = positive[positive > 1e-12]
    penalty_scale = float(positive.mean()) if len(positive) else 1.0
    system = (
        covariance
        + covariance_scale
        * config.smooth_lambda
        * (penalty / penalty_scale)
        + covariance_scale * config.ridge_lambda * np.eye(dimension)
    )
    system = 0.5 * (system + system.T)
    eigenvalues, eigenvectors = np.linalg.eigh(system)
    floor = max(
        epsilon * covariance_scale,
        np.finfo(np.float64).eps * max(float(eigenvalues[-1]), 1.0),
    )
    safe = np.maximum(eigenvalues, floor)
    theta = eigenvectors @ ((eigenvectors.T @ delta) / safe)
    mean_v = np.mean(
        np.concatenate(
            [features.v_human[indices], features.v_machine[indices]], axis=0
        ),
        axis=0,
    )
    norm2 = float(theta @ mean_v @ theta)
    if np.isfinite(norm2) and norm2 > epsilon:
        theta /= math.sqrt(norm2)
    else:
        norm = float(np.linalg.norm(theta))
        if not np.isfinite(norm) or norm <= 0:
            raise FloatingPointError("Water Rayleigh solver returned zero")
        theta /= norm
    human = water_scores(
        features.u_human[indices], features.v_human[indices], theta, epsilon
    )
    machine = water_scores(
        features.u_machine[indices], features.v_machine[indices], theta, epsilon
    )
    flipped = bool(machine.mean() < human.mean())
    if flipped:
        theta = -theta
        human = -human
        machine = -machine
    return WaterRayleighFit(
        theta=theta.reshape(spec.j_bases, spec.k_bases),
        train_auc=full_auc(human, machine),
        condition_number=float(safe[-1] / safe[0]),
        orientation_flipped=flipped,
    )


def water_config_key(record: dict[str, Any], selection_metric: str) -> tuple:
    primary = record["cv_auc"] if selection_metric == "auc" else record["cv_pauc"]
    secondary = record["cv_pauc"] if selection_metric == "auc" else record["cv_auc"]
    return (
        float(primary),
        float(secondary),
        -int(record["dimension"]),
        float(record["smooth_lambda"]),
        float(record["ridge_lambda"]),
    )


def observed_training_features(
    cache: ReferenceSplitCache,
    basis: OfficialAdaSpline,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    n = cache.num_documents
    d = basis.spec.dimension
    means = np.empty((n, d), dtype=np.float64)
    covariances = np.empty((n, d, d), dtype=np.float64)
    for index in range(n):
        phi = basis.transform(cache.observed_document(index))
        length = len(phi)
        if length < 2:
            raise ValueError(f"{split_name} document {index} has fewer than two tokens")
        mean = phi.mean(axis=0)
        centered = phi - mean
        covariance = (centered.T @ centered) / ((length - 1) * length)
        means[index] = mean
        covariances[index] = 0.5 * (covariance + covariance.T)
    return means, covariances


def load_cell(
    cache_root: Path,
    model: str,
    dataset: str,
    *,
    expected_documents: int,
    expected_fast_samples: int,
    expected_spec: AdaSplineSpec,
    expected_water_structures: list[tuple[int, int]],
    require_legacy_verification: bool,
) -> tuple[dict[str, ReferenceSplitCache], tuple[WaterSplineSpec, ...]]:
    cell_dir = cache_root / model / dataset
    metadata = load_json(cell_dir / "metadata.json")
    if int(metadata["processed_pairs"]) != expected_documents:
        raise ValueError(
            f"{cell_dir}: expected {expected_documents} documents, "
            f"metadata reports {metadata['processed_pairs']}"
        )
    if int(metadata["fast_mc_samples"]) != expected_fast_samples:
        raise ValueError(
            f"{cell_dir}: expected Fast-{expected_fast_samples}, "
            f"found Fast-{metadata['fast_mc_samples']}"
        )
    if not metadata.get("source_model_equals_sampling_model", False):
        raise ValueError(f"{cell_dir}: cache is not white-box source=sampling")
    if not metadata.get("source_model_equals_scoring_model", False):
        raise ValueError(f"{cell_dir}: cache is not white-box source=scoring")
    legacy_offsets_verified = metadata.get(
        "legacy_offsets_verified",
        metadata.get("legacy_cache_verified", False),
    )
    if require_legacy_verification and not legacy_offsets_verified:
        raise ValueError(
            f"{cell_dir}: token offsets were not verified against the frozen "
            "128-cache; rebuild with --legacy-cache-root or explicitly pass "
            "--allow-unverified-token-stream"
        )
    stored_spec = AdaSplineSpec(**metadata["ada_spline"])
    if stored_spec != expected_spec:
        raise ValueError(f"{cell_dir}: spline mismatch {stored_spec} != {expected_spec}")
    water_metadata = metadata.get("water_full", {})
    if not water_metadata.get("enabled", False):
        raise ValueError(
            f"{cell_dir}: cache has no Water-Full statistics; rebuild with v5"
        )
    water_specs = tuple(
        WaterSplineSpec(
            **{
                **value,
                "z_base_knots": tuple(value["z_base_knots"]),
                "h_base_knots": tuple(value["h_base_knots"]),
            }
        )
        for value in water_metadata["specs"]
    )
    stored_structures = [(item.j_bases, item.k_bases) for item in water_specs]
    if stored_structures != expected_water_structures:
        raise ValueError(
            f"{cell_dir}: Water structures {stored_structures} != "
            f"requested {expected_water_structures}"
        )
    caches = {
        split: ReferenceSplitCache.load(cell_dir / f"{split}_reference.npz")
        for split in ("human", "machine", "calibration")
    }
    for split, cache in caches.items():
        if cache.num_documents != expected_documents:
            raise ValueError(
                f"{cell_dir}/{split}: expected {expected_documents}, "
                f"found {cache.num_documents}"
            )
        if cache.ada_u.shape[1:] != (expected_spec.dimension,):
            raise ValueError(f"{cell_dir}/{split}: bad Ada u shape {cache.ada_u.shape}")
        if cache.ada_v.shape[1:] != (
            expected_spec.dimension,
            expected_spec.dimension,
        ):
            raise ValueError(f"{cell_dir}/{split}: bad Ada v shape {cache.ada_v.shape}")
        for water_spec in water_specs:
            name = water_spec.name
            if cache.water_u.get(name, np.empty(0)).shape != (
                expected_documents,
                water_spec.dimension,
            ):
                raise ValueError(
                    f"{cell_dir}/{split}: bad Water u shape for {name}"
                )
            if cache.water_v.get(name, np.empty(0)).shape != (
                expected_documents,
                water_spec.dimension,
                water_spec.dimension,
            ):
                raise ValueError(
                    f"{cell_dir}/{split}: bad Water v shape for {name}"
                )
    return caches, water_specs


def make_ada_features(
    caches: dict[str, ReferenceSplitCache], spec: AdaSplineSpec
) -> Ada128Features:
    basis = OfficialAdaSpline(spec)
    human_mean, human_cov = observed_training_features(
        caches["human"], basis, "human-training"
    )
    machine_mean, machine_cov = observed_training_features(
        caches["machine"], basis, "machine-training"
    )
    return Ada128Features(
        train_mean_human=human_mean,
        train_cov_human=human_cov,
        train_mean_machine=machine_mean,
        train_cov_machine=machine_cov,
        u_human=np.asarray(caches["human"].ada_u, dtype=np.float64),
        v_human=np.asarray(caches["human"].ada_v, dtype=np.float64),
        u_machine=np.asarray(caches["machine"].ada_u, dtype=np.float64),
        v_machine=np.asarray(caches["machine"].ada_v, dtype=np.float64),
        u_calibration=np.asarray(caches["calibration"].ada_u, dtype=np.float64),
        v_calibration=np.asarray(caches["calibration"].ada_v, dtype=np.float64),
    )


def score_metrics(
    human: np.ndarray,
    machine: np.ndarray,
    calibration: np.ndarray,
    alphas: list[float],
    max_fpr: float,
) -> dict[str, Any]:
    human = np.asarray(human, dtype=np.float64)
    machine = np.asarray(machine, dtype=np.float64)
    calibration = np.asarray(calibration, dtype=np.float64)
    human_p = conformal_p_values(calibration, human)
    machine_p = conformal_p_values(calibration, machine)
    return {
        "auc": full_auc(human, machine),
        "pauc": partial_auc(human, machine, max_fpr),
        "max_fpr": max_fpr,
        "metrics_by_alpha": {
            f"{alpha:g}": {
                "alpha": alpha,
                "type_i_error": float(np.mean(human_p <= alpha)),
                "power": float(np.mean(machine_p <= alpha)),
            }
            for alpha in alphas
        },
    }


def fit_development_ada(
    features: Ada128Features,
    *,
    n_pairs: int,
    cv_folds: int,
    seed: int,
    alphas: list[float],
    max_fpr: float,
    solver: str,
    eigenvalue_floor: float,
    ridge: float,
    score_epsilon: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    indices = np.arange(n_pairs, dtype=np.int64)
    folds = KFold(n_splits=cv_folds, shuffle=True, random_state=seed).split(indices)
    oof_human = np.empty(n_pairs, dtype=np.float64)
    oof_machine = np.empty(n_pairs, dtype=np.float64)
    oof_human_p = {alpha: np.empty(n_pairs, dtype=np.float64) for alpha in alphas}
    oof_machine_p = {alpha: np.empty(n_pairs, dtype=np.float64) for alpha in alphas}
    fold_records: list[dict[str, Any]] = []
    for fold_number, (train_indices, valid_indices) in enumerate(folds, 1):
        fit = fit_witness(
            features,
            train_indices,
            solver=solver,
            eigenvalue_floor=eigenvalue_floor,
            ridge=ridge,
        )
        human = score_from_features(
            features.u_human[valid_indices],
            features.v_human[valid_indices],
            fit.beta,
            score_epsilon,
        )
        machine = score_from_features(
            features.u_machine[valid_indices],
            features.v_machine[valid_indices],
            fit.beta,
            score_epsilon,
        )
        calibration = score_from_features(
            features.u_calibration,
            features.v_calibration,
            fit.beta,
            score_epsilon,
        )
        oof_human[valid_indices] = human
        oof_machine[valid_indices] = machine
        human_p = conformal_p_values(calibration, human)
        machine_p = conformal_p_values(calibration, machine)
        for alpha in alphas:
            oof_human_p[alpha][valid_indices] = human_p
            oof_machine_p[alpha][valid_indices] = machine_p
        fold_records.append(
            {
                "fold": fold_number,
                "train_size": int(len(train_indices)),
                "validation_size": int(len(valid_indices)),
                "auc": full_auc(human, machine),
                "beta": fit.beta.tolist(),
                "delta_norm": fit.delta_norm,
                "sigma_min_eigenvalue": fit.sigma_min_eigenvalue,
                "sigma_max_eigenvalue": fit.sigma_max_eigenvalue,
            }
        )

    metrics = {
        "auc": full_auc(oof_human, oof_machine),
        "pauc": partial_auc(oof_human, oof_machine, max_fpr),
        "max_fpr": max_fpr,
        "metrics_by_alpha": {
            f"{alpha:g}": {
                "alpha": alpha,
                "type_i_error": float(np.mean(oof_human_p[alpha] <= alpha)),
                "power": float(np.mean(oof_machine_p[alpha] <= alpha)),
            }
            for alpha in alphas
        },
        "folds": fold_records,
    }
    final_fit = fit_witness(
        features,
        indices,
        solver=solver,
        eigenvalue_floor=eigenvalue_floor,
        ridge=ridge,
    )
    final_calibration = score_from_features(
        features.u_calibration,
        features.v_calibration,
        final_fit.beta,
        score_epsilon,
    )
    metrics["final_fit"] = {
        "beta": final_fit.beta.tolist(),
        "delta_norm": final_fit.delta_norm,
        "sigma_min_eigenvalue": final_fit.sigma_min_eigenvalue,
        "sigma_max_eigenvalue": final_fit.sigma_max_eigenvalue,
    }
    arrays = {
        "human": oof_human,
        "machine": oof_machine,
        "calibration": final_calibration,
    }
    return metrics, arrays


def fit_development_water(
    caches: dict[str, ReferenceSplitCache],
    specs: tuple[WaterSplineSpec, ...],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    features_by_name = {
        spec.name: make_water_features(caches, spec) for spec in specs
    }
    configs = [
        WaterRayleighConfig(
            spec.j_bases,
            spec.k_bases,
            smooth,
            ridge,
            shrinkage,
        )
        for spec, smooth, ridge, shrinkage in itertools.product(
            specs,
            args.water_smooth_lambdas_list,
            args.water_ridge_lambdas_list,
            args.water_covariance_shrinkages_list,
        )
    ]
    folds = list(
        KFold(
            n_splits=args.cv_folds,
            shuffle=True,
            random_state=args.seed,
        ).split(np.arange(args.expected_pairs))
    )
    spec_by_name = {spec.name: spec for spec in specs}
    search_records: list[dict[str, Any]] = []
    for config in configs:
        spec = spec_by_name[f"J{config.j_bases}_K{config.k_bases}"]
        features = features_by_name[spec.name]
        oof_human = np.empty(args.expected_pairs, dtype=np.float64)
        oof_machine = np.empty(args.expected_pairs, dtype=np.float64)
        conditions = []
        for train_indices, valid_indices in folds:
            fit = fit_water_rayleigh(
                features,
                train_indices,
                spec,
                config,
                epsilon=args.score_epsilon,
            )
            oof_human[valid_indices] = water_scores(
                features.u_human[valid_indices],
                features.v_human[valid_indices],
                fit.theta,
                args.score_epsilon,
            )
            oof_machine[valid_indices] = water_scores(
                features.u_machine[valid_indices],
                features.v_machine[valid_indices],
                fit.theta,
                args.score_epsilon,
            )
            conditions.append(fit.condition_number)
        search_records.append(
            {
                **asdict(config),
                "name": config.name,
                "dimension": config.dimension,
                "cv_auc": full_auc(oof_human, oof_machine),
                "cv_pauc": partial_auc(oof_human, oof_machine, args.max_fpr),
                "mean_condition_number": float(np.mean(conditions)),
            }
        )

    selected_record = max(
        search_records,
        key=lambda item: water_config_key(item, args.water_selection_metric),
    )
    selected = WaterRayleighConfig(
        **{
            key: selected_record[key]
            for key in (
                "j_bases",
                "k_bases",
                "smooth_lambda",
                "ridge_lambda",
                "covariance_shrinkage",
            )
        }
    )
    selected_spec = spec_by_name[f"J{selected.j_bases}_K{selected.k_bases}"]
    features = features_by_name[selected_spec.name]
    oof_human = np.empty(args.expected_pairs, dtype=np.float64)
    oof_machine = np.empty(args.expected_pairs, dtype=np.float64)
    oof_human_p = np.empty(args.expected_pairs, dtype=np.float64)
    oof_machine_p = np.empty(args.expected_pairs, dtype=np.float64)
    fold_records = []
    for fold_number, (train_indices, valid_indices) in enumerate(folds, 1):
        fit = fit_water_rayleigh(
            features,
            train_indices,
            selected_spec,
            selected,
            epsilon=args.score_epsilon,
        )
        human = water_scores(
            features.u_human[valid_indices],
            features.v_human[valid_indices],
            fit.theta,
            args.score_epsilon,
        )
        machine = water_scores(
            features.u_machine[valid_indices],
            features.v_machine[valid_indices],
            fit.theta,
            args.score_epsilon,
        )
        calibration = water_scores(
            features.u_calibration,
            features.v_calibration,
            fit.theta,
            args.score_epsilon,
        )
        oof_human[valid_indices] = human
        oof_machine[valid_indices] = machine
        oof_human_p[valid_indices] = conformal_p_values(calibration, human)
        oof_machine_p[valid_indices] = conformal_p_values(calibration, machine)
        fold_records.append(
            {
                "fold": fold_number,
                "auc": full_auc(human, machine),
                "condition_number": fit.condition_number,
                "orientation_flipped": fit.orientation_flipped,
            }
        )

    final_fit = fit_water_rayleigh(
        features,
        np.arange(args.expected_pairs),
        selected_spec,
        selected,
        epsilon=args.score_epsilon,
    )
    final_calibration = water_scores(
        features.u_calibration,
        features.v_calibration,
        final_fit.theta,
        args.score_epsilon,
    )
    metrics = {
        "auc": full_auc(oof_human, oof_machine),
        "pauc": partial_auc(oof_human, oof_machine, args.max_fpr),
        "max_fpr": args.max_fpr,
        "metrics_by_alpha": {
            f"{alpha:g}": {
                "alpha": alpha,
                "type_i_error": float(np.mean(oof_human_p <= alpha)),
                "power": float(np.mean(oof_machine_p <= alpha)),
            }
            for alpha in args.alphas_list
        },
        "folds": fold_records,
        "hyperparameter_selection": {
            "metric": args.water_selection_metric,
            "selected": selected_record,
            "candidate_count": len(search_records),
            "candidates": search_records,
        },
        "final_fit": {
            "theta": final_fit.theta.tolist(),
            "basis_spec": asdict(selected_spec),
            "config": asdict(selected),
            "train_auc": final_fit.train_auc,
            "condition_number": final_fit.condition_number,
            "orientation_flipped": final_fit.orientation_flipped,
        },
    }
    arrays = {
        "human": oof_human,
        "machine": oof_machine,
        "calibration": final_calibration,
    }
    return metrics, arrays


def save_scores(path: Path, methods: dict[str, dict[str, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values: dict[str, np.ndarray] = {}
    for method, arrays in methods.items():
        for split, array in arrays.items():
            values[f"{method}__{split}"] = np.asarray(array, dtype=np.float64)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def summary_rows(
    *,
    model: str,
    dataset: str,
    protocol: str,
    methods: dict[str, dict[str, Any]],
    alphas: list[float],
    n_human: int,
    n_machine: int,
    n_calibration: int,
) -> list[dict[str, Any]]:
    rows = []
    for method, metrics in methods.items():
        row: dict[str, Any] = {
            "model": model,
            "dataset": dataset,
            "protocol": protocol,
            "method": method,
            "auc": metrics["auc"],
            f"pauc_at_{metrics['max_fpr']:g}": metrics["pauc"],
            "n_human": n_human,
            "n_machine": n_machine,
            "n_calibration": n_calibration,
        }
        for alpha in alphas:
            item = metrics["metrics_by_alpha"][f"{alpha:g}"]
            row[f"power_at_{alpha:g}"] = item["power"]
            row[f"type_i_at_{alpha:g}"] = item["type_i_error"]
        rows.append(row)
    return rows


def evaluate_development_cell(
    model: str,
    dataset: str,
    caches: dict[str, ReferenceSplitCache],
    spec: AdaSplineSpec,
    water_specs: tuple[WaterSplineSpec, ...],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    score_arrays: dict[str, dict[str, np.ndarray]] = {}
    methods: dict[str, dict[str, Any]] = {}
    for variant, name in (
        ("mc10000", f"Fast-DetectGPT-{args.expected_fast_samples}"),
        ("analytic", "Fast-DetectGPT-analytic"),
    ):
        arrays = {
            split: fast_scores(cache, variant, args.score_epsilon)
            for split, cache in caches.items()
        }
        score_arrays[name] = arrays
        methods[name] = score_metrics(
            arrays["human"],
            arrays["machine"],
            arrays["calibration"],
            args.alphas_list,
            args.max_fpr,
        )

    features = make_ada_features(caches, spec)
    ada_metrics, ada_arrays = fit_development_ada(
        features,
        n_pairs=args.expected_pairs,
        cv_folds=args.cv_folds,
        seed=args.seed,
        alphas=args.alphas_list,
        max_fpr=args.max_fpr,
        solver=args.solver,
        eigenvalue_floor=args.eigenvalue_floor,
        ridge=args.ridge,
        score_epsilon=args.score_epsilon,
    )
    methods["AdaDetectGPT-exact"] = ada_metrics
    score_arrays["AdaDetectGPT-exact"] = ada_arrays
    water_metrics, water_arrays = fit_development_water(
        caches, water_specs, args
    )
    methods["Water-Full"] = water_metrics
    score_arrays["Water-Full"] = water_arrays
    cell_dir = args.result_root / model / dataset
    save_scores(cell_dir / "scores.npz", score_arrays)
    result = {
        "model": model,
        "dataset": dataset,
        "protocol": {
            "name": "development paired cross-fitting",
            "paired_cv_folds": args.cv_folds,
            "seed": args.seed,
            "expected_pairs": args.expected_pairs,
            "calibration_humans": caches["calibration"].num_documents,
            "fast_mc_samples": args.expected_fast_samples,
            "fast_analytic": "exact full-vocabulary conditional moments",
            "ada_conditional_moments": "exact full-vocabulary",
            "ada_witness_training": "within-cell development folds",
            "water_conditional_moments": "exact full vocabulary",
            "water_witness_training": (
                "paired development-only cross-validation; final theta refit "
                "on complete development cell"
            ),
            "water_candidate_samples_used": 0,
        },
        "spline": asdict(spec),
        "water_splines": [asdict(item) for item in water_specs],
        "solver": {
            "name": args.solver,
            "eigenvalue_floor": args.eigenvalue_floor,
            "ridge": args.ridge,
        },
        "methods": methods,
    }
    atomic_write_json(cell_dir / "results.json", result)
    rows = summary_rows(
        model=model,
        dataset=dataset,
        protocol="development_cross_fitted",
        methods=methods,
        alphas=args.alphas_list,
        n_human=caches["human"].num_documents,
        n_machine=caches["machine"].num_documents,
        n_calibration=caches["calibration"].num_documents,
    )
    return result, rows


def evaluate_independent_cell(
    model: str,
    dataset: str,
    test_caches: dict[str, ReferenceSplitCache],
    dev_caches: dict[str, ReferenceSplitCache],
    water_specs: tuple[WaterSplineSpec, ...],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    development = load_json(args.dev_result_root / model / dataset / "results.json")
    beta = np.asarray(
        development["methods"]["AdaDetectGPT-exact"]["final_fit"]["beta"],
        dtype=np.float64,
    )
    score_arrays: dict[str, dict[str, np.ndarray]] = {}
    methods: dict[str, dict[str, Any]] = {}
    for variant, name in (
        ("mc10000", f"Fast-DetectGPT-{args.expected_fast_samples}"),
        ("analytic", "Fast-DetectGPT-analytic"),
    ):
        arrays = {
            "human": fast_scores(test_caches["human"], variant, args.score_epsilon),
            "machine": fast_scores(
                test_caches["machine"], variant, args.score_epsilon
            ),
            "calibration": fast_scores(
                dev_caches["calibration"], variant, args.score_epsilon
            ),
        }
        score_arrays[name] = arrays
        methods[name] = score_metrics(
            arrays["human"],
            arrays["machine"],
            arrays["calibration"],
            args.alphas_list,
            args.max_fpr,
        )

    ada_arrays = {
        "human": score_from_features(
            test_caches["human"].ada_u,
            test_caches["human"].ada_v,
            beta,
            args.score_epsilon,
        ),
        "machine": score_from_features(
            test_caches["machine"].ada_u,
            test_caches["machine"].ada_v,
            beta,
            args.score_epsilon,
        ),
        "calibration": score_from_features(
            dev_caches["calibration"].ada_u,
            dev_caches["calibration"].ada_v,
            beta,
            args.score_epsilon,
        ),
    }
    score_arrays["AdaDetectGPT-exact"] = ada_arrays
    methods["AdaDetectGPT-exact"] = score_metrics(
        ada_arrays["human"],
        ada_arrays["machine"],
        ada_arrays["calibration"],
        args.alphas_list,
        args.max_fpr,
    )
    water_final = development["methods"]["Water-Full"]["final_fit"]
    water_spec = WaterSplineSpec(
        **{
            **water_final["basis_spec"],
            "z_base_knots": tuple(water_final["basis_spec"]["z_base_knots"]),
            "h_base_knots": tuple(water_final["basis_spec"]["h_base_knots"]),
        }
    )
    available = {item.name: item for item in water_specs}
    if water_spec.name not in available or available[water_spec.name] != water_spec:
        raise ValueError(
            "independent Water basis does not match the frozen development basis"
        )
    water_theta = np.asarray(water_final["theta"], dtype=np.float64)
    water_arrays = {
        "human": water_scores(
            test_caches["human"].water_u[water_spec.name],
            test_caches["human"].water_v[water_spec.name],
            water_theta,
            args.score_epsilon,
        ),
        "machine": water_scores(
            test_caches["machine"].water_u[water_spec.name],
            test_caches["machine"].water_v[water_spec.name],
            water_theta,
            args.score_epsilon,
        ),
        "calibration": water_scores(
            dev_caches["calibration"].water_u[water_spec.name],
            dev_caches["calibration"].water_v[water_spec.name],
            water_theta,
            args.score_epsilon,
        ),
    }
    score_arrays["Water-Full"] = water_arrays
    methods["Water-Full"] = score_metrics(
        water_arrays["human"],
        water_arrays["machine"],
        water_arrays["calibration"],
        args.alphas_list,
        args.max_fpr,
    )
    cell_dir = args.output_root / model / dataset
    save_scores(cell_dir / "scores.npz", score_arrays)
    result = {
        "model": model,
        "dataset": dataset,
        "protocol": {
            "name": "frozen independent test",
            "test_human_machine_source": str(args.test_cache_root),
            "calibration_source": str(args.dev_cache_root),
            "ada_beta_source": str(
                args.dev_result_root / model / dataset / "results.json"
            ),
            "no_test_fitting_or_selection": True,
            "fast_mc_samples": args.expected_fast_samples,
            "fast_analytic": "exact full-vocabulary conditional moments",
            "ada_conditional_moments": "exact full-vocabulary",
            "water_conditional_moments": "exact full-vocabulary",
            "water_theta_source": str(
                args.dev_result_root / model / dataset / "results.json"
            ),
            "water_basis_frozen_from_development": True,
            "water_candidate_samples_used": 0,
        },
        "methods": methods,
    }
    atomic_write_json(cell_dir / "results.json", result)
    rows = summary_rows(
        model=model,
        dataset=dataset,
        protocol="frozen_independent_test",
        methods=methods,
        alphas=args.alphas_list,
        n_human=test_caches["human"].num_documents,
        n_machine=test_caches["machine"].num_documents,
        n_calibration=dev_caches["calibration"].num_documents,
    )
    return result, rows


def build_parser() -> argparse.ArgumentParser:
    water_root = infer_water_root()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("development", "independent"), required=True
    )
    parser.add_argument(
        "--dev-cache-root",
        type=Path,
        default=water_root / "cache/whitebox/train",
    )
    parser.add_argument(
        "--test-cache-root",
        type=Path,
        default=water_root / "cache/whitebox/test",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=water_root / "data/fits/whitebox",
        help="development result root",
    )
    parser.add_argument(
        "--dev-result-root",
        type=Path,
        default=water_root / "data/fits/whitebox",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=water_root / "results/whitebox/conditional_baselines",
    )
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--expected-pairs", type=int, default=500)
    parser.add_argument("--expected-fast-samples", type=int, default=10_000)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--alphas", default="0.01,0.05")
    parser.add_argument("--max-fpr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--spline-start", type=float, default=-32.0)
    parser.add_argument("--spline-end", type=float, default=0.0)
    parser.add_argument("--n-bases", type=int, default=7)
    parser.add_argument("--spline-degree", type=int, default=2)
    parser.add_argument("--no-intercept", action="store_true")
    parser.add_argument(
        "--solver",
        choices=("official-inv-sqrt", "rayleigh-solve"),
        default="official-inv-sqrt",
    )
    parser.add_argument("--eigenvalue-floor", type=float, default=1e-12)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--water-structures", default="4x4,6x6,8x6")
    parser.add_argument(
        "--water-smooth-lambdas", default="0,1e-3,1e-2,1e-1,1"
    )
    parser.add_argument(
        "--water-ridge-lambdas", default="1e-5,1e-4,1e-3,1e-2"
    )
    parser.add_argument(
        "--water-covariance-shrinkages", default="0,0.05,0.2"
    )
    parser.add_argument(
        "--water-selection-metric", choices=("auc", "pauc"), default="auc"
    )
    parser.add_argument("--score-epsilon", type=float, default=1e-8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-unverified-token-stream",
        action="store_true",
        help="Allow evaluation of caches not checked against Water's frozen 128-cache",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.alphas_list = parse_float_csv(args.alphas)
    try:
        args.water_structures_list = parse_structures(args.water_structures)
        args.water_smooth_lambdas_list = parse_float_csv(
            args.water_smooth_lambdas
        )
        args.water_ridge_lambdas_list = parse_float_csv(
            args.water_ridge_lambdas
        )
        args.water_covariance_shrinkages_list = parse_float_csv(
            args.water_covariance_shrinkages
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    if not all(
        (
            args.water_structures_list,
            args.water_smooth_lambdas_list,
            args.water_ridge_lambdas_list,
            args.water_covariance_shrinkages_list,
        )
    ):
        parser.error("Water structure and regularization grids cannot be empty")
    if any(value < 0 for value in args.water_smooth_lambdas_list):
        parser.error("--water-smooth-lambdas must be non-negative")
    if any(value < 0 for value in args.water_ridge_lambdas_list):
        parser.error("--water-ridge-lambdas must be non-negative")
    if any(
        not 0.0 <= value <= 1.0
        for value in args.water_covariance_shrinkages_list
    ):
        parser.error("--water-covariance-shrinkages must be in [0, 1]")
    if args.expected_pairs <= 0:
        parser.error("--expected-pairs must be positive")
    if args.expected_fast_samples < 2:
        parser.error("--expected-fast-samples must be at least two")
    if not 2 <= args.cv_folds <= args.expected_pairs:
        parser.error("--cv-folds must be in [2, expected-pairs]")
    if not 0 < args.max_fpr <= 1:
        parser.error("--max-fpr must be in (0, 1]")
    spec = AdaSplineSpec(
        start=args.spline_start,
        end=args.spline_end,
        n_bases=args.n_bases,
        degree=args.spline_degree,
        intercept=not args.no_intercept,
    )

    summary_path = (
        args.result_root / "summary.csv"
        if args.mode == "development"
        else args.output_root / "summary.csv"
    )
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{summary_path} exists; pass --overwrite to regenerate it"
        )

    rows: list[dict[str, Any]] = []
    for model in parse_csv(args.models):
        for dataset in parse_csv(args.datasets):
            if args.mode == "development":
                caches, water_specs = load_cell(
                    args.dev_cache_root,
                    model,
                    dataset,
                    expected_documents=args.expected_pairs,
                    expected_fast_samples=args.expected_fast_samples,
                    expected_spec=spec,
                    expected_water_structures=args.water_structures_list,
                    require_legacy_verification=(
                        not args.allow_unverified_token_stream
                    ),
                )
                result, cell_rows = evaluate_development_cell(
                    model, dataset, caches, spec, water_specs, args
                )
            else:
                test_caches, test_water_specs = load_cell(
                    args.test_cache_root,
                    model,
                    dataset,
                    expected_documents=args.expected_pairs,
                    expected_fast_samples=args.expected_fast_samples,
                    expected_spec=spec,
                    expected_water_structures=args.water_structures_list,
                    require_legacy_verification=(
                        not args.allow_unverified_token_stream
                    ),
                )
                dev_caches, dev_water_specs = load_cell(
                    args.dev_cache_root,
                    model,
                    dataset,
                    expected_documents=args.expected_pairs,
                    expected_fast_samples=args.expected_fast_samples,
                    expected_spec=spec,
                    expected_water_structures=args.water_structures_list,
                    require_legacy_verification=(
                        not args.allow_unverified_token_stream
                    ),
                )
                if test_water_specs != dev_water_specs:
                    raise ValueError(
                        f"{model}/{dataset}: independent Water basis differs "
                        "from development"
                    )
                result, cell_rows = evaluate_independent_cell(
                    model,
                    dataset,
                    test_caches,
                    dev_caches,
                    dev_water_specs,
                    args,
                )
            rows.extend(cell_rows)
            atomic_write_csv(summary_path, rows)
            print(
                f"[{model}/{dataset}] "
                + ", ".join(
                    f"{name}: AUC={metrics['auc']:.6f}"
                    for name, metrics in result["methods"].items()
                )
            )
    atomic_write_csv(summary_path, rows)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
