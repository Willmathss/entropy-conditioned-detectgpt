#!/usr/bin/env python3
"""Cache-only white-box H/S/C factorial ablation for the linear T_theta score.

This runner is designed for the current full-text independent protocol.  It
uses the exact white-box sufficient-statistic caches produced by
``build_whitebox_reference_cache.py`` and never loads an LLM.  All model and
dataset choices are therefore evaluated on CPU.

The three factorial switches are:

* H -- entropy conditioning.  When disabled, the tensor-product features are
  projected exactly to the probability-only B-spline space by summing over
  the entropy basis (partition of unity).
* S -- smoothness selection.  When disabled, the smoothness coefficient is
  fixed to zero.
* C -- covariance-shrinkage selection.  When disabled, shrinkage is fixed to
  zero.

Every non-Full variant selects its allowed hyperparameters by paired K-fold
cross-validation on the train split only, refits on all train pairs, freezes
the direction, and evaluates

    T_theta(X) = theta.T @ U_X

on the independent human/machine test passages.  Human calibration passages
come from the disjoint development calibration cache and are scored only after
the direction is frozen.  By default, the Full cell uses the exact frozen
direction/configuration from the current white-box main result so that the
factorial Full row reproduces the paper's primary method.  Pass
``--full-mode recompute`` for a fully reselected Full cell instead.

The script is resume-safe: each completed cell is bound to cache metadata,
file sizes/timestamps, grids, and (for a locked Full cell) the SHA-256 of the
frozen development result.  It intentionally does not bind checkpoints to the
script file hash, so harmless script copies do not invalidate completed work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import (
    auc as sklearn_auc,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold

from build_whitebox_reference_cache import (
    AdaSplineSpec,
    ReferenceSplitCache,
    WaterSplineSpec,
    parse_csv,
    parse_structures,
)


VERSION = "2026.09.09-v1-whitebox-hsc-factorial-ltheta"
DEFAULT_MODELS = ("falcon-7b", "llama-2-7b", "opt-2.7b", "opt-6.7b")
DEFAULT_DATASETS = ("c4", "pubmed", "squad", "writing", "xsum")
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


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    entropy_conditioning: bool
    smoothness: bool
    covariance_shrinkage: bool


VARIANTS = (
    Variant("b0", "B0", False, False, False),
    Variant("h", "B0+H", True, False, False),
    Variant("s", "B0+S", False, True, False),
    Variant("c", "B0+C", False, False, True),
    Variant("hs", "B0+H+S", True, True, False),
    Variant("hc", "B0+H+C", True, False, True),
    Variant("sc", "B0+S+C", False, True, True),
    Variant("full", "Full", True, True, True),
)
VARIANT_BY_KEY = {variant.key: variant for variant in VARIANTS}


@dataclass(frozen=True)
class Config:
    source_j_bases: int
    source_k_bases: int
    j_bases: int
    k_bases: int
    smooth_lambda: float
    ridge_lambda: float
    covariance_shrinkage: float

    @property
    def dimension(self) -> int:
        return self.j_bases * self.k_bases

    @property
    def source_name(self) -> str:
        return f"J{self.source_j_bases}_K{self.source_k_bases}"


@dataclass
class Features:
    human_u: np.ndarray
    human_v: np.ndarray
    machine_u: np.ndarray
    machine_v: np.ndarray
    calibration_u: np.ndarray | None = None
    calibration_v: np.ndarray | None = None


@dataclass
class FitStats:
    delta: np.ndarray
    covariance: np.ndarray
    mean_v: np.ndarray


@dataclass
class FitResult:
    theta: np.ndarray
    condition_number: float


def parse_float_csv(value: str) -> list[float]:
    return [float(item) for item in parse_csv(value)]


def infer_project_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_identity(path: Path, *, hash_content: bool) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    result: dict[str, Any] = {
        "name": path.name,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if hash_content:
        result["sha256"] = sha256_file(path)
    return result


def cell_identity(root: Path, model: str, dataset: str) -> dict[str, Any]:
    cell = root / model / dataset
    return {
        "root": str(root.resolve()),
        "metadata": file_identity(cell / "metadata.json", hash_content=True),
        "human": file_identity(cell / "human_reference.npz", hash_content=False),
        "machine": file_identity(cell / "machine_reference.npz", hash_content=False),
        "calibration": file_identity(
            cell / "calibration_reference.npz", hash_content=False
        ),
    }


def discover_model_root(
    base: Path, model: str, dataset: str, *, description: str
) -> Path:
    direct = base / model / dataset / "metadata.json"
    if direct.is_file():
        return base
    if not base.exists():
        raise FileNotFoundError(f"{description} does not exist: {base}")
    matches = sorted(
        path
        for path in base.rglob("metadata.json")
        if path.parent.name == dataset and path.parent.parent.name == model
    )
    roots = sorted({path.parent.parent.parent.resolve() for path in matches})
    if len(roots) == 1:
        return roots[0]
    if not roots:
        raise FileNotFoundError(
            f"cannot find {model}/{dataset}/metadata.json under {base} "
            f"for {description}"
        )
    raise ValueError(
        f"ambiguous {description}: found multiple cache roots under {base}: "
        + ", ".join(str(root) for root in roots)
    )


def resolve_cache_root(
    args: argparse.Namespace, model: str, dataset: str, *, split: str
) -> Path:
    if split not in {"train", "test"}:
        raise ValueError(split)
    if split == "train":
        if model == "llama-2-7b" and args.llama_dev_cache_root is not None:
            return discover_model_root(
                args.llama_dev_cache_root,
                model,
                dataset,
                description="corrected LLaMA train cache",
            )
        return discover_model_root(
            args.dev_cache_root, model, dataset, description="train cache"
        )

    if model != "llama-2-7b":
        return discover_model_root(
            args.test_cache_root, model, dataset, description="independent test cache"
        )
    if args.llama_test_cache_root is not None:
        return discover_model_root(
            args.llama_test_cache_root,
            model,
            dataset,
            description="corrected LLaMA independent test cache",
        )
    fixed_search = args.project_root / "cache/whitebox/test"
    try:
        return discover_model_root(
            fixed_search,
            model,
            dataset,
            description="auto-discovered corrected LLaMA independent cache",
        )
    except FileNotFoundError:
        if args.allow_base_llama_test_cache:
            return discover_model_root(
                args.test_cache_root,
                model,
                dataset,
                description="explicitly allowed base LLaMA test cache",
            )
        raise FileNotFoundError(
            "corrected LLaMA independent cache was not found. Pass "
            "--llama-test-cache-root PATH. The runner refuses to silently "
            "fall back to the pre-fix LLaMA cache; use "
            "--allow-base-llama-test-cache only after manually verifying it."
        )


def normalize_spec(spec: WaterSplineSpec) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(spec)))


def load_cell(
    root: Path,
    model: str,
    dataset: str,
    *,
    expected_pairs: int,
    expected_ada_spec: AdaSplineSpec,
    expected_structures: Sequence[tuple[int, int]],
    require_token_verification: bool,
) -> tuple[dict[str, ReferenceSplitCache], tuple[WaterSplineSpec, ...]]:
    cell = root / model / dataset
    metadata = load_json(cell / "metadata.json")
    if int(metadata.get("processed_pairs", -1)) != expected_pairs:
        raise ValueError(
            f"{cell}: expected {expected_pairs} pairs, metadata reports "
            f"{metadata.get('processed_pairs')}"
        )
    if not metadata.get("source_model_equals_sampling_model", False):
        raise ValueError(f"{cell}: source model is not the sampling model")
    if not metadata.get("source_model_equals_scoring_model", False):
        raise ValueError(f"{cell}: source model is not the scoring model")
    verified = metadata.get(
        "legacy_offsets_verified", metadata.get("legacy_cache_verified", False)
    )
    if require_token_verification and not verified:
        raise ValueError(
            f"{cell}: token offsets are not marked verified; pass "
            "--allow-unverified-token-stream only after checking provenance"
        )
    stored_ada = AdaSplineSpec(**metadata["ada_spline"])
    if stored_ada != expected_ada_spec:
        raise ValueError(f"{cell}: Ada spline metadata changed")
    water_meta = metadata.get("water_full", {})
    if not water_meta.get("enabled", False):
        raise ValueError(f"{cell}: entropy-conditioned features are absent")
    specs = tuple(
        WaterSplineSpec(
            **{
                **item,
                "z_base_knots": tuple(item["z_base_knots"]),
                "h_base_knots": tuple(item["h_base_knots"]),
            }
        )
        for item in water_meta["specs"]
    )
    structures = [(spec.j_bases, spec.k_bases) for spec in specs]
    if list(expected_structures) != structures:
        raise ValueError(
            f"{cell}: structures {structures} do not match requested "
            f"{list(expected_structures)}"
        )
    caches = {
        split: ReferenceSplitCache.load(cell / f"{split}_reference.npz")
        for split in ("human", "machine", "calibration")
    }
    for split, cache in caches.items():
        if cache.num_documents != expected_pairs:
            raise ValueError(
                f"{cell}/{split}: expected {expected_pairs}, found "
                f"{cache.num_documents}"
            )
        for spec in specs:
            if np.asarray(cache.water_u.get(spec.name, np.empty(0))).shape != (
                expected_pairs,
                spec.dimension,
            ):
                raise ValueError(f"{cell}/{split}: invalid U for {spec.name}")
            if np.asarray(cache.water_v.get(spec.name, np.empty(0))).shape != (
                expected_pairs,
                spec.dimension,
                spec.dimension,
            ):
                raise ValueError(f"{cell}/{split}: invalid V for {spec.name}")
    return caches, specs


def second_difference_matrix(size: int) -> np.ndarray:
    if size < 3:
        return np.zeros((0, size), dtype=np.float64)
    result = np.zeros((size - 2, size), dtype=np.float64)
    rows = np.arange(size - 2)
    result[rows, rows] = 1.0
    result[rows, rows + 1] = -2.0
    result[rows, rows + 2] = 1.0
    return result


def smoothness_matrix(j_bases: int, k_bases: int) -> np.ndarray:
    dz = second_difference_matrix(j_bases)
    dh = second_difference_matrix(k_bases)
    dimension = j_bases * k_bases
    penalty = np.zeros((dimension, dimension), dtype=np.float64)
    if len(dz):
        penalty += np.kron(dz.T @ dz, np.eye(k_bases)) / float(
            (j_bases - 2) * k_bases
        )
    if len(dh):
        penalty += np.kron(np.eye(j_bases), dh.T @ dh) / float(
            j_bases * (k_bases - 2)
        )
    return 0.5 * (penalty + penalty.T)


def within_class_covariance(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("at least two rows are needed for covariance")
    centered = values - values.mean(axis=0, keepdims=True)
    return centered.T @ centered / float(len(values) - 1)


def project_entropy_independent(
    u: np.ndarray, v: np.ndarray, j_bases: int, k_bases: int
) -> tuple[np.ndarray, np.ndarray]:
    """Use sum_k C_k(u)=1 to project B(z) x C(u) to B(z)."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    n = u.shape[0]
    reduced_u = u.reshape(n, j_bases, k_bases).sum(axis=2)
    reduced_v = v.reshape(
        n, j_bases, k_bases, j_bases, k_bases
    ).sum(axis=(2, 4))
    return reduced_u, 0.5 * (reduced_v + reduced_v.transpose(0, 2, 1))


def split_arrays(
    caches: dict[str, ReferenceSplitCache],
    spec: WaterSplineSpec,
    *,
    entropy_conditioning: bool,
    include_calibration: bool,
) -> Features:
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    roles = ["human", "machine"]
    if include_calibration:
        roles.append("calibration")
    for role in roles:
        cache = caches[role]
        u = np.asarray(cache.water_u[spec.name], dtype=np.float64)
        v = np.asarray(cache.water_v[spec.name], dtype=np.float64)
        arrays[role] = (
            (u, v)
            if entropy_conditioning
            else project_entropy_independent(
                u, v, spec.j_bases, spec.k_bases
            )
        )
    return Features(
        human_u=arrays["human"][0],
        human_v=arrays["human"][1],
        machine_u=arrays["machine"][0],
        machine_v=arrays["machine"][1],
        calibration_u=(arrays["calibration"][0] if include_calibration else None),
        calibration_v=(arrays["calibration"][1] if include_calibration else None),
    )


def make_stats(features: Features, indices: np.ndarray) -> FitStats:
    human = features.human_u[indices]
    machine = features.machine_u[indices]
    delta = machine.mean(axis=0) - human.mean(axis=0)
    covariance = within_class_covariance(human) + within_class_covariance(machine)
    mean_v = np.mean(
        np.concatenate(
            [features.human_v[indices], features.machine_v[indices]], axis=0
        ),
        axis=0,
    )
    return FitStats(
        delta=np.asarray(delta, dtype=np.float64),
        covariance=np.asarray(covariance, dtype=np.float64),
        mean_v=np.asarray(mean_v, dtype=np.float64),
    )


def fit_from_stats(stats: FitStats, config: Config, epsilon: float) -> FitResult:
    shrinkage = config.covariance_shrinkage
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("covariance shrinkage must lie in [0, 1]")
    covariance = (
        (1.0 - shrinkage) * stats.covariance
        + shrinkage * np.diag(np.diag(stats.covariance))
    )
    covariance = 0.5 * (covariance + covariance.T)
    dimension = config.dimension
    scale = float(np.trace(covariance) / dimension)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    penalty = smoothness_matrix(config.j_bases, config.k_bases)
    positive = np.linalg.eigvalsh(penalty)
    positive = positive[positive > 1e-12]
    penalty_scale = float(positive.mean()) if len(positive) else 1.0
    system = (
        covariance
        + scale * config.smooth_lambda * (penalty / penalty_scale)
        + scale * config.ridge_lambda * np.eye(dimension)
    )
    system = 0.5 * (system + system.T)
    eigenvalues, eigenvectors = np.linalg.eigh(system)
    floor = max(
        epsilon * scale,
        np.finfo(np.float64).eps * max(float(eigenvalues[-1]), 1.0),
    )
    safe = np.maximum(eigenvalues, floor)
    theta = eigenvectors @ ((eigenvectors.T @ stats.delta) / safe)
    norm2 = float(theta @ stats.mean_v @ theta)
    if np.isfinite(norm2) and norm2 > epsilon:
        theta = theta / math.sqrt(norm2)
    else:
        norm = float(np.linalg.norm(theta))
        if not np.isfinite(norm) or norm <= epsilon:
            raise FloatingPointError("Rayleigh solver returned a zero direction")
        theta = theta / norm
    if float(theta @ stats.delta) < 0:
        theta = -theta
    return FitResult(
        theta=np.asarray(theta, dtype=np.float64),
        condition_number=float(safe[-1] / safe[0]),
    )


def linear_scores(u: np.ndarray, theta: np.ndarray) -> np.ndarray:
    scores = np.asarray(u, dtype=np.float64) @ np.asarray(theta, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise FloatingPointError("non-finite linear scores")
    return scores


def candidate_configs(
    variant: Variant,
    specs: Sequence[WaterSplineSpec],
    smooth_lambdas: Sequence[float],
    ridge_lambdas: Sequence[float],
    shrinkages: Sequence[float],
) -> list[Config]:
    source_specs = list(specs)
    if not variant.entropy_conditioning:
        first_for_j: dict[int, WaterSplineSpec] = {}
        for spec in source_specs:
            first_for_j.setdefault(spec.j_bases, spec)
        source_specs = list(first_for_j.values())
    smooth_grid = smooth_lambdas if variant.smoothness else [0.0]
    shrink_grid = shrinkages if variant.covariance_shrinkage else [0.0]
    configs = []
    for spec, smooth, ridge, shrink in itertools.product(
        source_specs, smooth_grid, ridge_lambdas, shrink_grid
    ):
        configs.append(
            Config(
                source_j_bases=spec.j_bases,
                source_k_bases=spec.k_bases,
                j_bases=spec.j_bases,
                k_bases=(spec.k_bases if variant.entropy_conditioning else 1),
                smooth_lambda=float(smooth),
                ridge_lambda=float(ridge),
                covariance_shrinkage=float(shrink),
            )
        )
    if not configs:
        raise ValueError(f"variant {variant.key}: empty configuration grid")
    return configs


def labels_scores(human: np.ndarray, machine: np.ndarray):
    return (
        np.concatenate(
            [np.zeros(len(human), dtype=np.int8), np.ones(len(machine), dtype=np.int8)]
        ),
        np.concatenate([human, machine]),
    )


def full_auc(human: np.ndarray, machine: np.ndarray) -> float:
    labels, scores = labels_scores(human, machine)
    return float(roc_auc_score(labels, scores))


def normalized_partial_auc(
    human: np.ndarray, machine: np.ndarray, max_fpr: float
) -> float:
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
    ordered = np.sort(np.asarray(calibration, dtype=np.float64))
    scores = np.asarray(scores, dtype=np.float64)
    counts_ge = len(ordered) - np.searchsorted(ordered, scores, side="left")
    return (1.0 + counts_ge) / float(len(ordered) + 1)


def metrics_from_scores(
    human: np.ndarray, machine: np.ndarray, calibration: np.ndarray
) -> dict[str, float]:
    labels, scores = labels_scores(human, machine)
    precision, recall, _ = precision_recall_curve(labels, scores)
    human_p = conformal_pvalues(calibration, human)
    machine_p = conformal_pvalues(calibration, machine)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(sklearn_auc(recall, precision)),
        "pauc_001": normalized_partial_auc(human, machine, 0.01),
        "pauc_005": normalized_partial_auc(human, machine, 0.05),
        "tpr_at_fpr_001": tpr_at_fpr(human, machine, 0.01),
        "tpr_at_fpr_005": tpr_at_fpr(human, machine, 0.05),
        "conformal_fpr_001": float(np.mean(human_p <= 0.01)),
        "conformal_power_001": float(np.mean(machine_p <= 0.01)),
        "conformal_fpr_005": float(np.mean(human_p <= 0.05)),
        "conformal_power_005": float(np.mean(machine_p <= 0.05)),
    }


def select_variant(
    variant: Variant,
    configs: Sequence[Config],
    train_features: dict[str, Features],
    *,
    n_pairs: int,
    cv_folds: int,
    seed: int,
    epsilon: float,
) -> tuple[Config, list[dict[str, Any]]]:
    folds = list(
        KFold(n_splits=cv_folds, shuffle=True, random_state=seed).split(
            np.arange(n_pairs, dtype=np.int64)
        )
    )
    stats: dict[tuple[str, int], FitStats] = {}
    for source_name, features in train_features.items():
        for fold_index, (fit_indices, _) in enumerate(folds):
            stats[(source_name, fold_index)] = make_stats(features, fit_indices)

    records: list[dict[str, Any]] = []
    for config in configs:
        features = train_features[config.source_name]
        oof_human = np.empty(n_pairs, dtype=np.float64)
        oof_machine = np.empty(n_pairs, dtype=np.float64)
        fold_records = []
        for fold_index, (_, valid_indices) in enumerate(folds):
            fit = fit_from_stats(
                stats[(config.source_name, fold_index)], config, epsilon
            )
            human = linear_scores(features.human_u[valid_indices], fit.theta)
            machine = linear_scores(features.machine_u[valid_indices], fit.theta)
            oof_human[valid_indices] = human
            oof_machine[valid_indices] = machine
            fold_records.append(
                {
                    "fold": fold_index + 1,
                    "validation_size": int(len(valid_indices)),
                    "auroc": full_auc(human, machine),
                    "condition_number": fit.condition_number,
                }
            )
        records.append(
            {
                **asdict(config),
                "dimension": config.dimension,
                "cv_auroc": full_auc(oof_human, oof_machine),
                "cv_pauc_005": normalized_partial_auc(
                    oof_human, oof_machine, 0.05
                ),
                "mean_condition_number": float(
                    np.mean([item["condition_number"] for item in fold_records])
                ),
                "folds": fold_records,
            }
        )
    records.sort(
        key=lambda item: (
            item["cv_auroc"],
            item["cv_pauc_005"],
            -item["dimension"],
            -item["smooth_lambda"],
            -item["ridge_lambda"],
            -item["covariance_shrinkage"],
        ),
        reverse=True,
    )
    winner = records[0]
    selected = Config(
        **{
            key: winner[key]
            for key in (
                "source_j_bases",
                "source_k_bases",
                "j_bases",
                "k_bases",
                "smooth_lambda",
                "ridge_lambda",
                "covariance_shrinkage",
            )
        }
    )
    return selected, records


def locate_full_record(value: dict[str, Any]) -> dict[str, Any]:
    methods = value.get("methods", {})
    for key in ("Water-Full", "Ours", "Tensor-Rayleigh", "TensorRayleigh"):
        record = methods.get(key)
        if isinstance(record, dict) and isinstance(record.get("final_fit"), dict):
            return record["final_fit"]
    if isinstance(value.get("final_fit"), dict):
        return value["final_fit"]
    raise ValueError("cannot find frozen Full final_fit in development result")


def locked_full(
    path: Path,
    specs: Sequence[WaterSplineSpec],
    train_features: dict[str, Features],
    epsilon: float,
) -> tuple[Config, np.ndarray, dict[str, Any]]:
    value = load_json(path)
    final = locate_full_record(value)
    config_value = final.get("config", final.get("selected_config"))
    basis_value = final.get("basis_spec", final.get("selected_basis_spec"))
    if not isinstance(config_value, dict) or not isinstance(basis_value, dict):
        raise ValueError(f"{path}: Full fit lacks config or basis_spec")
    j_bases = int(config_value.get("j_bases", basis_value["j_bases"]))
    k_bases = int(config_value.get("k_bases", basis_value["k_bases"]))
    selected_specs = [
        spec for spec in specs if (spec.j_bases, spec.k_bases) == (j_bases, k_bases)
    ]
    if len(selected_specs) != 1:
        raise ValueError(f"{path}: frozen basis J{j_bases}_K{k_bases} unavailable")
    selected_spec = selected_specs[0]
    for key, expected in normalize_spec(selected_spec).items():
        if key in basis_value and basis_value[key] != expected:
            raise ValueError(f"{path}: frozen basis field {key} changed")
    config = Config(
        source_j_bases=j_bases,
        source_k_bases=k_bases,
        j_bases=j_bases,
        k_bases=k_bases,
        smooth_lambda=float(config_value["smooth_lambda"]),
        ridge_lambda=float(config_value["ridge_lambda"]),
        covariance_shrinkage=float(config_value["covariance_shrinkage"]),
    )
    theta = np.asarray(final["theta"], dtype=np.float64).reshape(-1)
    if theta.shape != (config.dimension,) or not np.isfinite(theta).all():
        raise ValueError(f"{path}: invalid frozen theta shape {theta.shape}")
    features = train_features[config.source_name]
    delta = features.machine_u.mean(axis=0) - features.human_u.mean(axis=0)
    if float(theta @ delta) <= 0:
        raise ValueError(f"{path}: frozen theta has the wrong train orientation")
    if float(np.linalg.norm(theta)) <= epsilon:
        raise ValueError(f"{path}: frozen theta is zero")
    return config, theta, {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
    }


def run_cell(
    args: argparse.Namespace, model: str, dataset: str
) -> dict[str, Any]:
    dev_root = resolve_cache_root(args, model, dataset, split="train")
    test_root = resolve_cache_root(args, model, dataset, split="test")
    main_result = args.main_result_root / model / dataset / "results.json"
    signature_payload: dict[str, Any] = {
        "version": VERSION,
        "model": model,
        "dataset": dataset,
        "score": "T_theta=theta^T U_X",
        "full_mode": args.full_mode,
        "expected_pairs": args.expected_pairs,
        "cv_folds": args.cv_folds,
        "seed": args.seed,
        "structures": args.structures_list,
        "smooth_lambdas": args.smooth_lambdas_list,
        "ridge_lambdas": args.ridge_lambdas_list,
        "covariance_shrinkages": args.covariance_shrinkages_list,
        "score_epsilon": args.score_epsilon,
        "dev_cache": cell_identity(dev_root, model, dataset),
        "test_cache": cell_identity(test_root, model, dataset),
    }
    if args.full_mode == "locked":
        signature_payload["main_result"] = {
            "path": str(main_result.resolve()),
            "sha256": sha256_file(main_result),
        }
    signature = canonical_digest(signature_payload)
    cell_dir = args.output_root / model / dataset
    result_path = cell_dir / "results.json"
    if result_path.is_file() and not args.overwrite:
        existing = load_json(result_path)
        if existing.get("run_signature") != signature:
            raise ValueError(
                f"{result_path}: existing checkpoint belongs to a different "
                "cache/configuration; use a new output root or pass --overwrite"
            )
        if existing.get("status") != "complete":
            raise ValueError(f"{result_path}: incomplete checkpoint")
        print(f"[{model}/{dataset}] resume: complete")
        return existing

    ada_spec = AdaSplineSpec(
        start=args.spline_start,
        end=args.spline_end,
        n_bases=args.n_bases,
        degree=args.spline_degree,
        intercept=not args.no_intercept,
    )
    dev_caches, dev_specs = load_cell(
        dev_root,
        model,
        dataset,
        expected_pairs=args.expected_pairs,
        expected_ada_spec=ada_spec,
        expected_structures=args.structures_list,
        require_token_verification=not args.allow_unverified_token_stream,
    )
    test_caches, test_specs = load_cell(
        test_root,
        model,
        dataset,
        expected_pairs=args.expected_pairs,
        expected_ada_spec=ada_spec,
        expected_structures=args.structures_list,
        require_token_verification=not args.allow_unverified_token_stream,
    )
    if [normalize_spec(spec) for spec in dev_specs] != [
        normalize_spec(spec) for spec in test_specs
    ]:
        raise ValueError(f"{model}/{dataset}: train/test tensor bases differ")
    spec_by_name = {spec.name: spec for spec in dev_specs}
    feature_cache: dict[tuple[str, bool, str], Features] = {}

    def features_for(
        spec: WaterSplineSpec, entropy: bool, split: str
    ) -> Features:
        key = (spec.name, entropy, split)
        if key not in feature_cache:
            feature_cache[key] = split_arrays(
                dev_caches if split == "train" else test_caches,
                spec,
                entropy_conditioning=entropy,
                include_calibration=(split == "train"),
            )
        return feature_cache[key]

    variants: dict[str, Any] = {}
    score_arrays: dict[str, np.ndarray] = {}
    for variant in VARIANTS:
        train_by_source = {
            spec.name: features_for(spec, variant.entropy_conditioning, "train")
            for spec in dev_specs
        }
        search: list[dict[str, Any]] | None
        provenance: dict[str, Any]
        if variant.key == "full" and args.full_mode == "locked":
            selected, theta, provenance = locked_full(
                main_result, dev_specs, train_by_source, args.score_epsilon
            )
            search = None
        else:
            configs = candidate_configs(
                variant,
                dev_specs,
                args.smooth_lambdas_list,
                args.ridge_lambdas_list,
                args.covariance_shrinkages_list,
            )
            selected, search = select_variant(
                variant,
                configs,
                train_by_source,
                n_pairs=args.expected_pairs,
                cv_folds=args.cv_folds,
                seed=args.seed,
                epsilon=args.score_epsilon,
            )
            final_features = train_by_source[selected.source_name]
            final_fit = fit_from_stats(
                make_stats(
                    final_features,
                    np.arange(args.expected_pairs, dtype=np.int64),
                ),
                selected,
                args.score_epsilon,
            )
            theta = final_fit.theta
            provenance = {
                "mode": "train-only-paired-cv-refit",
                "condition_number": final_fit.condition_number,
            }

        source_spec = spec_by_name[selected.source_name]
        train_features = features_for(
            source_spec, variant.entropy_conditioning, "train"
        )
        test_features = features_for(
            source_spec, variant.entropy_conditioning, "test"
        )
        human = linear_scores(test_features.human_u, theta)
        machine = linear_scores(test_features.machine_u, theta)
        if train_features.calibration_u is None:
            raise AssertionError("train calibration U is missing")
        calibration = linear_scores(train_features.calibration_u, theta)
        metrics = metrics_from_scores(human, machine, calibration)
        variants[variant.key] = {
            "label": variant.label,
            "H": int(variant.entropy_conditioning),
            "S": int(variant.smoothness),
            "C": int(variant.covariance_shrinkage),
            "selected_config": asdict(selected),
            "effective_dimension": selected.dimension,
            "theta": theta.tolist(),
            "selection": {
                "locked_full": bool(
                    variant.key == "full" and args.full_mode == "locked"
                ),
                "provenance": provenance,
                "candidates": search,
            },
            "metrics": metrics,
        }
        score_arrays[f"{variant.key}__human"] = human
        score_arrays[f"{variant.key}__machine"] = machine
        score_arrays[f"{variant.key}__calibration"] = calibration
        score_arrays[f"{variant.key}__theta"] = theta
        print(
            f"[{model}/{dataset}] {variant.key}: "
            f"AUC={metrics['auroc']:.6f}, config={selected.source_name}/"
            f"d{selected.dimension}/s{selected.smooth_lambda:g}/"
            f"r{selected.ridge_lambda:g}/c{selected.covariance_shrinkage:g}"
        )

    result = {
        "builder_version": VERSION,
        "status": "complete",
        "model": model,
        "dataset": dataset,
        "run_signature": signature,
        "signature_payload": signature_payload,
        "protocol": {
            "access": "white-box source=sampling=scoring",
            "text_scope": "full text",
            "training": "paired K-fold train-only selection and full-train refit",
            "test": "independent frozen human/machine passages",
            "calibration": "disjoint train-side human calibration passages",
            "primary_score": "T_theta(X)=theta^T U_X",
            "test_used_for_selection_or_orientation": False,
            "full_mode": args.full_mode,
        },
        "n_train_human": args.expected_pairs,
        "n_train_machine": args.expected_pairs,
        "n_test_human": args.expected_pairs,
        "n_test_machine": args.expected_pairs,
        "n_calibration": args.expected_pairs,
        "variants": variants,
    }
    atomic_npz(cell_dir / "scores.npz", **score_arrays)
    atomic_json(result_path, result)
    return result


def result_rows(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        for key, record in result["variants"].items():
            selected = record["selected_config"]
            row: dict[str, Any] = {
                "source_model": result["model"],
                "dataset": result["dataset"],
                "variant": key,
                "label": record["label"],
                "H": record["H"],
                "S": record["S"],
                "C": record["C"],
                "source_structure": (
                    f"{selected['source_j_bases']}x{selected['source_k_bases']}"
                ),
                "effective_dimension": record["effective_dimension"],
                "smooth_lambda": selected["smooth_lambda"],
                "ridge_lambda": selected["ridge_lambda"],
                "covariance_shrinkage": selected["covariance_shrinkage"],
                "n_human": result["n_test_human"],
                "n_machine": result["n_test_machine"],
                "n_calibration": result["n_calibration"],
            }
            row.update(record["metrics"])
            rows.append(row)
    rows.sort(key=lambda item: (item["source_model"], item["dataset"], item["variant"]))
    return rows


def macro_rows(cell_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for variant in VARIANTS:
        rows = [row for row in cell_rows if row["variant"] == variant.key]
        if not rows:
            continue
        record: dict[str, Any] = {
            "variant": variant.key,
            "label": variant.label,
            "H": int(variant.entropy_conditioning),
            "S": int(variant.smoothness),
            "C": int(variant.covariance_shrinkage),
            "n_cells": len(rows),
        }
        for metric in METRIC_NAMES:
            record[metric] = float(np.mean([float(row[metric]) for row in rows]))
        result.append(record)
    return result


def effect_rows(
    cell_rows: Sequence[dict[str, Any]], *, seed: int, bootstrap_samples: int
) -> list[dict[str, Any]]:
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in cell_rows:
        by_cell.setdefault((row["source_model"], row["dataset"]), []).append(row)
    rng = np.random.default_rng(seed)
    result = []
    for component in ("H", "S", "C"):
        for metric in METRIC_NAMES:
            deltas = []
            for _, rows in sorted(by_cell.items()):
                enabled = [float(row[metric]) for row in rows if int(row[component]) == 1]
                disabled = [float(row[metric]) for row in rows if int(row[component]) == 0]
                if len(enabled) != 4 or len(disabled) != 4:
                    continue
                deltas.append(float(np.mean(enabled) - np.mean(disabled)))
            if not deltas:
                continue
            values = np.asarray(deltas, dtype=np.float64)
            if bootstrap_samples > 0:
                draws = rng.integers(
                    0, len(values), size=(bootstrap_samples, len(values))
                )
                boot = values[draws].mean(axis=1)
                low, high = np.quantile(boot, [0.025, 0.975])
            else:
                low = high = float("nan")
            result.append(
                {
                    "effect_kind": "balanced_factorial_main_effect",
                    "term": component,
                    "metric": metric,
                    "n_cells": len(values),
                    "mean_delta": float(values.mean()),
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "positive_cells": int(np.sum(values > 0)),
                    "negative_cells": int(np.sum(values < 0)),
                    "zero_cells": int(np.sum(values == 0)),
                }
            )
    return result


def collect_results(
    output_root: Path, models: Sequence[str], datasets: Sequence[str]
) -> list[dict[str, Any]]:
    results = []
    for model in models:
        for dataset in datasets:
            path = output_root / model / dataset / "results.json"
            if path.is_file():
                value = load_json(path)
                if value.get("status") == "complete":
                    results.append(value)
    return results


def write_summaries(args: argparse.Namespace) -> None:
    results = collect_results(args.output_root, args.models_list, args.datasets_list)
    if not results:
        raise FileNotFoundError("no completed cell results are available")
    cells = result_rows(results)
    atomic_csv(args.output_root / "cell_summary.csv", cells)
    atomic_csv(args.output_root / "macro_summary.csv", macro_rows(cells))
    effects = effect_rows(
        cells, seed=args.seed + 10_000, bootstrap_samples=args.bootstrap_samples
    )
    if effects:
        atomic_csv(args.output_root / "factorial_effects.csv", effects)
    expected = len(args.models_list) * len(args.datasets_list)
    atomic_json(
        args.output_root / "run_manifest.json",
        {
            "builder_version": VERSION,
            "status": "complete" if len(results) == expected else "partial",
            "completed_cells": len(results),
            "expected_cells": expected,
            "models": args.models_list,
            "datasets": args.datasets_list,
            "primary_score": "T_theta(X)=theta^T U_X",
            "full_mode": args.full_mode,
        },
    )


def self_test() -> None:
    rng = np.random.default_rng(17)
    n, j, k = 40, 4, 3
    dimension = j * k
    human_u = rng.normal(size=(n, dimension))
    signal = np.zeros(dimension)
    signal[[0, 4, 8]] = [0.35, 0.2, 0.3]
    machine_u = rng.normal(size=(n, dimension)) + signal
    base_v = np.eye(dimension)
    human_v = np.repeat(base_v[None, :, :], n, axis=0)
    machine_v = np.repeat(base_v[None, :, :], n, axis=0)
    reduced_u, reduced_v = project_entropy_independent(human_u, human_v, j, k)
    assert reduced_u.shape == (n, j)
    assert reduced_v.shape == (n, j, j)
    features = Features(human_u, human_v, machine_u, machine_v)
    config = Config(j, k, j, k, 0.01, 1e-3, 0.05)
    fit = fit_from_stats(make_stats(features, np.arange(n)), config, 1e-8)
    human = linear_scores(human_u, fit.theta)
    machine = linear_scores(machine_u, fit.theta)
    metrics = metrics_from_scores(human, machine, human.copy())
    assert all(np.isfinite(metrics[name]) for name in METRIC_NAMES)
    assert float(fit.theta @ (machine_u.mean(0) - human_u.mean(0))) > 0
    print("SELF-TEST PASSED: projection, Rayleigh fit, L-theta scoring, metrics")


def build_parser() -> argparse.ArgumentParser:
    project = infer_project_root()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--project-root", type=Path, default=project)
    parser.add_argument(
        "--dev-cache-root",
        type=Path,
        default=project / "cache/whitebox/train",
    )
    parser.add_argument(
        "--test-cache-root",
        type=Path,
        default=project / "cache/whitebox/test",
    )
    parser.add_argument("--llama-dev-cache-root", type=Path)
    parser.add_argument("--llama-test-cache-root", type=Path)
    parser.add_argument(
        "--main-result-root",
        type=Path,
        default=project / "data/fits/whitebox",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project / "results/whitebox/factorial",
    )
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--expected-pairs", type=int, default=500)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--structures", default="4x4,6x6,8x6")
    parser.add_argument("--smooth-lambdas", default="0,1e-3,1e-2,1e-1,1")
    parser.add_argument("--ridge-lambdas", default="1e-5,1e-4,1e-3,1e-2")
    parser.add_argument("--covariance-shrinkages", default="0,0.05,0.2")
    parser.add_argument("--score-epsilon", type=float, default=1e-8)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--full-mode", choices=("locked", "recompute"), default="locked")
    parser.add_argument("--spline-start", type=float, default=-32.0)
    parser.add_argument("--spline-end", type=float, default=0.0)
    parser.add_argument("--n-bases", type=int, default=7)
    parser.add_argument("--spline-degree", type=int, default=2)
    parser.add_argument("--no-intercept", action="store_true")
    parser.add_argument("--allow-unverified-token-stream", action="store_true")
    parser.add_argument("--allow-base-llama-test-cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--rebuild-summary-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    args.project_root = args.project_root.resolve()
    for name in (
        "dev_cache_root",
        "test_cache_root",
        "main_result_root",
        "output_root",
        "llama_dev_cache_root",
        "llama_test_cache_root",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    args.models_list = parse_csv(args.models)
    args.datasets_list = parse_csv(args.datasets)
    args.structures_list = parse_structures(args.structures)
    args.smooth_lambdas_list = parse_float_csv(args.smooth_lambdas)
    args.ridge_lambdas_list = parse_float_csv(args.ridge_lambdas)
    args.covariance_shrinkages_list = parse_float_csv(
        args.covariance_shrinkages
    )
    unknown_models = sorted(set(args.models_list) - set(DEFAULT_MODELS))
    unknown_datasets = sorted(set(args.datasets_list) - set(DEFAULT_DATASETS))
    if unknown_models:
        parser.error(f"unknown models: {unknown_models}")
    if unknown_datasets:
        parser.error(f"unknown datasets: {unknown_datasets}")
    if not args.models_list or not args.datasets_list:
        parser.error("models and datasets cannot be empty")
    if not 2 <= args.cv_folds <= args.expected_pairs:
        parser.error("--cv-folds must be in [2, expected-pairs]")
    if args.expected_pairs < 2:
        parser.error("--expected-pairs must be at least two")
    if args.score_epsilon <= 0:
        parser.error("--score-epsilon must be positive")
    if any(value < 0 for value in args.smooth_lambdas_list):
        parser.error("smoothness values must be non-negative")
    if any(value <= 0 for value in args.ridge_lambdas_list):
        parser.error("ridge values must be positive")
    if any(not 0 <= value <= 1 for value in args.covariance_shrinkages_list):
        parser.error("shrinkage values must be in [0, 1]")
    if args.rebuild_summary_only:
        write_summaries(args)
        print(f"Rebuilt summaries in {args.output_root}")
        return 0

    if args.preflight_only:
        ada_spec = AdaSplineSpec(
            start=args.spline_start,
            end=args.spline_end,
            n_bases=args.n_bases,
            degree=args.spline_degree,
            intercept=not args.no_intercept,
        )
        for model in args.models_list:
            for dataset in args.datasets_list:
                dev_root = resolve_cache_root(args, model, dataset, split="train")
                test_root = resolve_cache_root(args, model, dataset, split="test")
                dev, dev_specs = load_cell(
                    dev_root,
                    model,
                    dataset,
                    expected_pairs=args.expected_pairs,
                    expected_ada_spec=ada_spec,
                    expected_structures=args.structures_list,
                    require_token_verification=not args.allow_unverified_token_stream,
                )
                test, test_specs = load_cell(
                    test_root,
                    model,
                    dataset,
                    expected_pairs=args.expected_pairs,
                    expected_ada_spec=ada_spec,
                    expected_structures=args.structures_list,
                    require_token_verification=not args.allow_unverified_token_stream,
                )
                if [normalize_spec(spec) for spec in dev_specs] != [
                    normalize_spec(spec) for spec in test_specs
                ]:
                    raise ValueError(f"{model}/{dataset}: train/test bases differ")
                if args.full_mode == "locked":
                    result = args.main_result_root / model / dataset / "results.json"
                    train_by_source = {
                        spec.name: split_arrays(
                            dev,
                            spec,
                            entropy_conditioning=True,
                            include_calibration=True,
                        )
                        for spec in dev_specs
                    }
                    locked_full(result, dev_specs, train_by_source, args.score_epsilon)
                del dev, test
                print(
                    f"PREFLIGHT OK {model}/{dataset}: train={dev_root}, "
                    f"test={test_root}"
                )
        print("INPUT CHECK PASSED")
        return 0

    results = []
    for model in args.models_list:
        for dataset in args.datasets_list:
            results.append(run_cell(args, model, dataset))
            write_summaries(args)
    write_summaries(args)
    print(
        f"COMPLETE: {len(results)} cells; summaries in {args.output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
