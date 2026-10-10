#!/usr/bin/env python3
"""Matched white-box deployment-score ablation: frozen theta, L_theta vs S_theta.

The development-selected Water basis, hyperparameters, and final theta are
loaded verbatim.  Independent human/machine U/V features and development human
calibration U/V features are read from the existing reference caches.  No text
generation, LM forward pass, fitting, hyperparameter selection, or test-set
orientation is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import auc as sklearn_auc
from sklearn.metrics import precision_recall_curve


VERSION = "2026.08.25-v1-whitebox-ltheta-vs-stheta"
DEFAULT_MODELS = ("llama-2-7b", "opt-2.7b", "opt-6.7b", "falcon-7b")
DEFAULT_DATASETS = ("c4", "squad", "xsum", "writing", "pubmed")


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_float_csv(value: str) -> list[float]:
    return [float(item) for item in parse_csv(value)]


def import_file(path: Path, module_name: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


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
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
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


def digest_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def score_both(
    u: np.ndarray,
    v: np.ndarray,
    theta: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u_array = np.asarray(u, dtype=np.float64)
    v_array = np.asarray(v, dtype=np.float64)
    flat = np.asarray(theta, dtype=np.float64).reshape(-1)
    if u_array.ndim != 2 or u_array.shape[1] != len(flat):
        raise ValueError(
            f"U/theta shape mismatch: U={u_array.shape}, theta={flat.shape}"
        )
    if v_array.shape != (len(u_array), len(flat), len(flat)):
        raise ValueError(
            f"V shape mismatch: got {v_array.shape}, expected "
            f"{(len(u_array), len(flat), len(flat))}"
        )
    ltheta = u_array @ flat
    denominator2 = np.einsum(
        "d,nde,e->n", flat, v_array, flat, optimize=True
    )
    denominator = np.sqrt(np.clip(denominator2, epsilon, None))
    stheta = ltheta / denominator
    if not all(np.isfinite(item).all() for item in (ltheta, stheta, denominator)):
        raise FloatingPointError("non-finite L_theta/S_theta score")
    return ltheta, stheta, denominator


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
        raise ValueError("correlation inputs must be equal-length vectors")
    if float(left.std()) == 0.0 or float(right.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def diagnostics(
    ltheta: np.ndarray,
    stheta: np.ndarray,
    denominator: np.ndarray,
    token_counts: np.ndarray,
    n_human: int,
) -> dict[str, float]:
    denominator = np.asarray(denominator, dtype=np.float64)
    mean_denominator = float(denominator.mean())
    q05, median, q95 = np.quantile(denominator, [0.05, 0.5, 0.95])
    return {
        "spearman_ltheta_stheta": safe_correlation(
            average_ranks(ltheta), average_ranks(stheta)
        ),
        "pearson_ltheta_stheta": safe_correlation(ltheta, stheta),
        "pearson_ltheta_token_count": safe_correlation(ltheta, token_counts),
        "pearson_stheta_token_count": safe_correlation(stheta, token_counts),
        "pearson_denominator_token_count": safe_correlation(
            denominator, token_counts
        ),
        "denominator_mean": mean_denominator,
        "denominator_std": float(denominator.std()),
        "denominator_cv": (
            float(denominator.std() / mean_denominator)
            if mean_denominator != 0.0
            else float("nan")
        ),
        "denominator_min": float(denominator.min()),
        "denominator_q05": float(q05),
        "denominator_median": float(median),
        "denominator_q95": float(q95),
        "denominator_max": float(denominator.max()),
        "denominator_human_mean": float(denominator[:n_human].mean()),
        "denominator_machine_mean": float(denominator[n_human:].mean()),
    }


def pr_auc(human: np.ndarray, machine: np.ndarray) -> float:
    labels = np.concatenate(
        [np.zeros(len(human), dtype=np.int8), np.ones(len(machine), dtype=np.int8)]
    )
    scores = np.concatenate([human, machine])
    precision, recall, _ = precision_recall_curve(labels, scores)
    return float(sklearn_auc(recall, precision))


def metric_bundle(
    evaluator,
    human: np.ndarray,
    machine: np.ndarray,
    calibration: np.ndarray,
    alphas: list[float],
    max_fpr: float,
) -> dict[str, Any]:
    result = evaluator.score_metrics(
        human, machine, calibration, alphas, max_fpr
    )
    result["pr_auc"] = pr_auc(human, machine)
    return result


def flat_metrics(prefix: str, metrics: dict[str, Any], alphas: list[float]):
    row: dict[str, Any] = {
        f"{prefix}_auc": metrics["auc"],
        f"{prefix}_pr_auc": metrics["pr_auc"],
        f"{prefix}_pauc_at_{metrics['max_fpr']:g}": metrics["pauc"],
    }
    for alpha in alphas:
        values = metrics["metrics_by_alpha"][f"{alpha:g}"]
        row[f"{prefix}_type_i_at_{alpha:g}"] = values["type_i_error"]
        row[f"{prefix}_power_at_{alpha:g}"] = values["power"]
    return row


def parse_water_spec(evaluator, value: dict[str, Any]):
    fields = dict(value)
    fields["z_base_knots"] = tuple(fields["z_base_knots"])
    fields["h_base_knots"] = tuple(fields["h_base_knots"])
    return evaluator.WaterSplineSpec(**fields)


def resolve_recorded_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def check_inputs(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    evaluator = args.project_root / "experiments/evaluate_whitebox_reference.py"
    if not evaluator.is_file():
        errors.append(f"missing evaluator: {evaluator}")
    for model in args.models_list:
        for dataset in args.datasets_list:
            if args.reference_result_root is not None:
                reference = (
                    args.reference_result_root / model / dataset / "results.json"
                )
                if not reference.is_file():
                    errors.append(f"missing independent reference result: {reference}")
                    continue
                try:
                    with reference.open("r", encoding="utf-8") as stream:
                        protocol = json.load(stream)["protocol"]
                    roots_and_result = (
                        resolve_recorded_path(
                            args.project_root, protocol["calibration_source"]
                        ),
                        resolve_recorded_path(
                            args.project_root, protocol["test_human_machine_source"]
                        ),
                        resolve_recorded_path(
                            args.project_root, protocol["water_theta_source"]
                        ),
                    )
                    dev_root, test_root, development = roots_and_result
                except Exception as exc:
                    errors.append(f"invalid reference provenance {reference}: {exc}")
                    continue
            else:
                dev_root = args.dev_cache_root
                test_root = args.test_cache_root
                development = (
                    args.dev_result_root / model / dataset / "results.json"
                )
            if not development.is_file():
                errors.append(f"missing frozen development result: {development}")
            for root, label in (
                (dev_root, "development"),
                (test_root, "independent test"),
            ):
                cell = root / model / dataset
                for filename in (
                    "human_reference.npz",
                    "machine_reference.npz",
                    "calibration_reference.npz",
                    "water_reference.npz",
                    "metadata.json",
                ):
                    if not (cell / filename).is_file():
                        errors.append(f"missing {label} cache: {cell / filename}")
    return errors


def build_parser() -> argparse.ArgumentParser:
    project_default = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Frozen white-box theta: compare L_theta and S_theta"
    )
    parser.add_argument("--project-root", type=Path, default=project_default)
    parser.add_argument("--dev-cache-root", type=Path)
    parser.add_argument("--test-cache-root", type=Path)
    parser.add_argument("--dev-result-root", type=Path)
    parser.add_argument(
        "--reference-result-root",
        type=Path,
        help=(
            "Existing independent S_theta result root. When provided, read "
            "test cache, calibration cache, and frozen theta paths from each "
            "cell's results.json provenance; this is recommended for a "
            "separate corrected-LLaMA run."
        ),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--expected-pairs", type=int, default=500)
    parser.add_argument("--expected-fast-samples", type=int, default=10_000)
    parser.add_argument("--alphas", default="0.01,0.05")
    parser.add_argument("--max-fpr", type=float, default=0.05)
    parser.add_argument("--spline-start", type=float, default=-32.0)
    parser.add_argument("--spline-end", type=float, default=0.0)
    parser.add_argument("--n-bases", type=int, default=7)
    parser.add_argument("--spline-degree", type=int, default=2)
    parser.add_argument("--no-intercept", action="store_true")
    parser.add_argument("--water-structures", default="4x4,6x6,8x6")
    parser.add_argument("--score-epsilon", type=float, default=1e-8)
    parser.add_argument("--allow-unverified-token-stream", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def prepare_args(args: argparse.Namespace) -> None:
    args.project_root = args.project_root.resolve()
    if args.dev_cache_root is None:
        args.dev_cache_root = args.project_root / "cache/whitebox/train"
    if args.test_cache_root is None:
        args.test_cache_root = args.project_root / "cache/whitebox/test"
    if args.dev_result_root is None:
        args.dev_result_root = args.project_root / "data/fits/whitebox"
    if args.output_root is None:
        args.output_root = args.project_root / "results/deployment_ablation/whitebox_independent"
    for name in (
        "dev_cache_root",
        "test_cache_root",
        "dev_result_root",
        "output_root",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.reference_result_root is not None:
        args.reference_result_root = args.reference_result_root.resolve()
    args.models_list = parse_csv(args.models)
    args.datasets_list = parse_csv(args.datasets)
    args.alphas_list = parse_float_csv(args.alphas)
    if not args.models_list or not args.datasets_list:
        raise ValueError("model and dataset lists cannot be empty")
    if args.expected_pairs <= 0:
        raise ValueError("--expected-pairs must be positive")
    if args.expected_fast_samples < 2:
        raise ValueError("--expected-fast-samples must be at least two")
    if not 0.0 < args.max_fpr <= 1.0:
        raise ValueError("--max-fpr must be in (0, 1]")
    if any(not 0.0 < alpha < 1.0 for alpha in args.alphas_list):
        raise ValueError("all conformal alphas must be in (0, 1)")
    if args.score_epsilon <= 0.0:
        raise ValueError("--score-epsilon must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        prepare_args(args)
    except Exception as exc:
        parser.error(str(exc))
    errors = check_inputs(args)
    if errors:
        print("INPUT CHECK FAILED", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 2
    print(
        f"INPUT CHECK PASSED: {len(args.models_list) * len(args.datasets_list)} "
        "white-box cells",
        flush=True,
    )
    if args.check_only:
        return 0

    evaluator = import_file(
        args.project_root / "experiments/evaluate_whitebox_reference.py",
        "water_whitebox_reference_for_ltheta_ablation",
    )
    structures = evaluator.parse_structures(args.water_structures)
    ada_spec = evaluator.AdaSplineSpec(
        start=args.spline_start,
        end=args.spline_end,
        n_bases=args.n_bases,
        degree=args.spline_degree,
        intercept=not args.no_intercept,
    )
    rows: list[dict[str, Any]] = []

    for model in args.models_list:
        for dataset in args.datasets_list:
            output_dir = args.output_root / model / dataset
            output_json = output_dir / "results.json"
            if output_json.exists() and not args.overwrite:
                raise FileExistsError(
                    f"{output_json} exists; pass --overwrite or choose another "
                    "--output-root"
                )
            reference_result_path: Path | None = None
            if args.reference_result_root is not None:
                reference_result_path = (
                    args.reference_result_root / model / dataset / "results.json"
                )
                reference_result = evaluator.load_json(reference_result_path)
                reference_protocol = reference_result["protocol"]
                test_cache_root = resolve_recorded_path(
                    args.project_root,
                    reference_protocol["test_human_machine_source"],
                )
                dev_cache_root = resolve_recorded_path(
                    args.project_root,
                    reference_protocol["calibration_source"],
                )
                development_path = resolve_recorded_path(
                    args.project_root,
                    reference_protocol["water_theta_source"],
                )
            else:
                test_cache_root = args.test_cache_root
                dev_cache_root = args.dev_cache_root
                development_path = (
                    args.dev_result_root / model / dataset / "results.json"
                )

            test_caches, test_specs = evaluator.load_cell(
                test_cache_root,
                model,
                dataset,
                expected_documents=args.expected_pairs,
                expected_fast_samples=args.expected_fast_samples,
                expected_spec=ada_spec,
                expected_water_structures=structures,
                require_legacy_verification=(
                    not args.allow_unverified_token_stream
                ),
            )
            dev_caches, dev_specs = evaluator.load_cell(
                dev_cache_root,
                model,
                dataset,
                expected_documents=args.expected_pairs,
                expected_fast_samples=args.expected_fast_samples,
                expected_spec=ada_spec,
                expected_water_structures=structures,
                require_legacy_verification=(
                    not args.allow_unverified_token_stream
                ),
            )
            if test_specs != dev_specs:
                raise ValueError(
                    f"{model}/{dataset}: test and development Water bases differ"
                )

            development = evaluator.load_json(development_path)
            water_final = development["methods"]["Water-Full"]["final_fit"]
            selected_spec = parse_water_spec(
                evaluator, water_final["basis_spec"]
            )
            available = {item.name: item for item in dev_specs}
            if (
                selected_spec.name not in available
                or available[selected_spec.name] != selected_spec
            ):
                raise ValueError(
                    f"{model}/{dataset}: frozen Water basis is absent from cache"
                )
            theta = np.asarray(water_final["theta"], dtype=np.float64)

            human_l, human_s, human_d = score_both(
                test_caches["human"].water_u[selected_spec.name],
                test_caches["human"].water_v[selected_spec.name],
                theta,
                args.score_epsilon,
            )
            machine_l, machine_s, machine_d = score_both(
                test_caches["machine"].water_u[selected_spec.name],
                test_caches["machine"].water_v[selected_spec.name],
                theta,
                args.score_epsilon,
            )
            calibration_l, calibration_s, calibration_d = score_both(
                dev_caches["calibration"].water_u[selected_spec.name],
                dev_caches["calibration"].water_v[selected_spec.name],
                theta,
                args.score_epsilon,
            )

            metrics_l = metric_bundle(
                evaluator,
                human_l,
                machine_l,
                calibration_l,
                args.alphas_list,
                args.max_fpr,
            )
            metrics_s = metric_bundle(
                evaluator,
                human_s,
                machine_s,
                calibration_s,
                args.alphas_list,
                args.max_fpr,
            )
            reference_error: float | None = None
            if reference_result_path is not None:
                reference_scores_path = reference_result_path.with_name("scores.npz")
                if reference_scores_path.is_file():
                    with np.load(reference_scores_path, allow_pickle=False) as arrays:
                        human_key = "Water-Full__human"
                        machine_key = "Water-Full__machine"
                        if human_key in arrays and machine_key in arrays:
                            old_human = np.asarray(arrays[human_key], dtype=np.float64)
                            old_machine = np.asarray(arrays[machine_key], dtype=np.float64)
                            reference_error = max(
                                float(np.max(np.abs(old_human - human_s))),
                                float(np.max(np.abs(old_machine - machine_s))),
                            )
                            if not (
                                np.allclose(old_human, human_s, rtol=1e-7, atol=1e-9)
                                and np.allclose(
                                    old_machine, machine_s, rtol=1e-7, atol=1e-9
                                )
                            ):
                                raise ValueError(
                                    f"{model}/{dataset}: recomputed S_theta differs "
                                    "from the supplied independent reference "
                                    f"(max abs error={reference_error:.3e})"
                                )
            human_tokens = np.diff(test_caches["human"].offsets).astype(float)
            machine_tokens = np.diff(test_caches["machine"].offsets).astype(float)
            if len(human_tokens) != len(human_l) or len(machine_tokens) != len(machine_l):
                raise ValueError(f"{model}/{dataset}: token-count alignment failed")
            combined_l = np.concatenate([human_l, machine_l])
            combined_s = np.concatenate([human_s, machine_s])
            combined_d = np.concatenate([human_d, machine_d])
            combined_tokens = np.concatenate([human_tokens, machine_tokens])
            diagnostic = diagnostics(
                combined_l,
                combined_s,
                combined_d,
                combined_tokens,
                len(human_l),
            )
            diagnostic.update(
                {
                    "calibration_denominator_mean": float(calibration_d.mean()),
                    "calibration_denominator_cv": float(
                        calibration_d.std() / calibration_d.mean()
                    ),
                    "reference_stheta_max_abs_error": reference_error,
                }
            )

            result = {
                "schema_version": 1,
                "builder_version": VERSION,
                "status": "complete_cache_only_matched_deployment_ablation",
                "model": model,
                "dataset": dataset,
                "protocol": {
                    "text_scope": "unchanged full text from independent cache",
                    "theta_training": "unchanged frozen development Water-Full theta",
                    "hyperparameter_selection": "unchanged development selection",
                    "test_fitting_or_reorientation": False,
                    "calibration_source": "development human calibration cache",
                    "calibration_scores_recomputed_separately": True,
                    "ltheta": "theta^T U_X",
                    "stheta": (
                        "theta^T U_X / sqrt(max(theta^T V_X theta, epsilon))"
                    ),
                    "score_epsilon": args.score_epsilon,
                },
                "frozen_development_result": str(development_path),
                "frozen_development_result_sha256": digest_file(development_path),
                "selected_basis_spec": water_final["basis_spec"],
                "selected_config": water_final["config"],
                "independent_reference_result": (
                    None
                    if reference_result_path is None
                    else str(reference_result_path)
                ),
                "n_human": len(human_l),
                "n_machine": len(machine_l),
                "n_calibration": len(calibration_l),
                "metrics": {"S_theta": metrics_s, "L_theta": metrics_l},
                "diagnostics": diagnostic,
            }
            atomic_savez(
                output_dir / "scores.npz",
                human_ltheta=human_l,
                machine_ltheta=machine_l,
                calibration_ltheta=calibration_l,
                human_stheta=human_s,
                machine_stheta=machine_s,
                calibration_stheta=calibration_s,
                human_denominator=human_d,
                machine_denominator=machine_d,
                calibration_denominator=calibration_d,
                human_token_count=human_tokens,
                machine_token_count=machine_tokens,
                theta=theta,
            )
            atomic_json(output_json, result)
            row: dict[str, Any] = {
                "model": model,
                "dataset": dataset,
                "n_human": len(human_l),
                "n_machine": len(machine_l),
                "n_calibration": len(calibration_l),
            }
            row.update(flat_metrics("stheta", metrics_s, args.alphas_list))
            row.update(flat_metrics("ltheta", metrics_l, args.alphas_list))
            comparable = [
                "auc",
                "pr_auc",
                f"pauc_at_{args.max_fpr:g}",
                *[
                    value
                    for alpha in args.alphas_list
                    for value in (
                        f"type_i_at_{alpha:g}",
                        f"power_at_{alpha:g}",
                    )
                ],
            ]
            for name in comparable:
                row[f"delta_l_minus_s_{name}"] = (
                    float(row[f"ltheta_{name}"])
                    - float(row[f"stheta_{name}"])
                )
            row.update(diagnostic)
            rows.append(row)
            atomic_csv(args.output_root / "cell_comparison.csv", rows)
            print(
                f"DONE {model}/{dataset}: "
                f"S-AUC={metrics_s['auc']:.6f} "
                f"L-AUC={metrics_l['auc']:.6f} "
                f"delta={metrics_l['auc'] - metrics_s['auc']:+.6f}",
                flush=True,
            )

    metric_names = [
        "auc",
        "pr_auc",
        f"pauc_at_{args.max_fpr:g}",
        *[
            value
            for alpha in args.alphas_list
            for value in (f"type_i_at_{alpha:g}", f"power_at_{alpha:g}")
        ],
    ]
    macro: dict[str, Any] = {"n_cells": len(rows)}
    for name in metric_names:
        s_values = np.asarray([row[f"stheta_{name}"] for row in rows], dtype=float)
        l_values = np.asarray([row[f"ltheta_{name}"] for row in rows], dtype=float)
        macro[f"mean_stheta_{name}"] = float(s_values.mean())
        macro[f"mean_ltheta_{name}"] = float(l_values.mean())
        macro[f"mean_delta_l_minus_s_{name}"] = float((l_values - s_values).mean())
        macro[f"ltheta_better_cells_{name}"] = int(np.sum(l_values > s_values))
        macro[f"stheta_better_cells_{name}"] = int(np.sum(s_values > l_values))
    macro["mean_spearman_ltheta_stheta"] = float(
        np.nanmean([row["spearman_ltheta_stheta"] for row in rows])
    )
    macro["mean_denominator_cv"] = float(
        np.nanmean([row["denominator_cv"] for row in rows])
    )
    atomic_csv(args.output_root / "macro_comparison.csv", [macro])
    print(f"WROTE {args.output_root / 'cell_comparison.csv'}", flush=True)
    print(f"WROTE {args.output_root / 'macro_comparison.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
