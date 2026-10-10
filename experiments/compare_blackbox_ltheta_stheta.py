#!/usr/bin/env python3
"""Matched black-box deployment-score ablation: frozen theta, L_theta vs S_theta.

This script deliberately does not fit a direction, select hyperparameters, load
the source generator, or rebuild text.  It reuses the completed ``full`` cell
from ``run_blackbox_factorial_ablation.py`` and the exact U/V feature store.

For every source-model/test-domain cell it computes

    L_theta(X) = theta^T U_X
    S_theta(X) = L_theta(X) / sqrt(max(theta^T V_X theta, epsilon))

with exactly the same frozen theta.  Human calibration scores are recomputed
under each deployment score before conformal metrics are evaluated.
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
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np


VERSION = "2026.08.25-v1-blackbox-ltheta-vs-stheta"
DEFAULT_MODELS = ("falcon-7b", "llama-2-7b", "opt-2.7b", "opt-6.7b")
DEFAULT_DATASETS = ("c4", "pubmed", "squad", "writing", "xsum")


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def import_file(path: Path, module_name: str):
    if not path.is_file():
        raise FileNotFoundError(path)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
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
    quantiles = np.quantile(denominator, [0.05, 0.5, 0.95])
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
        "denominator_q05": float(quantiles[0]),
        "denominator_median": float(quantiles[1]),
        "denominator_q95": float(quantiles[2]),
        "denominator_max": float(denominator.max()),
        "denominator_human_mean": float(denominator[:n_human].mean()),
        "denominator_machine_mean": float(denominator[n_human:].mean()),
    }


def check_inputs(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    required_scripts = (
        args.project_root / "experiments/detect_tensor_rayleigh_blackbox.py",
        args.project_root / "experiments/run_blackbox_factorial_ablation.py",
    )
    for path in required_scripts:
        if not path.is_file():
            errors.append(f"missing script: {path}")
    for path in (args.sampling_model_dir, args.scoring_model_dir):
        if not path.is_dir():
            errors.append(f"missing local model directory: {path}")
    for source in args.source_models_list:
        for dataset in args.datasets_list:
            directory = args.factorial_root / source / dataset / "full"
            for filename in ("results.json", "frozen_model.npz", "scores.npz"):
                path = directory / filename
                if not path.is_file():
                    errors.append(f"missing completed Full artifact: {path}")
            adapter = (
                args.run_root / "input_adapters" / source /
                f"{dataset}.raw_data.json"
            )
            if not adapter.is_file():
                errors.append(f"missing independent adapter: {adapter}")
    progress = list(
        (args.run_root / "tensor_cache").rglob("build_progress_v2.json")
    ) if (args.run_root / "tensor_cache").is_dir() else []
    if progress:
        errors.append(f"incomplete exact feature cache: {progress[0]}")
    return errors


def build_parser() -> argparse.ArgumentParser:
    project_default = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Frozen black-box theta: compare L_theta and S_theta"
    )
    parser.add_argument("--project-root", type=Path, default=project_default)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--factorial-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--sampling-model-dir", type=Path)
    parser.add_argument("--scoring-model-dir", type=Path)
    parser.add_argument("--source-models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--structures", default="4x4,6x6,8x6")
    parser.add_argument("--sampling-dtype", default="float16")
    parser.add_argument("--scoring-dtype", default="float16")
    parser.add_argument("--sampling-device", default="cpu")
    parser.add_argument("--scoring-device", default="cpu")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--chunk-tokens", type=int, default=16)
    parser.add_argument("--document-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--score-epsilon", type=float, default=1e-8)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def prepare_args(args: argparse.Namespace) -> None:
    args.project_root = args.project_root.resolve()
    if args.run_root is None:
        args.run_root = (
            args.project_root / "cache/blackbox/independent"
        )
    args.run_root = args.run_root.resolve()
    if args.factorial_root is None:
        args.factorial_root = args.run_root / "tensor_factorial_ablation"
    args.factorial_root = args.factorial_root.resolve()
    if args.output_root is None:
        args.output_root = args.run_root / "ltheta_deployment_ablation"
    args.output_root = args.output_root.resolve()
    if args.sampling_model_dir is None:
        args.sampling_model_dir = args.project_root / "models/gpt-j-6b"
    if args.scoring_model_dir is None:
        args.scoring_model_dir = args.project_root / "models/gpt-neo-2.7b"
    args.sampling_model_dir = args.sampling_model_dir.resolve()
    args.scoring_model_dir = args.scoring_model_dir.resolve()
    args.source_models_list = parse_csv(args.source_models)
    args.datasets_list = parse_csv(args.datasets)
    if not args.source_models_list or not args.datasets_list:
        raise ValueError("source-model and dataset lists cannot be empty")
    if args.score_epsilon <= 0.0:
        raise ValueError("--score-epsilon must be positive")


def flatten_metrics(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


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
        f"INPUT CHECK PASSED: {len(args.source_models_list) * len(args.datasets_list)} "
        "completed Full cells",
        flush=True,
    )
    if args.check_only:
        return 0

    tensor = import_file(
        args.project_root / "experiments/detect_tensor_rayleigh_blackbox.py",
        "water_tensor_blackbox_for_ltheta_ablation",
    )
    runner = import_file(
        args.project_root / "experiments/run_blackbox_factorial_ablation.py",
        "water_blackbox_factorial_for_ltheta_ablation",
    )
    structures = runner.parse_structures(args.structures)
    unknown_models = set(args.source_models_list) - set(runner.SOURCE_MODELS)
    unknown_datasets = set(args.datasets_list) - set(runner.DATASETS)
    if unknown_models or unknown_datasets:
        raise ValueError(
            f"unknown models={sorted(unknown_models)}, "
            f"datasets={sorted(unknown_datasets)}"
        )

    context = SimpleNamespace(
        project_root=args.project_root,
        run_root=args.run_root,
        output_root=args.factorial_root,
        sampling_model_dir=args.sampling_model_dir,
        scoring_model_dir=args.scoring_model_dir,
        sampling_dtype=args.sampling_dtype,
        scoring_dtype=args.scoring_dtype,
        sampling_device=args.sampling_device,
        scoring_device=args.scoring_device,
        max_length=args.max_length,
        chunk_tokens=args.chunk_tokens,
        document_batch_size=args.document_batch_size,
        seed=args.seed,
        structures_list=structures,
    )
    store = runner.build_store(tensor, context)
    full_variant = runner.VARIANT_BY_KEY["full"]
    rows: list[dict[str, Any]] = []

    for source in args.source_models_list:
        for dataset in args.datasets_list:
            output_dir = args.output_root / source / dataset
            output_json = output_dir / "results.json"
            if output_json.is_file() and not args.overwrite:
                raise FileExistsError(
                    f"{output_json} exists; pass --overwrite or choose another "
                    "--output-root"
                )
            (
                frozen_dir,
                frozen_result,
                selected,
                theta,
                source_spec,
                sorted_entropy,
                request_id,
            ) = runner.load_frozen_context(
                tensor, context, source, dataset, full_variant
            )
            (
                _training_domains,
                raw_data,
                token_caches,
                requests,
                _lookup,
            ) = runner.load_cell_context(tensor, store, context, source, dataset)
            if request_id not in requests:
                raise ValueError(
                    f"{source}/{dataset}: frozen exact request is absent from "
                    "the current cache plan"
                )
            test_pair = tensor.load_exact_domain_pair(
                store, raw_data[dataset], request_id, requests[request_id]
            )
            test_pair = runner.effective_pair(
                test_pair, selected, full_variant.entropy_conditioning
            )
            human_l, human_s, human_d = score_both(
                test_pair[0][0], test_pair[0][1], theta, args.score_epsilon
            )
            machine_l, machine_s, machine_d = score_both(
                test_pair[1][0], test_pair[1][1], theta, args.score_epsilon
            )

            calibration_texts = runner.load_adapter(
                runner.adapter_path(
                    args.run_root, dataset, source, development=False
                )
            )["calibration"]
            if not store.is_valid(
                calibration_texts, request_id, source_spec, sorted_entropy
            ):
                raise FileNotFoundError(
                    f"{source}/{dataset}: exact calibration U/V cache is missing. "
                    "Finish the calibration phase of the existing factorial run; "
                    "this script intentionally refuses to load either LM."
                )
            calibration_u, calibration_v = store.load(
                calibration_texts, request_id, source_spec, sorted_entropy
            )
            if not full_variant.entropy_conditioning:
                calibration_u, calibration_v = runner.project_entropy_independent(
                    calibration_u,
                    calibration_v,
                    selected.source_j_bases,
                    selected.source_k_bases,
                )
            calibration_l, calibration_s, calibration_d = score_both(
                calibration_u, calibration_v, theta, args.score_epsilon
            )

            with np.load(frozen_dir / "scores.npz", allow_pickle=False) as old:
                old_human = np.asarray(old["human"], dtype=np.float64)
                old_machine = np.asarray(old["machine"], dtype=np.float64)
            reference_error = max(
                float(np.max(np.abs(old_human - human_s))),
                float(np.max(np.abs(old_machine - machine_s))),
            )
            if not (
                np.allclose(old_human, human_s, rtol=1e-7, atol=1e-9)
                and np.allclose(old_machine, machine_s, rtol=1e-7, atol=1e-9)
            ):
                raise ValueError(
                    f"{source}/{dataset}: recomputed S_theta differs from the "
                    f"frozen Full score (max abs error={reference_error:.3e})"
                )

            human_tokens = np.diff(token_caches[dataset][0].offsets).astype(float)
            machine_tokens = np.diff(token_caches[dataset][1].offsets).astype(float)
            if len(human_tokens) != len(human_l) or len(machine_tokens) != len(machine_l):
                raise ValueError(f"{source}/{dataset}: token-count alignment failed")
            combined_l = np.concatenate([human_l, machine_l])
            combined_s = np.concatenate([human_s, machine_s])
            combined_d = np.concatenate([human_d, machine_d])
            combined_tokens = np.concatenate([human_tokens, machine_tokens])

            metrics_l = runner.metrics_from_scores(
                human_l, machine_l, calibration_l
            )
            metrics_s = runner.metrics_from_scores(
                human_s, machine_s, calibration_s
            )
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
                "source_model": source,
                "dataset": dataset,
                "protocol": {
                    "text_scope": "unchanged full text from frozen adapter/cache",
                    "theta_training": "unchanged frozen Full theta",
                    "hyperparameter_selection": "unchanged frozen Full selection",
                    "test_fitting_or_reorientation": False,
                    "calibration_reused_as_texts_only": True,
                    "calibration_scores_recomputed_separately": True,
                    "ltheta": "theta^T U_X",
                    "stheta": (
                        "theta^T U_X / sqrt(max(theta^T V_X theta, epsilon))"
                    ),
                    "score_epsilon": args.score_epsilon,
                },
                "frozen_full_result": str(frozen_dir / "results.json"),
                "frozen_full_result_sha256": digest_file(
                    frozen_dir / "results.json"
                ),
                "selected_hyperparameters": frozen_result[
                    "selected_hyperparameters"
                ],
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
                theta=np.asarray(theta, dtype=np.float64),
            )
            atomic_json(output_json, result)
            row: dict[str, Any] = {
                "source_model": source,
                "dataset": dataset,
                "n_human": len(human_l),
                "n_machine": len(machine_l),
                "n_calibration": len(calibration_l),
            }
            row.update(flatten_metrics("stheta", metrics_s))
            row.update(flatten_metrics("ltheta", metrics_l))
            for name in metrics_s:
                left = metrics_l[name]
                right = metrics_s[name]
                row[f"delta_l_minus_s_{name}"] = (
                    None if left is None or right is None else float(left) - float(right)
                )
            row.update(diagnostic)
            rows.append(row)
            atomic_csv(args.output_root / "cell_comparison.csv", rows)
            print(
                f"DONE {source}/{dataset}: "
                f"S-AUROC={metrics_s['auroc']:.6f} "
                f"L-AUROC={metrics_l['auroc']:.6f} "
                f"delta={metrics_l['auroc'] - metrics_s['auroc']:+.6f}",
                flush=True,
            )

    metric_names = list(runner.metrics_from_scores(
        np.asarray([0.0, 0.1]),
        np.asarray([0.9, 1.0]),
        np.asarray([0.0, 0.1]),
    ))
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
