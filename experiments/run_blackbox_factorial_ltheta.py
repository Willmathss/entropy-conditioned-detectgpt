#!/usr/bin/env python3
"""Cache-only black-box H/S/C factorial under the linear T_theta score.

This runner converts the completed black-box 2^3 factorial from the historical
passage-normalized deployment score to the paper's current primary score

    T_theta(X) = theta.T @ U_X.

It deliberately freezes every previously fitted factorial direction and every
train-only selected hyperparameter.  Thus H/S/C are unchanged and the only
intervention is the deployment score.  Independent-test and calibration U/V
features are loaded from the existing exact feature store; neither proxy LLM
nor any source generator is loaded.  The corrected LLaMA run is routed through
``blackbox_independent_llama_fixed_20260821`` while Falcon/OPT use the frozen
August run.

Before accepting a cell, the runner reconstructs the old S_theta score from
the same U/V and theta and requires it to match the completed factorial score.
For Full, it also requires T_theta metrics to match the corrected 20-cell
deployment comparison used by the current paper.  These checks prevent a
silent fallback to the pre-fix LLaMA artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np


VERSION = "2026.09.10-v1-blackbox-factorial-ltheta-rescore"
SOURCE_MODELS = ("falcon-7b", "llama-2-7b", "opt-2.7b", "opt-6.7b")
DATASETS = ("c4", "pubmed", "squad", "writing", "xsum")
VARIANT_KEYS = ("b0", "h", "s", "c", "hs", "hc", "sc", "full")
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
EXPECTED_COUNT = 500
EXPECTED_CORRECTED_LLAMA_GENERATION = {
    "sampling_mode": "top_k",
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 50,
    "prompt_tokens": 120,
    "min_length": 150,
    "max_length": 200,
}


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


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


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_both(
    u: np.ndarray,
    v: np.ndarray,
    theta: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u_array = np.asarray(u, dtype=np.float64)
    v_array = np.asarray(v, dtype=np.float64)
    direction = np.asarray(theta, dtype=np.float64).reshape(-1)
    if u_array.ndim != 2 or u_array.shape[1] != len(direction):
        raise ValueError(
            f"U/theta mismatch: U={u_array.shape}, theta={direction.shape}"
        )
    expected_v = (len(u_array), len(direction), len(direction))
    if v_array.shape != expected_v:
        raise ValueError(f"V shape {v_array.shape}, expected {expected_v}")
    linear = u_array @ direction
    variance = np.einsum(
        "d,nde,e->n", direction, v_array, direction, optimize=True
    )
    denominator = np.sqrt(np.clip(variance, epsilon, None))
    normalized = linear / denominator
    for name, values in (
        ("T_theta", linear),
        ("S_theta", normalized),
        ("denominator", denominator),
    ):
        if not np.isfinite(values).all():
            raise FloatingPointError(f"non-finite {name}")
    return linear, normalized, denominator


def assert_array_close(
    actual: np.ndarray,
    expected: np.ndarray,
    label: str,
    *,
    rtol: float,
    atol: float,
) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape:
        raise ValueError(
            f"{label}: shape mismatch {actual.shape} != {expected.shape}"
        )
    error = float(np.max(np.abs(actual - expected))) if len(actual) else 0.0
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        raise ValueError(
            f"{label}: reconstructed S_theta differs from frozen score; "
            f"max_abs_error={error:.6g}"
        )
    return error


def assert_metrics_close(
    actual: dict[str, Any],
    expected: dict[str, Any],
    label: str,
    atol: float,
) -> None:
    for metric in METRIC_NAMES:
        if metric not in expected:
            raise ValueError(f"{label}: reference is missing {metric}")
        left = float(actual[metric])
        right = float(expected[metric])
        if not math.isfinite(left) or abs(left - right) > atol:
            raise ValueError(
                f"{label}/{metric}: expected {right:.12g}, found {left:.12g}"
            )


def validate_llama_manifest(run_root: Path, datasets: Sequence[str]) -> None:
    path = run_root / "input_adapters/adapter_manifest.json"
    manifest = load_json(path)
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{path}: missing adapter records")
    by_dataset = {
        str(item.get("dataset")): item
        for item in records
        if isinstance(item, dict)
    }
    for dataset in datasets:
        if dataset not in by_dataset:
            raise ValueError(f"{path}: missing corrected {dataset} record")
        generation = by_dataset[dataset].get("generation")
        if not isinstance(generation, dict):
            raise ValueError(f"{path}: {dataset} generation metadata is missing")
        differences = {
            key: {"expected": wanted, "actual": generation.get(key)}
            for key, wanted in EXPECTED_CORRECTED_LLAMA_GENERATION.items()
            if generation.get(key) != wanted
        }
        if differences:
            raise ValueError(
                f"{path}: {dataset} is not the corrected LLaMA generation: "
                f"{differences}"
            )


def bootstrap_interval(
    values: np.ndarray,
    *,
    seed: int,
    samples: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or samples < 1:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 500):
        stop = min(samples, start + 500)
        indices = rng.integers(0, len(values), (stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def source_paths(args: argparse.Namespace, source: str) -> tuple[Path, Path]:
    if source == "llama-2-7b":
        return args.llama_run_root, args.llama_factorial_root
    return args.base_run_root, args.base_factorial_root


def make_context(
    args: argparse.Namespace,
    run_root: Path,
    factorial_root: Path,
) -> SimpleNamespace:
    return SimpleNamespace(
        project_root=args.project_root,
        run_root=run_root,
        output_root=factorial_root,
        sampling_model_dir=args.sampling_model_dir,
        scoring_model_dir=args.scoring_model_dir,
        sampling_dtype=args.sampling_dtype,
        scoring_dtype=args.scoring_dtype,
        sampling_device="cpu",
        scoring_device="cpu",
        max_length=args.max_length,
        chunk_tokens=args.chunk_tokens,
        document_batch_size=args.document_batch_size,
        seed=args.seed,
        structures_list=args.structures_list,
    )


def load_reference_metrics(args: argparse.Namespace) -> dict[tuple[str, str], dict[str, float]]:
    path = args.main_ltheta_comparison_root / "cell_comparison.csv"
    rows = read_csv(path)
    result: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        key = (row.get("source_model", ""), row.get("dataset", ""))
        if key in result:
            raise ValueError(f"{path}: duplicate row {key}")
        metrics = {}
        for metric in METRIC_NAMES:
            field = f"ltheta_{metric}"
            if field not in row:
                raise ValueError(f"{path}: missing {field}")
            metrics[metric] = float(row[field])
        result[key] = metrics
    expected = {
        (source, dataset)
        for source in args.source_models_list
        for dataset in args.datasets_list
    }
    missing = expected - set(result)
    if missing:
        raise ValueError(f"{path}: missing current Full L_theta rows {sorted(missing)}")
    return result


def validate_static_inputs(args: argparse.Namespace, runner) -> None:
    validate_llama_manifest(args.llama_run_root, args.datasets_list)
    for root in {args.base_run_root, args.llama_run_root}:
        if not root.is_dir():
            raise FileNotFoundError(root)
        progress = list((root / "tensor_cache").rglob("build_progress_v2.json"))
        if progress:
            raise ValueError(f"incomplete exact feature cache: {progress[0]}")
    for source in args.source_models_list:
        run_root, factorial_root = source_paths(args, source)
        for dataset in args.datasets_list:
            adapter = runner.adapter_path(
                run_root, dataset, source, development=False
            )
            if not adapter.is_file():
                raise FileNotFoundError(adapter)
            for variant_key in VARIANT_KEYS:
                directory = factorial_root / source / dataset / variant_key
                for filename in ("results.json", "frozen_model.npz", "scores.npz"):
                    if not (directory / filename).is_file():
                        raise FileNotFoundError(directory / filename)
                result = load_json(directory / "results.json")
                if result.get("status") != "complete":
                    raise ValueError(f"{directory}: old factorial result is incomplete")
                if int(result.get("n_human", -1)) != args.expected_pairs:
                    raise ValueError(f"{directory}: wrong human count")
                if int(result.get("n_machine", -1)) != args.expected_pairs:
                    raise ValueError(f"{directory}: wrong machine count")
                if int(result.get("n_calibration", -1)) != args.expected_pairs:
                    raise ValueError(f"{directory}: wrong calibration count")


def signature_for(
    args: argparse.Namespace,
    source: str,
    dataset: str,
    variant_key: str,
    frozen_dir: Path,
    adapter_path: Path,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "version": VERSION,
        "source_model": source,
        "dataset": dataset,
        "variant": variant_key,
        "score": "T_theta(X)=theta^T U_X",
        "score_epsilon_for_reference_check": args.score_epsilon,
        "frozen_result_sha256": sha256_file(frozen_dir / "results.json"),
        "frozen_model_sha256": sha256_file(frozen_dir / "frozen_model.npz"),
        "frozen_score_sha256": sha256_file(frozen_dir / "scores.npz"),
        "adapter_sha256": sha256_file(adapter_path),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), payload


def result_is_reusable(
    output_dir: Path,
    signature: str,
    *,
    overwrite: bool,
) -> bool:
    result_path = output_dir / "results.json"
    score_path = output_dir / "scores.npz"
    if not result_path.exists() and not score_path.exists():
        return False
    if overwrite:
        return False
    if not result_path.is_file() or not score_path.is_file():
        raise ValueError(f"{output_dir}: incomplete output; use --overwrite")
    result = load_json(result_path)
    if result.get("run_signature") != signature:
        raise ValueError(
            f"{result_path}: frozen input or protocol changed; use --overwrite "
            "only after confirming the new input"
        )
    if result.get("status") != "complete":
        raise ValueError(f"{result_path}: incomplete output; use --overwrite")
    return True


def score_source(
    tensor,
    runner,
    args: argparse.Namespace,
    source: str,
    full_reference: dict[tuple[str, str], dict[str, float]],
    *,
    write_outputs: bool,
) -> None:
    run_root, factorial_root = source_paths(args, source)
    context = make_context(args, run_root, factorial_root)
    store = runner.build_store(tensor, context)
    variants = {key: runner.VARIANT_BY_KEY[key] for key in VARIANT_KEYS}

    for dataset in args.datasets_list:
        print(f"CELL {source}/{dataset}: loading exact cache", flush=True)
        (
            _training_domains,
            raw_data,
            token_caches,
            requests,
            _lookup,
        ) = runner.load_cell_context(tensor, store, context, source, dataset)
        adapter = runner.adapter_path(
            run_root, dataset, source, development=False
        )
        calibration_texts = runner.load_adapter(adapter)["calibration"]
        test_pairs: dict[str, Any] = {}
        calibration_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for variant_key in VARIANT_KEYS:
            variant = variants[variant_key]
            (
                frozen_dir,
                frozen_result,
                selected,
                theta,
                source_spec,
                sorted_entropy,
                request_id,
            ) = runner.load_frozen_context(
                tensor, context, source, dataset, variant
            )
            signature, signature_payload = signature_for(
                args, source, dataset, variant_key, frozen_dir, adapter
            )
            output_dir = args.output_root / source / dataset / variant_key
            if write_outputs and result_is_reusable(
                output_dir, signature, overwrite=args.overwrite
            ):
                print(f"SKIP {source}/{dataset}/{variant_key}", flush=True)
                continue

            if request_id not in requests:
                raise ValueError(
                    f"{source}/{dataset}/{variant_key}: frozen request is "
                    "absent from the exact test cache plan"
                )
            if request_id not in test_pairs:
                test_pairs[request_id] = tensor.load_exact_domain_pair(
                    store, raw_data[dataset], request_id, requests[request_id]
                )
            test_pair = runner.effective_pair(
                test_pairs[request_id],
                selected,
                variant.entropy_conditioning,
            )
            human_t, human_s, human_d = score_both(
                test_pair[0][0], test_pair[0][1], theta, args.score_epsilon
            )
            machine_t, machine_s, machine_d = score_both(
                test_pair[1][0], test_pair[1][1], theta, args.score_epsilon
            )

            if request_id not in calibration_pairs:
                if not store.is_valid(
                    calibration_texts, request_id, source_spec, sorted_entropy
                ):
                    raise FileNotFoundError(
                        f"{source}/{dataset}/{variant_key}: exact calibration "
                        "U/V is missing. Finish the old factorial calibration "
                        "phase first; this runner never loads either proxy LM."
                    )
                calibration_pairs[request_id] = store.load(
                    calibration_texts, request_id, source_spec, sorted_entropy
                )
            calibration_u, calibration_v = calibration_pairs[request_id]
            if variant.entropy_conditioning:
                effective_u = np.asarray(calibration_u)
                effective_v = np.asarray(calibration_v)
            else:
                effective_u, effective_v = runner.project_entropy_independent(
                    calibration_u,
                    calibration_v,
                    selected.source_j_bases,
                    selected.source_k_bases,
                )
            calibration_t, calibration_s, calibration_d = score_both(
                effective_u, effective_v, theta, args.score_epsilon
            )

            with np.load(frozen_dir / "scores.npz", allow_pickle=False) as old:
                required = {"human", "machine", "calibration"}
                missing = required - set(old.files)
                if missing:
                    raise ValueError(
                        f"{frozen_dir / 'scores.npz'}: missing {sorted(missing)}"
                    )
                errors = {
                    "human": assert_array_close(
                        human_s, old["human"],
                        f"{source}/{dataset}/{variant_key}/human",
                        rtol=args.reference_rtol, atol=args.reference_atol,
                    ),
                    "machine": assert_array_close(
                        machine_s, old["machine"],
                        f"{source}/{dataset}/{variant_key}/machine",
                        rtol=args.reference_rtol, atol=args.reference_atol,
                    ),
                    "calibration": assert_array_close(
                        calibration_s, old["calibration"],
                        f"{source}/{dataset}/{variant_key}/calibration",
                        rtol=args.reference_rtol, atol=args.reference_atol,
                    ),
                }

            metrics = runner.metrics_from_scores(
                human_t, machine_t, calibration_t
            )
            if any(metrics.get(name) is None for name in METRIC_NAMES):
                raise ValueError(f"{source}/{dataset}/{variant_key}: missing metric")
            if variant_key == "full":
                assert_metrics_close(
                    metrics,
                    full_reference[(source, dataset)],
                    f"current Full T_theta {source}/{dataset}",
                    args.metric_atol,
                )

            if write_outputs:
                result = {
                    "schema_version": 1,
                    "builder_version": VERSION,
                    "run_signature": signature,
                    "signature_payload": signature_payload,
                    "status": "complete",
                    "source_model": source,
                    "dataset": dataset,
                    "variant": variant_key,
                    "label": frozen_result.get("label", variant.label),
                    "modules": frozen_result["modules"],
                    "protocol": {
                        "access": "black-box fixed GPT-J/GPT-Neo proxies",
                        "text_scope": "unchanged frozen full text",
                        "direction": "unchanged frozen factorial theta",
                        "hyperparameter_selection": (
                            "unchanged train-domain-only factorial selection"
                        ),
                        "deployment_score": "T_theta(X)=theta^T U_X",
                        "old_score_reconstructed_only_for_validation": (
                            "S_theta(X)=T_theta(X)/sqrt(theta^T V_X theta)"
                        ),
                        "test_used_for_fitting_selection_or_orientation": False,
                        "calibration_used_for_fitting_selection_or_orientation": False,
                        "calibration_scores_recomputed_under_T_theta": True,
                        "proxy_or_source_model_loaded": False,
                    },
                    "frozen_factorial_result": str(frozen_dir / "results.json"),
                    "selected_hyperparameters": frozen_result[
                        "selected_hyperparameters"
                    ],
                    "source_basis_spec": frozen_result["source_basis_spec"],
                    "exact_request_id": request_id,
                    "n_human": len(human_t),
                    "n_machine": len(machine_t),
                    "n_calibration": len(calibration_t),
                    "metrics": metrics,
                    "validation": {
                        "reconstructed_stheta_max_abs_error": max(errors.values()),
                        "full_matches_current_ltheta_main_result": (
                            variant_key == "full"
                        ),
                    },
                    "denominator_diagnostics": {
                        "human_mean": float(human_d.mean()),
                        "machine_mean": float(machine_d.mean()),
                        "calibration_mean": float(calibration_d.mean()),
                    },
                }
                atomic_savez(
                    output_dir / "scores.npz",
                    human=human_t,
                    machine=machine_t,
                    calibration=calibration_t,
                    theta=np.asarray(theta, dtype=np.float64),
                )
                atomic_json(output_dir / "results.json", result)
                print(
                    f"DONE {source}/{dataset}/{variant_key}: "
                    f"T-AUROC={float(metrics['auroc']):.6f}",
                    flush=True,
                )
            else:
                print(
                    f"PREFLIGHT OK {source}/{dataset}/{variant_key}: "
                    f"old-S maxerr={max(errors.values()):.3e}",
                    flush=True,
                )

        del raw_data, token_caches, requests, test_pairs, calibration_pairs


def collect_rows(args: argparse.Namespace, runner) -> list[dict[str, Any]]:
    rows = []
    variant_map = runner.VARIANT_BY_KEY
    for source in args.source_models_list:
        for dataset in args.datasets_list:
            for variant_key in VARIANT_KEYS:
                path = (
                    args.output_root / source / dataset / variant_key / "results.json"
                )
                result = load_json(path)
                if result.get("status") != "complete":
                    raise ValueError(f"incomplete result: {path}")
                variant = variant_map[variant_key]
                selected = result["selected_hyperparameters"]
                row = {
                    "source_model": source,
                    "dataset": dataset,
                    "variant": variant_key,
                    "label": result["label"],
                    "H": int(variant.entropy_conditioning),
                    "S": int(variant.smoothness),
                    "C": int(variant.covariance_shrinkage),
                    "source_structure": (
                        f"{selected['source_j_bases']}x{selected['source_k_bases']}"
                    ),
                    "effective_dimension": (
                        int(selected["j_bases"]) * int(selected["k_bases"])
                    ),
                    "smooth_lambda": selected["smooth_lambda"],
                    "ridge_lambda": selected["ridge_lambda"],
                    "covariance_shrinkage": selected["covariance_shrinkage"],
                    "n_human": result["n_human"],
                    "n_machine": result["n_machine"],
                    "n_calibration": result["n_calibration"],
                }
                row.update(result["metrics"])
                rows.append(row)
    return rows


def write_summaries(args: argparse.Namespace, runner) -> None:
    rows = collect_rows(args, runner)
    atomic_csv(args.output_root / "cell_summary.csv", rows)
    macro_rows = []
    for variant_key in VARIANT_KEYS:
        selected = [row for row in rows if row["variant"] == variant_key]
        variant = runner.VARIANT_BY_KEY[variant_key]
        output = {
            "variant": variant_key,
            "label": variant.label,
            "H": int(variant.entropy_conditioning),
            "S": int(variant.smoothness),
            "C": int(variant.covariance_shrinkage),
            "n_cells": len(selected),
        }
        for metric in METRIC_NAMES:
            output[metric] = float(
                np.mean([float(row[metric]) for row in selected])
            )
        macro_rows.append(output)
    atomic_csv(args.output_root / "macro_summary.csv", macro_rows)

    by_cell = {
        (row["source_model"], row["dataset"], row["variant"]): row
        for row in rows
    }
    cells = list(itertools.product(args.source_models_list, args.datasets_list))
    balanced = {
        "H": [("h", "b0"), ("hs", "s"), ("hc", "c"), ("full", "sc")],
        "S": [("s", "b0"), ("hs", "h"), ("sc", "c"), ("full", "hc")],
        "C": [("c", "b0"), ("hc", "h"), ("sc", "s"), ("full", "hs")],
    }
    leave_one_out = {
        "H": ("full", "sc"),
        "S": ("full", "hc"),
        "C": ("full", "hs"),
    }
    effect_rows = []
    for term_index, (term, pairs) in enumerate(balanced.items()):
        for metric_index, metric in enumerate(METRIC_NAMES):
            values = np.asarray(
                [
                    np.mean(
                        [
                            float(by_cell[(source, dataset, on)][metric])
                            - float(by_cell[(source, dataset, off)][metric])
                            for on, off in pairs
                        ]
                    )
                    for source, dataset in cells
                ],
                dtype=np.float64,
            )
            low, high = bootstrap_interval(
                values,
                seed=args.bootstrap_seed + 1000 * term_index + metric_index,
                samples=args.bootstrap_samples,
            )
            effect_rows.append(
                {
                    "effect_kind": "balanced_factorial_main_effect",
                    "term": term,
                    "metric": metric,
                    "n_cells": len(values),
                    "mean_delta": float(values.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "positive_cells": int(np.sum(values > 0)),
                    "negative_cells": int(np.sum(values < 0)),
                    "zero_cells": int(np.sum(values == 0)),
                }
            )
    for term_index, (term, (full_key, removed_key)) in enumerate(
        leave_one_out.items()
    ):
        for metric_index, metric in enumerate(METRIC_NAMES):
            values = np.asarray(
                [
                    float(by_cell[(source, dataset, full_key)][metric])
                    - float(by_cell[(source, dataset, removed_key)][metric])
                    for source, dataset in cells
                ],
                dtype=np.float64,
            )
            low, high = bootstrap_interval(
                values,
                seed=(
                    args.bootstrap_seed + 10_000 + 1000 * term_index + metric_index
                ),
                samples=args.bootstrap_samples,
            )
            effect_rows.append(
                {
                    "effect_kind": "leave_one_component_out_from_full",
                    "term": term,
                    "metric": metric,
                    "n_cells": len(values),
                    "mean_delta": float(values.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "positive_cells": int(np.sum(values > 0)),
                    "negative_cells": int(np.sum(values < 0)),
                    "zero_cells": int(np.sum(values == 0)),
                }
            )
    atomic_csv(args.output_root / "factorial_effects.csv", effect_rows)
    atomic_json(
        args.output_root / "run_manifest.json",
        {
            "builder_version": VERSION,
            "status": "complete",
            "primary_score": "T_theta(X)=theta^T U_X",
            "factorial_directions": "frozen from completed train-only S-era run",
            "models": args.source_models_list,
            "datasets": args.datasets_list,
            "expected_cells": len(cells),
            "completed_cells": len(cells),
            "variants": list(VARIANT_KEYS),
            "corrected_llama_run_root": str(args.llama_run_root),
            "full_reference_comparison": str(args.main_ltheta_comparison_root),
        },
    )


def self_test() -> None:
    u = np.asarray([[1.0, 2.0], [3.0, -1.0], [0.5, 0.25]])
    v = np.asarray(
        [
            [[2.0, 0.1], [0.1, 1.0]],
            [[1.5, 0.0], [0.0, 0.8]],
            [[0.7, 0.2], [0.2, 1.2]],
        ]
    )
    theta = np.asarray([0.75, -0.5])
    linear, normalized, denominator = score_both(u, v, theta, 1e-8)
    assert np.allclose(linear, u @ theta)
    expected_var = np.einsum("d,nde,e->n", theta, v, theta)
    assert np.allclose(denominator, np.sqrt(expected_var))
    assert np.allclose(normalized, linear / np.sqrt(expected_var))
    values = np.asarray([0.1, 0.2, 0.3])
    low, high = bootstrap_interval(values, seed=7, samples=1000)
    assert 0.1 <= low <= high <= 0.3
    print("SELF-TEST PASSED: T_theta scoring, S_theta reconstruction, bootstrap")


def build_parser() -> argparse.ArgumentParser:
    project = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Cache-only black-box frozen H/S/C factorial under T_theta",
    )
    parser.add_argument("--project-root", type=Path, default=project)
    parser.add_argument(
        "--base-run-root",
        type=Path,
        default=Path("cache/blackbox/independent"),
    )
    parser.add_argument(
        "--llama-run-root",
        type=Path,
        default=Path("cache/blackbox/llama"),
    )
    parser.add_argument("--base-factorial-root", type=Path)
    parser.add_argument("--llama-factorial-root", type=Path)
    parser.add_argument(
        "--main-ltheta-comparison-root",
        type=Path,
        default=Path("results/deployment_ablation/blackbox_independent"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/blackbox/factorial"),
    )
    parser.add_argument("--sampling-model-dir", type=Path)
    parser.add_argument("--scoring-model-dir", type=Path)
    parser.add_argument("--source-models", default=",".join(SOURCE_MODELS))
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--structures", default="4x4,6x6,8x6")
    parser.add_argument("--sampling-dtype", default="float16")
    parser.add_argument("--scoring-dtype", default="float16")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--chunk-tokens", type=int, default=16)
    parser.add_argument("--document-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--expected-pairs", type=int, default=EXPECTED_COUNT)
    parser.add_argument("--score-epsilon", type=float, default=1e-8)
    parser.add_argument("--reference-rtol", type=float, default=1e-7)
    parser.add_argument("--reference-atol", type=float, default=1e-9)
    parser.add_argument("--metric-atol", type=float, default=1e-10)
    parser.add_argument("--bootstrap-seed", type=int, default=20260910)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def resolve_project_path(project: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (project / value).resolve()


def prepare_args(args: argparse.Namespace, runner) -> None:
    args.project_root = args.project_root.resolve()
    for name in (
        "base_run_root",
        "llama_run_root",
        "main_ltheta_comparison_root",
        "output_root",
    ):
        setattr(
            args,
            name,
            resolve_project_path(args.project_root, getattr(args, name)),
        )
    args.base_factorial_root = resolve_project_path(
        args.project_root,
        args.base_factorial_root
        or args.base_run_root / "tensor_factorial_ablation",
    )
    args.llama_factorial_root = resolve_project_path(
        args.project_root,
        args.llama_factorial_root
        or args.llama_run_root / "tensor_factorial_ablation",
    )
    args.sampling_model_dir = resolve_project_path(
        args.project_root,
        args.sampling_model_dir or Path("../gpt-j-6b"),
    )
    args.scoring_model_dir = resolve_project_path(
        args.project_root,
        args.scoring_model_dir or Path("../gpt-neo-2.7b"),
    )
    args.source_models_list = parse_csv(args.source_models)
    args.datasets_list = parse_csv(args.datasets)
    args.structures_list = runner.parse_structures(args.structures)
    unknown_models = set(args.source_models_list) - set(SOURCE_MODELS)
    unknown_datasets = set(args.datasets_list) - set(DATASETS)
    if unknown_models:
        raise ValueError(f"unknown source models: {sorted(unknown_models)}")
    if unknown_datasets:
        raise ValueError(f"unknown datasets: {sorted(unknown_datasets)}")
    if set(args.source_models_list) != set(SOURCE_MODELS):
        raise ValueError("this final factorial runner requires all four source models")
    if set(args.datasets_list) != set(DATASETS):
        raise ValueError("this final factorial runner requires all five datasets")
    if args.expected_pairs != EXPECTED_COUNT:
        raise ValueError(f"--expected-pairs must remain {EXPECTED_COUNT}")
    if args.score_epsilon <= 0:
        raise ValueError("--score-epsilon must be positive")
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive")
    for path in (
        args.sampling_model_dir,
        args.scoring_model_dir,
        args.base_factorial_root,
        args.llama_factorial_root,
        args.main_ltheta_comparison_root,
    ):
        if not path.exists():
            raise FileNotFoundError(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    args.project_root = args.project_root.resolve()
    tensor_path = args.project_root / "experiments/detect_tensor_rayleigh_blackbox.py"
    runner_path = args.project_root / "experiments/run_blackbox_factorial_ablation.py"
    try:
        tensor = import_file(tensor_path, "blackbox_tensor_for_ltheta_factorial")
        runner = import_file(runner_path, "blackbox_factorial_source_for_ltheta")
        prepare_args(args, runner)
        validate_static_inputs(args, runner)
        full_reference = load_reference_metrics(args)
    except Exception as exc:
        parser.error(str(exc))

    print(
        "STATIC INPUT CHECK PASSED: 20 cells, 8 frozen variants, corrected LLaMA",
        flush=True,
    )
    for source in args.source_models_list:
        score_source(
            tensor,
            runner,
            args,
            source,
            full_reference,
            write_outputs=not args.preflight_only,
        )
    if args.preflight_only:
        print(
            "PREFLIGHT PASSED: all old S_theta scores reproduced and every "
            "Full T_theta result matches the corrected current main result",
            flush=True,
        )
        return 0

    write_summaries(args, runner)
    print(f"COMPLETE: summaries in {args.output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
