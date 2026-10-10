#!/usr/bin/env python3
"""Fresh confirmatory white-box evaluation with frozen L_theta as primary."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from confirmatory_common import (
    PROTOCOL_NAME,
    atomic_csv,
    atomic_json,
    atomic_savez,
    file_sha256,
    flatten_metrics,
    macro_rows,
    metrics_from_scores,
    parse_csv,
    read_json,
    require_validation_lock,
    score_diagnostics,
    score_ltheta,
    score_stheta,
    two_way_bootstrap_rows,
)


VERSION = "2026.09.02-v1-whitebox-frozen-ltheta-confirmation"
DEFAULT_MODELS = ("falcon-7b", "llama-2-7b", "opt-2.7b", "opt-6.7b")
DEFAULT_DATASETS = ("c4", "pubmed", "squad", "writing", "xsum")


def import_file(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_structures(value: str) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for item in parse_csv(value):
        pieces = item.lower().split("x")
        if len(pieces) != 2:
            raise ValueError(f"invalid structure: {item}")
        result.append((int(pieces[0]), int(pieces[1])))
    return result


def selected_spec(evaluator, value: dict[str, Any]):
    fields = dict(value)
    fields["z_base_knots"] = tuple(fields["z_base_knots"])
    fields["h_base_knots"] = tuple(fields["h_base_knots"])
    return evaluator.WaterSplineSpec(**fields)


def run_signature(
    validation_lock: Path, frozen_result: Path, cache_metadata: Path
) -> str:
    value = {
        "version": VERSION,
        "protocol": PROTOCOL_NAME,
        "primary_score": "L_theta",
        "validation_lock": file_sha256(validation_lock),
        "frozen_result": file_sha256(frozen_result),
        "fresh_cache_metadata": file_sha256(cache_metadata),
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--fresh-cache-root", type=Path, required=True)
    parser.add_argument("--frozen-result-root", type=Path, required=True)
    parser.add_argument("--validation-lock", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--expected-pairs", type=int, default=500)
    parser.add_argument("--expected-fast-samples", type=int, default=10_000)
    parser.add_argument("--water-structures", default="4x4,6x6,8x6")
    parser.add_argument("--spline-start", type=float, default=-32.0)
    parser.add_argument("--spline-end", type=float, default=0.0)
    parser.add_argument("--n-bases", type=int, default=7)
    parser.add_argument("--spline-degree", type=int, default=2)
    parser.add_argument("--no-intercept", action="store_true")
    parser.add_argument("--score-epsilon", type=float, default=1e-8)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--overwrite-results", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in (
        "project_root",
        "generated_root",
        "fresh_cache_root",
        "frozen_result_root",
        "validation_lock",
        "output_root",
    ):
        setattr(args, name, getattr(args, name).resolve())
    models = parse_csv(args.models)
    datasets = parse_csv(args.datasets)
    structures = parse_structures(args.water_structures)
    if not models or not datasets or not structures or args.expected_pairs <= 0:
        parser.error("invalid model/dataset/structure/sample configuration")
    require_validation_lock(
        args.validation_lock, args.generated_root, models, datasets
    )
    evaluator = import_file(
        args.project_root / "experiments/evaluate_whitebox_reference.py",
        "water_whitebox_evaluator_for_fresh_ltheta",
    )
    ada_spec = evaluator.AdaSplineSpec(
        start=args.spline_start,
        end=args.spline_end,
        n_bases=args.n_bases,
        degree=args.spline_degree,
        intercept=not args.no_intercept,
    )

    rows: list[dict[str, Any]] = []
    for model in models:
        for dataset in datasets:
            generated_path = args.generated_root / model / f"{dataset}.jsonl"
            cache_metadata_path = args.fresh_cache_root / model / dataset / "metadata.json"
            frozen_result_path = args.frozen_result_root / model / dataset / "results.json"
            for path in (generated_path, cache_metadata_path, frozen_result_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            cache_metadata = read_json(cache_metadata_path)
            cached_generated = Path(cache_metadata["generated_path"]).expanduser()
            if not cached_generated.is_absolute():
                cached_generated = args.project_root / cached_generated
            if not cached_generated.is_file():
                # Historical cache metadata may name the old workstation path.
                # The content hash below, not that absolute path, defines the input.
                cached_generated = generated_path
            if file_sha256(cached_generated) != file_sha256(generated_path):
                raise ValueError(f"{model}/{dataset}: cache was built from other text")

            signature = run_signature(
                args.validation_lock, frozen_result_path, cache_metadata_path
            )
            output_dir = args.output_root / model / dataset
            output_json = output_dir / "results.json"
            if (
                output_json.is_file()
                and not args.overwrite_results
                and not args.check_only
            ):
                previous = read_json(output_json)
                if previous.get("run_signature") != signature:
                    raise FileExistsError(
                        f"{output_json} belongs to another run; choose a new output root"
                    )
                row = previous.get("summary_row")
                if not isinstance(row, dict):
                    raise ValueError(f"{output_json}: missing summary_row")
                rows.append(row)
                print(f"SKIP verified {model}/{dataset}")
                continue

            caches, specs = evaluator.load_cell(
                args.fresh_cache_root,
                model,
                dataset,
                expected_documents=args.expected_pairs,
                expected_fast_samples=args.expected_fast_samples,
                expected_spec=ada_spec,
                expected_water_structures=structures,
                require_legacy_verification=False,
            )
            development = read_json(frozen_result_path)
            final_fit = development["methods"]["Water-Full"]["final_fit"]
            spec = selected_spec(evaluator, final_fit["basis_spec"])
            available = {item.name: item for item in specs}
            if spec.name not in available or available[spec.name] != spec:
                raise ValueError(f"{model}/{dataset}: frozen basis mismatch")
            theta = np.asarray(final_fit["theta"], dtype=np.float64).reshape(-1)
            feature_dimension = caches["human"].water_u[spec.name].shape[1]
            if theta.ndim != 1 or theta.size != feature_dimension:
                raise ValueError(
                    f"{model}/{dataset}: theta dimension {theta.shape} does not "
                    f"match fresh feature dimension {feature_dimension}"
                )
            if args.check_only:
                print(
                    f"CHECKED {model}/{dataset}: immutable input, fresh cache, "
                    "frozen basis, and frozen theta agree",
                    flush=True,
                )
                continue
            scored: dict[str, dict[str, np.ndarray]] = {}
            for split in ("human", "machine", "calibration"):
                u = caches[split].water_u[spec.name]
                v = caches[split].water_v[spec.name]
                ltheta = score_ltheta(u, theta)
                stheta, denominator = score_stheta(
                    u, v, theta, args.score_epsilon
                )
                scored[split] = {
                    "ltheta": ltheta,
                    "stheta": stheta,
                    "denominator": denominator,
                    "token_count": np.diff(caches[split].offsets).astype(float),
                }

            metrics_l = metrics_from_scores(
                scored["human"]["ltheta"],
                scored["machine"]["ltheta"],
                scored["calibration"]["ltheta"],
            )
            metrics_s = metrics_from_scores(
                scored["human"]["stheta"],
                scored["machine"]["stheta"],
                scored["calibration"]["stheta"],
            )
            diagnostic = score_diagnostics(
                scored["human"]["ltheta"],
                scored["machine"]["ltheta"],
                scored["human"]["stheta"],
                scored["machine"]["stheta"],
                scored["human"]["denominator"],
                scored["machine"]["denominator"],
                scored["human"]["token_count"],
                scored["machine"]["token_count"],
            )
            row: dict[str, Any] = {
                "model": model,
                "dataset": dataset,
                "n_human": args.expected_pairs,
                "n_machine": args.expected_pairs,
                "n_calibration": args.expected_pairs,
            }
            row.update(flatten_metrics("ltheta", metrics_l))
            row.update(flatten_metrics("stheta", metrics_s))
            for key in metrics_l:
                row[f"delta_l_minus_s_{key}"] = metrics_l[key] - metrics_s[key]
            row.update(diagnostic)
            result = {
                "schema_version": 1,
                "builder_version": VERSION,
                "protocol_name": PROTOCOL_NAME,
                "status": "complete_fresh_confirmatory_evaluation",
                "primary_score": "L_theta",
                "secondary_locked_ablation": "S_theta",
                "model": model,
                "dataset": dataset,
                "run_signature": signature,
                "validation_lock_sha256": file_sha256(args.validation_lock),
                "generated_file_sha256": file_sha256(generated_path),
                "fresh_cache_metadata_sha256": file_sha256(cache_metadata_path),
                "frozen_train_result": str(frozen_result_path),
                "frozen_train_result_sha256": file_sha256(frozen_result_path),
                "selected_basis_spec": final_fit["basis_spec"],
                "selected_config": final_fit["config"],
                "test_fitting_selection_or_orientation": False,
                "metrics": {"L_theta_primary": metrics_l, "S_theta_ablation": metrics_s},
                "diagnostics": diagnostic,
                "summary_row": row,
            }
            atomic_savez(
                output_dir / "scores.npz",
                human_ltheta=scored["human"]["ltheta"],
                machine_ltheta=scored["machine"]["ltheta"],
                calibration_ltheta=scored["calibration"]["ltheta"],
                human_stheta=scored["human"]["stheta"],
                machine_stheta=scored["machine"]["stheta"],
                calibration_stheta=scored["calibration"]["stheta"],
                human_denominator=scored["human"]["denominator"],
                machine_denominator=scored["machine"]["denominator"],
                calibration_denominator=scored["calibration"]["denominator"],
                human_token_count=scored["human"]["token_count"],
                machine_token_count=scored["machine"]["token_count"],
                calibration_token_count=scored["calibration"]["token_count"],
                theta=theta,
            )
            atomic_json(output_json, result)
            rows.append(row)
            atomic_csv(args.output_root / "cell_results.csv", rows)
            print(
                f"DONE {model}/{dataset}: L={metrics_l['auroc']:.6f}, "
                f"S={metrics_s['auroc']:.6f}",
                flush=True,
            )

    if args.check_only:
        return 0
    atomic_csv(args.output_root / "cell_results.csv", rows)
    atomic_csv(args.output_root / "macro_results.csv", macro_rows(rows))
    atomic_csv(
        args.output_root / "two_way_bootstrap.csv",
        two_way_bootstrap_rows(rows),
    )
    print(f"WROTE {args.output_root / 'macro_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
