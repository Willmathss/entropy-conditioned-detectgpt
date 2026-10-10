#!/usr/bin/env python3
"""Fresh confirmatory black-box evaluation with frozen L_theta as primary."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
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
    normalize_text,
    parse_csv,
    read_json,
    read_jsonl,
    require_validation_lock,
    score_diagnostics,
    score_ltheta,
    score_stheta,
    two_way_bootstrap_rows,
)


VERSION = "2026.09.02-v1-blackbox-frozen-ltheta-confirmation"
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


@dataclass
class FrozenCell:
    source: str
    dataset: str
    generated_path: Path
    result_path: Path
    frozen_model_path: Path
    result: dict[str, Any]
    theta: np.ndarray
    spec: Any
    sorted_entropy: np.ndarray
    request_id: str
    texts: dict[str, list[str]]
    word_counts: dict[str, np.ndarray]


def load_generated(path: Path, expected: int) -> tuple[dict[str, list[str]], dict[str, np.ndarray]]:
    rows = read_jsonl(path)
    if len(rows) != expected:
        raise ValueError(f"{path}: expected {expected} rows, found {len(rows)}")
    mapping = {
        "human": "human_text",
        "machine": "machine_text",
        "calibration": "calibration_text",
    }
    texts: dict[str, list[str]] = {}
    counts: dict[str, np.ndarray] = {}
    for split, field in mapping.items():
        values = [normalize_text(row.get(field, "")) for row in rows]
        if any(not value for value in values):
            raise ValueError(f"{path}: empty {field}")
        texts[split] = values
        counts[split] = np.asarray([len(value.split()) for value in values], dtype=float)
    if not np.array_equal(counts["human"], counts["machine"]):
        raise ValueError(f"{path}: human/machine word counts are not paired")
    return texts, counts


def load_frozen_cell(
    tensor,
    source: str,
    dataset: str,
    generated_root: Path,
    frozen_root: Path,
    expected: int,
) -> FrozenCell:
    generated_path = generated_root / source / f"{dataset}.jsonl"
    directory = frozen_root / source / dataset / "full"
    result_path = directory / "results.json"
    frozen_model_path = directory / "frozen_model.npz"
    for path in (generated_path, result_path, frozen_model_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    result = read_json(result_path)
    with np.load(frozen_model_path, allow_pickle=False) as arrays:
        theta = np.asarray(arrays["theta"], dtype=np.float64)
        sorted_entropy = np.asarray(arrays["sorted_entropy"], dtype=np.float64)
    spec = tensor.BasisSpec(**result["source_basis_spec"])
    request_id = tensor.exact_request_digest(spec, sorted_entropy)
    if request_id != result.get("exact_request_id"):
        raise ValueError(f"{source}/{dataset}: frozen request digest mismatch")
    if theta.ndim != 1 or len(theta) != spec.dimension or not np.isfinite(theta).all():
        raise ValueError(f"{source}/{dataset}: invalid frozen theta")
    texts, word_counts = load_generated(generated_path, expected)
    return FrozenCell(
        source,
        dataset,
        generated_path,
        result_path,
        frozen_model_path,
        result,
        theta,
        spec,
        sorted_entropy,
        request_id,
        texts,
        word_counts,
    )


def build_store(tensor, args: argparse.Namespace):
    from transformers import AutoConfig

    sampling_config = AutoConfig.from_pretrained(
        str(args.sampling_model_dir), local_files_only=True, trust_remote_code=True
    )
    scoring_config = AutoConfig.from_pretrained(
        str(args.scoring_model_dir), local_files_only=True, trust_remote_code=True
    )
    common_vocab_size = min(
        int(sampling_config.vocab_size), int(scoring_config.vocab_size)
    )
    context_id, context = tensor.exact_model_context(
        args.sampling_model_dir,
        args.scoring_model_dir,
        sampling_dtype=args.sampling_dtype,
        scoring_dtype=args.scoring_dtype,
        max_length=args.max_length,
        common_vocab_size=common_vocab_size,
    )
    return (
        tensor.ExactFeatureStore(
            args.cache_root,
            model_context_id=context_id,
            model_context=context,
        ),
        context_id,
    )


def warm_features(tensor, store, cells: Sequence[FrozenCell], args: argparse.Namespace) -> int:
    pending: list[tuple[str, list[str], dict[str, tuple[Any, np.ndarray]]]] = []
    for cell in cells:
        request = {cell.request_id: (cell.spec, cell.sorted_entropy)}
        for split in ("human", "machine", "calibration"):
            texts = cell.texts[split]
            if not store.is_valid(texts, cell.request_id, cell.spec, cell.sorted_entropy):
                pending.append((f"{cell.source}/{cell.dataset}/{split}", texts, request))
    if not pending:
        print("BLACKBOX CACHE HIT: all fresh exact features are complete")
        return 0
    if args.score_only:
        raise FileNotFoundError(
            f"{len(pending)} fresh feature sets are missing under --score-only"
        )
    sampling_device, scoring_device = tensor.resolve_proxy_devices(
        args.sampling_device, args.scoring_device
    )
    print(
        f"BUILDING {len(pending)} fresh feature sets; "
        f"GPT-J={sampling_device}, GPT-Neo={scoring_device}",
        flush=True,
    )
    sampling_model, sampling_tokenizer = tensor.load_local_model(
        args.sampling_model_dir, args.sampling_dtype, sampling_device
    )
    scoring_model, scoring_tokenizer = tensor.load_local_model(
        args.scoring_model_dir, args.scoring_dtype, scoring_device
    )
    built = 0
    try:
        for label, texts, request in pending:
            built += tensor.write_exact_feature_set(
                store,
                texts,
                request,
                sampling_model=sampling_model,
                sampling_tokenizer=sampling_tokenizer,
                scoring_model=scoring_model,
                scoring_tokenizer=scoring_tokenizer,
                max_length=args.max_length,
                chunk_tokens=args.chunk_tokens,
                document_batch_size=args.document_batch_size,
                request_batch_size=1,
                label=label,
                overwrite=False,
            )
    finally:
        del sampling_model, sampling_tokenizer, scoring_model, scoring_tokenizer
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return built


def run_signature(cell: FrozenCell, validation_lock: Path, model_context_id: str) -> str:
    value = {
        "version": VERSION,
        "protocol": PROTOCOL_NAME,
        "primary_score": "L_theta",
        "validation_lock": file_sha256(validation_lock),
        "generated": file_sha256(cell.generated_path),
        "frozen_result": file_sha256(cell.result_path),
        "frozen_model": file_sha256(cell.frozen_model_path),
        "model_context_id": model_context_id,
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--validation-lock", type=Path, required=True)
    parser.add_argument("--frozen-factorial-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sampling-model-dir", type=Path, required=True)
    parser.add_argument("--scoring-model-dir", type=Path, required=True)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--expected-pairs", type=int, default=500)
    parser.add_argument("--sampling-dtype", default="float16")
    parser.add_argument("--scoring-dtype", default="float16")
    parser.add_argument("--sampling-device", default="cuda:0")
    parser.add_argument("--scoring-device", default="cuda:1")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--chunk-tokens", type=int, default=16)
    parser.add_argument("--document-batch-size", type=int, default=2)
    parser.add_argument("--score-epsilon", type=float, default=1e-8)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--overwrite-results", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in (
        "project_root",
        "generated_root",
        "validation_lock",
        "frozen_factorial_root",
        "cache_root",
        "output_root",
        "sampling_model_dir",
        "scoring_model_dir",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    models = parse_csv(args.models)
    datasets = parse_csv(args.datasets)
    if not models or not datasets or args.expected_pairs <= 0:
        parser.error("invalid model/dataset/sample configuration")
    if args.score_epsilon <= 0 or args.chunk_tokens <= 0 or args.document_batch_size <= 0:
        parser.error("epsilon/chunk/batch values must be positive")
    require_validation_lock(
        args.validation_lock, args.generated_root, models, datasets
    )
    tensor = import_file(
        args.project_root / "experiments/detect_tensor_rayleigh_blackbox.py",
        "water_tensor_for_fresh_ltheta",
    )
    cells = [
        load_frozen_cell(
            tensor,
            source,
            dataset,
            args.generated_root,
            args.frozen_factorial_root,
            args.expected_pairs,
        )
        for source in models
        for dataset in datasets
    ]
    print(f"INPUT CHECK PASSED: {len(cells)} frozen cells")
    if args.check_only:
        return 0
    store, model_context_id = build_store(tensor, args)
    built = warm_features(tensor, store, cells, args)
    print(f"FRESH EXACT FEATURES READY: {built} entries built")

    rows: list[dict[str, Any]] = []
    for cell in cells:
        signature = run_signature(cell, args.validation_lock, model_context_id)
        output_dir = args.output_root / cell.source / cell.dataset
        output_json = output_dir / "results.json"
        if output_json.is_file() and not args.overwrite_results:
            previous = read_json(output_json)
            if previous.get("run_signature") != signature:
                raise FileExistsError(
                    f"{output_json} belongs to another run; choose a new output root"
                )
            row = previous.get("summary_row")
            if not isinstance(row, dict):
                raise ValueError(f"{output_json}: missing summary_row")
            rows.append(row)
            print(f"SKIP verified {cell.source}/{cell.dataset}")
            continue

        scored: dict[str, dict[str, np.ndarray]] = {}
        for split in ("human", "machine", "calibration"):
            u, v = store.load(
                cell.texts[split], cell.request_id, cell.spec, cell.sorted_entropy
            )
            ltheta = score_ltheta(u, cell.theta)
            stheta, denominator = score_stheta(
                u, v, cell.theta, args.score_epsilon
            )
            scored[split] = {
                "ltheta": ltheta,
                "stheta": stheta,
                "denominator": denominator,
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
            cell.word_counts["human"],
            cell.word_counts["machine"],
        )
        row: dict[str, Any] = {
            "source_model": cell.source,
            "dataset": cell.dataset,
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
            "source_model": cell.source,
            "dataset": cell.dataset,
            "run_signature": signature,
            "validation_lock_sha256": file_sha256(args.validation_lock),
            "generated_file_sha256": file_sha256(cell.generated_path),
            "frozen_result": str(cell.result_path),
            "frozen_result_sha256": file_sha256(cell.result_path),
            "frozen_model_sha256": file_sha256(cell.frozen_model_path),
            "exact_model_context_id": model_context_id,
            "exact_request_id": cell.request_id,
            "selected_hyperparameters": cell.result["selected_hyperparameters"],
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
            human_word_count=cell.word_counts["human"],
            machine_word_count=cell.word_counts["machine"],
            calibration_word_count=cell.word_counts["calibration"],
            theta=cell.theta,
        )
        atomic_json(output_json, result)
        rows.append(row)
        atomic_csv(args.output_root / "cell_results.csv", rows)
        print(
            f"DONE {cell.source}/{cell.dataset}: L={metrics_l['auroc']:.6f}, "
            f"S={metrics_s['auroc']:.6f}",
            flush=True,
        )

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
