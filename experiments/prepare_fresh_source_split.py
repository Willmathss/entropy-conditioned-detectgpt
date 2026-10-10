#!/usr/bin/env python3
"""Select and freeze fresh test/calibration source passages.

This script reuses the project's existing dataset readers, but supports several
exclusion roots at once.  It selects 1,000 previously unused source passages
per domain, assigns 500 to independent test and 500 to human calibration, and
writes a protocol lock before any model output is inspected.
"""

from __future__ import annotations

import argparse
import heapq
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from confirmatory_common import (
    PROTOCOL_NAME,
    SCHEMA_VERSION,
    atomic_json,
    atomic_jsonl,
    file_sha256,
    parse_csv,
    read_jsonl,
)


DEFAULT_DATASETS = ("c4", "pubmed", "squad", "writing", "xsum")
DEFAULT_EXCLUSIONS = (
    "data/train/sources",
    "data/train/text",
    "data/test/sources",
    "data/test/text",
    "data/test/text",
)


def import_original(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("water_original_splitter", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def exclusion_files(roots: Sequence[Path], dataset: str) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root.is_file():
            candidates = [root] if root.suffix == ".jsonl" else []
        elif root.is_dir():
            candidates = list(root.rglob(f"{dataset}.jsonl"))
        else:
            continue
        for path in candidates:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                result.append(resolved)
    return sorted(result)


def role_rows(
    original,
    selected: Sequence[Any],
    dataset: str,
    role: str,
    selection_seed: int,
    role_assignment_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(selected):
        text = str(example.source_text)
        rows.append(
            {
                "record_id": f"{dataset}-{role}-{index:04d}",
                "dataset": dataset,
                "role": role,
                "split_index": index,
                "source_index": int(example.source_index),
                "source_id": str(example.source_id),
                "source_text": text,
                "source_text_sha256": original.text_hash(text),
                "source_prefix48_sha256": original.prefix_hash(text, 48),
                "selection_seed": selection_seed,
                "role_assignment_seed": role_assignment_seed,
            }
        )
    return rows


def write_immutable_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        if read_jsonl(path) != rows:
            raise FileExistsError(
                f"{path} already exists with different content; choose a new "
                "--output-root rather than overwriting a viewed split"
            )
        return
    atomic_jsonl(path, rows)


def select_domain(original, args: argparse.Namespace, dataset: str) -> dict[str, Any]:
    manifest = original.load_manifest(args.source_manifest)
    spec = manifest[dataset]
    paths = exclusion_files(args.exclude_roots, dataset)
    if not paths:
        raise FileNotFoundError(
            f"{dataset}: no exclusion files found; refusing an unverified split"
        )
    used_indices, old_full_hashes, old_prefix_hashes = original.collect_exclusions(paths)
    min_words = int(spec.get("min_words", 1))
    max_source_records = spec.get("max_source_records")
    max_source_records = int(max_source_records) if max_source_records else None
    needed = args.n_test + args.n_calibration

    candidates: list[tuple[int, int, Any]] = []
    seen_full: set[str] = set()
    seen_prefix: set[str] = set()
    scanned = excluded_index = excluded_text = excluded_short = 0
    for example in original.iter_source_examples(dataset, spec, args.cache_dir):
        scanned += 1
        if max_source_records is not None and scanned > max_source_records:
            break
        if int(example.source_index) in used_indices:
            excluded_index += 1
            continue
        text = str(example.source_text)
        if len(original.normalize_text(text).split()) < min_words:
            excluded_short += 1
            continue
        # Match collect_exclusions exactly, including its PubMed separator rule.
        full = original.text_hash(text)
        prefix = original.prefix_hash(text, 48)
        if (
            full in old_full_hashes
            or prefix in old_prefix_hashes
            or full in seen_full
            or prefix in seen_prefix
        ):
            excluded_text += 1
            continue
        seen_full.add(full)
        seen_prefix.add(prefix)
        priority = original.stable_priority(
            args.seed, dataset, str(example.source_id), text
        )
        item = (-priority, scanned, example)
        if len(candidates) < needed:
            heapq.heappush(candidates, item)
        elif priority < -candidates[0][0]:
            heapq.heapreplace(candidates, item)

    selected = [
        example
        for _, _, example in sorted(candidates, key=lambda item: (-item[0], item[1]))
    ]
    if len(selected) != needed:
        raise RuntimeError(
            f"{dataset}: only {len(selected)} unused passages found; need {needed}"
        )
    # Use a second independent deterministic priority for the test/calibration
    # assignment.  The selection priority only decides which unused records enter
    # the 1,000-passage pool; it does not decide their experimental role.
    role_assignment_seed = args.seed + 1
    selected.sort(
        key=lambda example: original.stable_priority(
            role_assignment_seed,
            dataset,
            str(example.source_id),
            str(example.source_text),
        )
    )
    test_examples = selected[: args.n_test]
    calibration_examples = selected[args.n_test :]
    test_rows = role_rows(
        original,
        test_examples,
        dataset,
        "independent_test",
        args.seed,
        role_assignment_seed,
    )
    calibration_rows = role_rows(
        original,
        calibration_examples,
        dataset,
        "human_calibration",
        args.seed,
        role_assignment_seed,
    )

    test_path = args.output_root / "test" / f"{dataset}.jsonl"
    calibration_path = args.output_root / "calibration" / f"{dataset}.jsonl"
    write_immutable_jsonl(test_path, test_rows)
    write_immutable_jsonl(calibration_path, calibration_rows)
    return {
        "dataset": dataset,
        "test_path": str(test_path.resolve()),
        "test_sha256": file_sha256(test_path),
        "calibration_path": str(calibration_path.resolve()),
        "calibration_sha256": file_sha256(calibration_path),
        "n_test": len(test_rows),
        "n_calibration": len(calibration_rows),
        "source_records_scanned": scanned,
        "source_spec": spec,
        "excluded_by_source_index": excluded_index,
        "excluded_by_text_fingerprint": excluded_text,
        "excluded_too_short": excluded_short,
        "exclusion_files": [str(path) for path in paths],
        "exclusion_file_sha256": {str(path): file_sha256(path) for path in paths},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--n-calibration", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--cache-dir")
    parser.add_argument(
        "--exclude-root",
        action="append",
        dest="exclude_roots",
        type=Path,
        help="Repeat for every earlier train/calibration/test data root.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.project_root = args.project_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.n_test <= 0 or args.n_calibration <= 0:
        parser.error("split sizes must be positive")
    args.datasets_list = parse_csv(args.datasets)
    if not args.datasets_list:
        parser.error("--datasets cannot be empty")
    if args.exclude_roots is None:
        args.exclude_roots = [args.project_root / path for path in DEFAULT_EXCLUSIONS]
    else:
        args.exclude_roots = [path.resolve() for path in args.exclude_roots]

    original_path = args.project_root / "experiments/generate_independent_whitebox_test.py"
    original = import_original(original_path)
    cells = [select_domain(original, args, dataset) for dataset in args.datasets_list]
    lock = {
        "schema_version": SCHEMA_VERSION,
        "protocol_name": PROTOCOL_NAME,
        "status": "source_split_frozen_before_generation",
        "primary_score": "L_theta",
        "secondary_locked_ablation": "S_theta",
        "theta_and_hyperparameters": "reuse frozen train-only fit",
        "calibration": "fresh independent human set",
        "test": "fresh independent human/machine set",
        "selection_seed": args.seed,
        "role_assignment_seed": args.seed + 1,
        "source_reader_path": str(original_path),
        "source_reader_sha256": file_sha256(original_path),
        "prefix_overlap_words": 48,
        "planned_metrics": [
            "AUROC",
            "PR-AUC",
            "normalized pAUC@1%",
            "normalized pAUC@5%",
            "TPR@FPR=1%",
            "TPR@FPR=5%",
            "conformal FPR/power@1%",
            "conformal FPR/power@5%",
        ],
        "confirmatory_interpretation": {
            "primary_endpoints": [
                "white-box macro mean delta AUROC (L_theta minus S_theta)",
                "black-box macro mean delta AUROC (L_theta minus S_theta)",
            ],
            "directional_replication": "both primary endpoint deltas are positive",
            "strong_replication": (
                "both descriptive model-by-domain bootstrap 95% intervals "
                "exclude zero on the positive side"
            ),
            "bootstrap_seed": 20260902,
            "bootstrap_repetitions": 20000,
            "all_secondary_endpoints_reported_regardless_of_direction": True,
        },
        "cells": cells,
    }
    lock_path = args.output_root.parent / "source_split_lock.json"
    if lock_path.exists():
        with lock_path.open("r", encoding="utf-8") as stream:
            existing = json.load(stream)
        if existing != lock:
            raise FileExistsError(
                f"{lock_path} differs; choose a new output root instead of overwriting"
            )
    else:
        atomic_json(lock_path, lock)
    print(f"FROZEN {len(cells)} source domains: {lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
