#!/usr/bin/env python3
"""Validate fresh generated cells and write the immutable scoring lock."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from confirmatory_common import (
    PROTOCOL_NAME,
    SCHEMA_VERSION,
    atomic_json,
    file_sha256,
    normalize_text,
    parse_csv,
    prefix_sha256,
    read_json,
    read_jsonl,
    text_sha256,
)


DEFAULT_MODELS = ("falcon-7b", "llama-2-7b", "opt-2.7b", "opt-6.7b")
DEFAULT_DATASETS = ("c4", "pubmed", "squad", "writing", "xsum")


def text_values(value: Any, parent: str = "") -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key).lower()
            if isinstance(child, str) and (
                name in {
                    "text",
                    "human_text",
                    "machine_text",
                    "calibration_text",
                    "source_text",
                    "original",
                    "sampled",
                    "prompt_text",
                }
                or any(token in name for token in ("human", "calibration", "source"))
            ):
                if len(normalize_text(child).split()) >= 8:
                    yield child
            else:
                yield from text_values(child, name)
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, str) and parent in {"original", "sampled", "calibration"}:
                if len(normalize_text(child).split()) >= 8:
                    yield child
            else:
                yield from text_values(child, parent)


def old_fingerprints(roots: Sequence[Path]) -> tuple[set[str], set[str]]:
    full: set[str] = set()
    prefix: set[str] = set()
    for root in roots:
        if root.is_file():
            paths = [root]
        elif root.is_dir():
            paths = list(root.rglob("*.jsonl")) + list(root.rglob("*.raw_data.json"))
        else:
            continue
        for path in sorted(set(paths)):
            try:
                values: list[Any]
                if path.suffix == ".jsonl":
                    values = read_jsonl(path)
                else:
                    values = [read_json(path)]
                for value in values:
                    for text in text_values(value):
                        full.add(text_sha256(text))
                        prefix.add(prefix_sha256(text, 48))
            except (OSError, ValueError):
                continue
    return full, prefix


def require_unique(values: Sequence[str], label: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{label}: duplicate text detected")


def validate_cell(
    path: Path,
    metadata_path: Path,
    test_source_path: Path,
    calibration_source_path: Path,
    expected: int,
    old_full: set[str],
    old_prefix: set[str],
    test_sources: Sequence[Mapping[str, Any]],
    calibration_sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = read_jsonl(path)
    metadata = read_json(metadata_path)
    if len(rows) != expected:
        raise ValueError(f"{path}: expected {expected} rows, found {len(rows)}")
    if metadata.get("output_sha256") != file_sha256(path):
        raise ValueError(f"{metadata_path}: output hash mismatch")
    if metadata.get("protocol_name") != PROTOCOL_NAME:
        raise ValueError(f"{metadata_path}: wrong protocol")
    if metadata.get("status") != "generated_and_frozen":
        raise ValueError(f"{metadata_path}: generation is not complete and frozen")
    if metadata.get("test_source_sha256") != file_sha256(test_source_path):
        raise ValueError(f"{metadata_path}: test source hash mismatch")
    if metadata.get("calibration_source_sha256") != file_sha256(calibration_source_path):
        raise ValueError(f"{metadata_path}: calibration source hash mismatch")
    for index, row in enumerate(rows):
        test_source = test_sources[index]
        calibration_source = calibration_sources[index]
        if (
            row.get("test_record_id") != test_source.get("record_id")
            or row.get("test_source_sha256")
            != test_source.get("source_text_sha256")
        ):
            raise ValueError(f"{path}: test source mismatch at row {index}")
        if (
            row.get("calibration_record_id")
            != calibration_source.get("record_id")
            or row.get("calibration_source_sha256")
            != calibration_source.get("source_text_sha256")
        ):
            raise ValueError(f"{path}: calibration source mismatch at row {index}")

    human = [normalize_text(row.get("human_text", "")) for row in rows]
    machine = [normalize_text(row.get("machine_text", "")) for row in rows]
    calibration = [normalize_text(row.get("calibration_text", "")) for row in rows]
    if any(not text for values in (human, machine, calibration) for text in values):
        raise ValueError(f"{path}: empty text")
    require_unique(human, f"{path} human")
    require_unique(machine, f"{path} machine")
    require_unique(calibration, f"{path} calibration")
    if any(len(left.split()) != len(right.split()) for left, right in zip(human, machine)):
        raise ValueError(f"{path}: human/machine word lengths are not paired")

    human_full = {text_sha256(text) for text in human}
    calibration_full = {text_sha256(text) for text in calibration}
    human_prefix = {prefix_sha256(text, 48) for text in human}
    calibration_prefix = {prefix_sha256(text, 48) for text in calibration}
    if human_full.intersection(calibration_full):
        raise ValueError(f"{path}: human test/calibration exact overlap")
    if human_prefix.intersection(calibration_prefix):
        raise ValueError(f"{path}: human test/calibration 48-word-prefix overlap")
    if human_full.intersection(old_full) or calibration_full.intersection(old_full):
        raise ValueError(f"{path}: fresh human/calibration text overlaps old data")
    if human_prefix.intersection(old_prefix) or calibration_prefix.intersection(old_prefix):
        raise ValueError(f"{path}: fresh human/calibration prefix overlaps old data")

    machine_full = {text_sha256(text) for text in machine}
    if machine_full.intersection(old_full):
        raise ValueError(f"{path}: fresh machine text exactly overlaps old data")
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": file_sha256(metadata_path),
        "n_human": len(human),
        "n_machine": len(machine),
        "n_calibration": len(calibration),
        "paired_test_word_lengths": True,
        "human_calibration_exact_and_prefix_disjoint": True,
        "old_data_overlap": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-split-root", type=Path, required=True)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--output-lock", type=Path, required=True)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--expected-pairs", type=int, default=500)
    parser.add_argument(
        "--exclude-root",
        action="append",
        dest="exclude_roots",
        type=Path,
        required=True,
        help="Repeat for every earlier train/calibration/test text root.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    models = parse_csv(args.models)
    datasets = parse_csv(args.datasets)
    if args.expected_pairs <= 0 or not models or not datasets:
        parser.error("invalid expected-pairs/model/dataset list")
    args.source_split_root = args.source_split_root.resolve()
    args.generated_root = args.generated_root.resolve()
    source_lock_path = args.source_split_root.parent / "source_split_lock.json"
    source_lock = read_json(source_lock_path)
    if source_lock.get("status") != "source_split_frozen_before_generation":
        raise ValueError(f"{source_lock_path}: invalid source lock")
    old_full, old_prefix = old_fingerprints(
        [path.resolve() for path in args.exclude_roots]
    )
    locked_cells = source_lock.get("cells")
    if not isinstance(locked_cells, list):
        raise ValueError(f"{source_lock_path}: missing locked source cells")
    locked_by_dataset = {
        str(cell.get("dataset")): cell
        for cell in locked_cells
        if isinstance(cell, Mapping)
    }
    source_rows: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for dataset in datasets:
        locked = locked_by_dataset.get(dataset)
        if not isinstance(locked, Mapping):
            raise ValueError(f"{source_lock_path}: dataset {dataset} is not locked")
        test_path = args.source_split_root / "test" / f"{dataset}.jsonl"
        calibration_path = (
            args.source_split_root / "calibration" / f"{dataset}.jsonl"
        )
        if (
            file_sha256(test_path) != locked.get("test_sha256")
            or file_sha256(calibration_path) != locked.get("calibration_sha256")
        ):
            raise ValueError(
                f"{dataset}: source split differs from source_split_lock.json"
            )
        test_rows = read_jsonl(test_path)
        calibration_rows = read_jsonl(calibration_path)
        if len(test_rows) != args.expected_pairs or len(calibration_rows) != args.expected_pairs:
            raise ValueError(f"{dataset}: locked source split count mismatch")
        source_rows[dataset] = (test_rows, calibration_rows)
    generated_files: dict[str, Any] = {}
    for model in models:
        for dataset in datasets:
            path = args.generated_root / model / f"{dataset}.jsonl"
            metadata_path = path.with_suffix(".metadata.json")
            if not path.is_file() or not metadata_path.is_file():
                raise FileNotFoundError(path if not path.is_file() else metadata_path)
            generated_files[f"{model}/{dataset}"] = validate_cell(
                path,
                metadata_path,
                args.source_split_root / "test" / f"{dataset}.jsonl",
                args.source_split_root / "calibration" / f"{dataset}.jsonl",
                args.expected_pairs,
                old_full,
                old_prefix,
                source_rows[dataset][0],
                source_rows[dataset][1],
            )

    lock = {
        "schema_version": SCHEMA_VERSION,
        "protocol_name": PROTOCOL_NAME,
        "status": "passed_and_frozen",
        "primary_score": "L_theta",
        "secondary_locked_ablation": "S_theta",
        "no_score_selection_on_fresh_data": True,
        "source_split_lock": str(source_lock_path),
        "source_split_lock_sha256": file_sha256(source_lock_path),
        "expected_pairs": args.expected_pairs,
        "models": models,
        "datasets": datasets,
        "generated_files": generated_files,
        "exclusion_roots": [str(path.resolve()) for path in args.exclude_roots],
        "old_full_text_fingerprints": len(old_full),
        "old_prefix48_fingerprints": len(old_prefix),
    }
    args.output_lock = args.output_lock.resolve()
    if args.output_lock.exists():
        if read_json(args.output_lock) != lock:
            raise FileExistsError(
                f"{args.output_lock} differs; do not overwrite a scoring lock"
            )
    else:
        atomic_json(args.output_lock, lock)
    print(f"VALIDATION PASSED: {len(generated_files)} cells; {args.output_lock}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
