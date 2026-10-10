#!/usr/bin/env python3
"""Build black-box raw-data adapters from fixed independent LLaMA JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


DATASETS = ("c4", "pubmed", "squad", "writing", "xsum")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/test/text/llama-2-7b"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/adapters/test/llama-2-7b"),
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def validate_effective_config(metadata_path: Path) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    generation = metadata.get("generation")
    if not isinstance(generation, dict):
        raise ValueError(f"{metadata_path}: missing generation mapping")
    expected = {
        "sampling_mode": "top_k",
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 50,
        "prompt_tokens": 120,
        "min_length": 150,
        "max_length": 200,
        "batch_size": 8,
    }
    differences = {
        key: {"expected": wanted, "actual": generation.get(key)}
        for key, wanted in expected.items()
        if generation.get(key) != wanted
    }
    if differences:
        raise ValueError(
            f"{metadata_path}: effective generation config mismatch: {differences}"
        )
    return generation


def build_dataset(args: argparse.Namespace, dataset: str) -> dict[str, Any]:
    source = args.source_root / f"{dataset}.jsonl"
    metadata_path = source.with_suffix(".metadata.json")
    if not source.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing fixed generation files for {dataset}: {source}, {metadata_path}"
        )
    generation = validate_effective_config(metadata_path)
    rows = read_jsonl(source)
    if len(rows) != args.n_samples:
        raise ValueError(f"{source}: expected {args.n_samples} rows, found {len(rows)}")

    required = ("human_text", "machine_text", "calibration_text")
    for index, row in enumerate(rows):
        missing = [key for key in required if not isinstance(row.get(key), str) or not row[key]]
        if missing:
            raise ValueError(f"{source}: row {index} invalid fields: {missing}")
        row_generation = row.get("generation")
        if not isinstance(row_generation, dict):
            raise ValueError(f"{source}: row {index} missing generation metadata")
        for key in ("temperature", "top_p", "top_k"):
            if row_generation.get(key) != generation.get(key):
                raise ValueError(
                    f"{source}: row {index} has inconsistent generation.{key}"
                )

    output = args.output_root / f"{dataset}.raw_data.json"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
    payload = {
        "original": [row["human_text"] for row in rows],
        "sampled": [row["machine_text"] for row in rows],
        "calibration": [row["calibration_text"] for row in rows],
    }
    atomic_json(output, payload)
    return {
        "dataset": dataset,
        "source": str(source),
        "source_sha256": sha256(source),
        "adapter": str(output),
        "adapter_sha256": sha256(output),
        "n_samples": len(rows),
        "generation": generation,
    }


def main() -> None:
    args = parse_args()
    if args.n_samples <= 0:
        raise ValueError("--n-samples must be positive")
    records = [build_dataset(args, dataset) for dataset in args.datasets]
    manifest = args.output_root.parent / "adapter_manifest.json"
    if manifest.exists() and not args.overwrite:
        raise FileExistsError(f"{manifest} exists; pass --overwrite to replace it")
    atomic_json(
        manifest,
        {
            "schema_version": 1,
            "purpose": "fixed_independent_llama_blackbox_adapters",
            "model": "llama-2-7b",
            "records": records,
        },
    )
    print(f"Built {len(records)} validated adapters under {args.output_root}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
