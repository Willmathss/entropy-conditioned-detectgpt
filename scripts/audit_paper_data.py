#!/usr/bin/env python3
"""Validate and inventory the materialized text inputs used by the paper."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


MODELS = ("llama-2-7b", "opt-2.7b", "opt-6.7b", "falcon-7b")
DATASETS = ("c4", "pubmed", "squad", "writing", "xsum")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_text(value: str) -> str:
    """Match the independent-test generator's calibration text normalization."""
    return " ".join(value.split())


def record(root: Path, path: Path, n_records: int | None = None) -> dict:
    item = {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": digest(path),
    }
    if n_records is not None:
        item["records"] = n_records
    return item


def audit(root: Path) -> dict:
    train_root = root / "data/train"
    test_root = root / "data/test"
    entries = []
    source_rows = {}
    test_source_rows = {}
    for dataset in DATASETS:
        raw_path = train_root / "sources" / f"{dataset}.jsonl"
        test_path = test_root / "sources" / f"{dataset}.jsonl"
        raw = read_jsonl(raw_path)
        test = read_jsonl(test_path)
        assert Counter(row["role"] for row in raw) == {"paired": 500, "calibration": 500}, dataset
        assert len(test) == 500 and all(row["role"] == "independent_test" for row in test), dataset
        assert {row["source_index"] for row in raw}.isdisjoint(
            row["source_index"] for row in test
        ), dataset
        source_rows[dataset], test_source_rows[dataset] = raw, test
        entries += [record(root, raw_path, 1000), record(root, test_path, 500)]
        audit_path = test_root / "sources" / f"{dataset}.audit.json"
        entries.append(record(root, audit_path))

    for model in MODELS:
        generation_manifest = train_root / "text" / model / "generation_manifest.json"
        entries.append(record(root, generation_manifest))
        for dataset in DATASETS:
            train_path = train_root / "text" / model / f"{dataset}.jsonl"
            test_path = test_root / "text" / model / f"{dataset}.jsonl"
            metadata_path = test_root / "text" / model / f"{dataset}.metadata.json"
            train, test = read_jsonl(train_path), read_jsonl(test_path)
            assert len(train) == len(test) == 500, (model, dataset)
            assert all(row["dataset"] == dataset for row in train + test), (model, dataset)
            assert all(row["generator_model"] == model for row in train + test), (model, dataset)
            assert all(row["human_text"] and row["machine_text"] and row["calibration_text"]
                       for row in train + test), (model, dataset)
            assert [normalize_text(row["calibration_text"]) for row in train] == [
                row["calibration_text"] for row in test
            ], (model, dataset, "calibration")
            assert [row["source_index"] for row in test] == [
                row["source_index"] for row in test_source_rows[dataset]
            ], (model, dataset, "source index")

            train_adapter_path = root / "data/adapters/train" / model / f"{dataset}.raw_data.json"
            test_adapter_path = root / "data/adapters/test" / model / f"{dataset}.raw_data.json"
            for path, rows in ((train_adapter_path, train), (test_adapter_path, test)):
                adapter = json.loads(path.read_text(encoding="utf-8"))
                assert adapter["original"] == [row["human_text"] for row in rows], path
                assert adapter["sampled"] == [row["machine_text"] for row in rows], path
                assert adapter["calibration"] == [row["calibration_text"] for row in rows], path

            fit_path = root / "data/fits/whitebox" / model / dataset / "results.json"
            assert fit_path.is_file(), fit_path
            entries += [record(root, train_path, 500), record(root, test_path, 500),
                        record(root, metadata_path), record(root, train_adapter_path),
                        record(root, test_adapter_path), record(root, fit_path)]

    fresh_root = root / "data/fresh"
    for dataset in DATASETS:
        split_rows = {}
        for role, directory in (("independent_test", "test"),
                                ("human_calibration", "calibration")):
            path = fresh_root / "sources" / directory / f"{dataset}.jsonl"
            rows = read_jsonl(path)
            assert len(rows) == 500 and all(row["role"] == role for row in rows), path
            split_rows[role] = rows
            entries.append(record(root, path, 500))
        assert {row["source_index"] for row in split_rows["independent_test"]}.isdisjoint(
            row["source_index"] for row in split_rows["human_calibration"]
        ), dataset
        for model in MODELS:
            path = fresh_root / "text" / model / f"{dataset}.jsonl"
            metadata = path.with_suffix(".metadata.json")
            rows = read_jsonl(path)
            assert len(rows) == 500, path
            assert [row["test_record_id"] for row in rows] == [
                row["record_id"] for row in split_rows["independent_test"]
            ], path
            assert [row["calibration_record_id"] for row in rows] == [
                row["record_id"] for row in split_rows["human_calibration"]
            ], path
            assert all(row["human_text"] and row["machine_text"] and row["calibration_text"]
                       for row in rows), path
            entries += [record(root, path, 500), record(root, metadata)]
    source_lock_path = fresh_root / "source_split_lock.json"
    source_lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    assert {cell["dataset"] for cell in source_lock["cells"]} == set(DATASETS)
    for cell in source_lock["cells"]:
        for role in ("test", "calibration"):
            path = fresh_root / "sources" / role / f"{cell['dataset']}.jsonl"
            assert cell[f"{role}_path"] == path.relative_to(root).as_posix()
            assert cell[f"{role}_sha256"] == digest(path), path
        for relative, expected_sha in cell["exclusion_file_sha256"].items():
            path = root / relative
            if path.is_file():
                assert digest(path) == expected_sha, path

    validation_lock_path = fresh_root / "validation_lock.json"
    validation_lock = json.loads(validation_lock_path.read_text(encoding="utf-8"))
    assert validation_lock["source_split_lock"] == source_lock_path.relative_to(root).as_posix()
    assert validation_lock["source_split_lock_sha256"] == digest(source_lock_path)
    for key, value in validation_lock["generated_files"].items():
        model, dataset = key.split("/")
        path = fresh_root / "text" / model / f"{dataset}.jsonl"
        metadata = path.with_suffix(".metadata.json")
        assert value["path"] == path.relative_to(root).as_posix()
        assert value["metadata_path"] == metadata.relative_to(root).as_posix()
        assert value["sha256"] == digest(path)
        assert value["metadata_sha256"] == digest(metadata)
    entries.append(record(root, source_lock_path))
    entries.append(record(root, validation_lock_path))

    entries.sort(key=lambda item: item["path"])
    return {
        "description": "Main and fresh-confirmation paper text inputs plus frozen white-box fit metadata",
        "models": list(MODELS),
        "datasets": list(DATASETS),
        "train_pairs_per_configuration": 500,
        "independent_calibration_humans_per_configuration": 500,
        "independent_test_pairs_per_configuration": 500,
        "corrected_llama_test_root": "data/test/text/llama-2-7b",
        "n_files": len(entries),
        "total_bytes": sum(item["bytes"] for item in entries),
        "files": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Update data/manifests/paper_data_manifest.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    current = audit(root)
    output = root / "data/manifests/paper_data_manifest.json"
    if args.write:
        output.write_text(json.dumps(current, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        expected = json.loads(output.read_text(encoding="utf-8"))
        assert current == expected, "Paper data differ from the checked-in manifest; use --write after review"
    print(f"validated {current['n_files']} files, {current['total_bytes']:,} bytes")


if __name__ == "__main__":
    main()
