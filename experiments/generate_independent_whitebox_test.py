#!/usr/bin/env python3
"""
Build and generate a strictly independent white-box test set for Water.

The script has three subcommands:

1. ``select`` chooses 500 new source documents for each domain while excluding
   every source index and human/calibration-text prefix found in the existing
   development data.
2. ``generate`` loads one source model, generates machine continuations for the
   frozen source split, copies the *existing* 500 calibration humans, and writes
   a ``generated_strict``-compatible JSONL file.
3. ``validate`` checks counts, uniqueness, calibration identity, and overlap
   against the old development data.

Run this file from the root of this repository. All default
source loaders are offline and point to the original local source pools under
the directory supplied through ``EC_DETECT_DATA_ROOT``:

* C4: ``MarkLLM/dataset/c4/c4.json``, field ``text``
* XSum: ``datasets/xsum/train-*.parquet``, field ``document``
* SQuAD: ``datasets/squad/train-*.parquet``, field ``context``
* WritingPrompts: ``AdaDetectGPT-main/data/writingPrompts/valid.wp_{source,target}``
* PubMedQA: ``datasets/pubmedqa/pqa_unlabeled/train-*.parquet``, field
  ``context["contexts"]``

If the old experiment used different dataset revisions or source pools, pass a
JSON manifest through ``--source-manifest``.  The selector deliberately fails
instead of silently changing the source distribution when fewer than 500
unused eligible documents remain.

Example
-------

Select the shared frozen test prompts:

    python experiments/generate_independent_whitebox_test.py select \
      --datasets c4 squad xsum writing pubmed \
      --existing-raw-root data/train/sources \
      --existing-generated-root data/train/text \
      --output-root data/test/sources \
      --n-samples 500 \
      --seed 20260729

Generate one model/dataset cell.  Use the SAME generation settings as the old
experiment; the mutually exclusive sampling-mode flag is required on purpose:

    CUDA_VISIBLE_DEVICES=0 python experiments/generate_independent_whitebox_test.py generate \
      --dataset c4 \
      --model-label llama-2-7b \
      --source-root data/test/sources \
      --old-generated-root data/train/text \
      --output-root data/test/text \
      --prompt-tokens 120 \
      --max-length 200 \
      --do-top-k --top-k 50 \
      --seed 20260729

Validate all completed cells:

    python experiments/generate_independent_whitebox_test.py validate \
      --datasets c4 squad xsum writing pubmed \
      --models llama-2-7b opt-2.7b opt-6.7b falcon-7b \
      --existing-raw-root data/train/sources \
      --existing-generated-root data/train/text \
      --source-root data/test/sources \
      --new-generated-root data/test/text \
      --n-samples 500
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import heapq
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


SEPARATOR = "<<<SEP>>>"
DATASETS = ("c4", "squad", "xsum", "writing", "pubmed")
MODELS = ("llama-2-7b", "opt-2.7b", "opt-6.7b", "falcon-7b")
DEFAULT_MODEL_PATHS = {
    "llama-2-7b": Path("models/llama-2-7b"),
    "opt-2.7b": Path("models/opt-2.7b"),
    "opt-6.7b": Path("models/opt-6.7b"),
    "falcon-7b": Path("models/falcon-7b"),
}
TEXT_PREFIX_WORDS = 48
LOCAL_DATA_ROOT = Path(os.environ.get("EC_DETECT_DATA_ROOT", "data/raw"))


DEFAULT_SOURCE_MANIFEST: dict[str, dict[str, Any]] = {
    "c4": {
        "kind": "json",
        "path": str(LOCAL_DATA_ROOT / "MarkLLM/dataset/c4/c4.json"),
        "fields": ["text", "natural_text"],
        "id_fields": ["id", "document_id", "url"],
        "min_words": 250,
    },
    "xsum": {
        "kind": "parquet",
        "data_files": str(LOCAL_DATA_ROOT / "datasets/xsum/train-*.parquet"),
        "fields": ["document"],
        "id_fields": ["id", "document_id"],
        "min_words": 250,
    },
    "squad": {
        "kind": "parquet",
        "data_files": str(LOCAL_DATA_ROOT / "datasets/squad/train-*.parquet"),
        "fields": ["context"],
        "id_fields": ["id"],
        # Match the original Water raw-source protocol: deduplicate SQuAD by
        # context and retain every non-empty context.  A 250-word cutoff is far
        # too restrictive for SQuAD paragraphs and leaves fewer than 500
        # examples after excluding the old paired/calibration split.
        "min_words": 1,
    },
    "writing": {
        "kind": "writing_prompts",
        "source_file": str(
            LOCAL_DATA_ROOT
            / "AdaDetectGPT-main/data/writingPrompts/valid.wp_source"
        ),
        "target_file": str(
            LOCAL_DATA_ROOT
            / "AdaDetectGPT-main/data/writingPrompts/valid.wp_target"
        ),
        "min_words": 250,
    },
    "pubmed": {
        "kind": "pubmed_qa_contexts",
        "data_files": str(
            LOCAL_DATA_ROOT
            / "datasets/pubmedqa/pqa_unlabeled/train-*.parquet"
        ),
        "field": "context.contexts",
        "id_fields": ["pubid", "id"],
        "min_words": 150,
    },
}


@dataclass(frozen=True)
class SourceExample:
    source_index: int
    source_id: str
    source_text: str


@dataclass(frozen=True)
class GenerationConfig:
    prompt_tokens: int
    min_length: int
    max_length: int
    min_generated_words: int
    batch_size: int
    sampling_mode: str
    temperature: float | None
    top_p: float | None
    top_k: int | None
    seed: int


def normalize_text(text: str) -> str:
    return " ".join(str(text).split())


def fingerprint_text(text: str) -> str:
    # PubMed source rows contain a private question/answer separator that is
    # removed in final human_text. Ignore it when checking split overlap.
    return normalize_text(str(text).replace(SEPARATOR, " "))


def prefix_text(text: str, n_words: int = TEXT_PREFIX_WORDS) -> str:
    return " ".join(fingerprint_text(text).split()[:n_words])


def text_hash(text: str) -> str:
    return hashlib.sha256(fingerprint_text(text).encode("utf-8")).hexdigest()


def prefix_hash(text: str, n_words: int = TEXT_PREFIX_WORDS) -> str:
    return hashlib.sha256(prefix_text(text, n_words).encode("utf-8")).hexdigest()


def stable_priority(seed: int, dataset: str, source_id: str, text: str) -> int:
    payload = f"{seed}\0{dataset}\0{source_id}\0{prefix_hash(text)}"
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest(), 16)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def iter_nested_items(value: Any, parent_key: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_string = str(key)
            yield key_string, child
            yield from iter_nested_items(child, key_string)
    elif isinstance(value, list):
        for child in value:
            yield from iter_nested_items(child, parent_key)


def collect_exclusions(paths: Sequence[Path]) -> tuple[set[int], set[str], set[str]]:
    """Return used source indices, normalized full hashes, and prefix hashes."""
    used_indices: set[int] = set()
    full_hashes: set[str] = set()
    prefix_hashes: set[str] = set()

    index_keys = {
        "source_index",
        "paired_source_index",
        "human_source_index",
        "calibration_source_index",
        "test_source_index",
    }
    text_key_fragments = (
        "human",
        "calibration",
        "source_text",
        "raw_text",
        "document",
        "context",
    )

    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            for key, value in iter_nested_items(row):
                key_lower = key.lower()
                if key_lower in index_keys and isinstance(value, int):
                    used_indices.add(value)
                if (
                    isinstance(value, str)
                    and len(normalize_text(value).split()) >= 8
                    and (
                        key_lower in {"text", "prompt_text"}
                        or any(
                            fragment in key_lower
                            for fragment in text_key_fragments
                        )
                    )
                ):
                    full_hashes.add(text_hash(value))
                    prefix_hashes.add(prefix_hash(value))
    return used_indices, full_hashes, prefix_hashes


def load_manifest(path: Path | None) -> dict[str, dict[str, Any]]:
    manifest = json.loads(json.dumps(DEFAULT_SOURCE_MANIFEST))
    if path is None:
        return manifest
    with path.open("r", encoding="utf-8") as handle:
        override = json.load(handle)
    if not isinstance(override, dict):
        raise ValueError("--source-manifest must contain a JSON object")
    for dataset, spec in override.items():
        if dataset not in DATASETS:
            raise ValueError(f"unknown dataset in source manifest: {dataset}")
        if not isinstance(spec, dict):
            raise ValueError(f"manifest entry {dataset!r} must be an object")
        manifest[dataset] = {**manifest.get(dataset, {}), **spec}
    return manifest


def process_writing_prompt(text: str) -> str:
    return text.replace("[ WP ]", "").replace("[ OT ]", "")


def process_writing_spaces(text: str) -> str:
    replacements = (
        (" ,", ","),
        (" .", "."),
        (" ?", "?"),
        (" !", "!"),
        (" ;", ";"),
        (" \'", "'"),
        (" ’ ", "'"),
        (" :", ":"),
        ("<newline>", "\n"),
        ("`` ", '"'),
        (" \'\'", '"'),
        ("\'\'", '"'),
        (".. ", "... "),
        (" )", ")"),
        ("( ", "("),
        (" n\'t", "n't"),
        (" i ", " I "),
        (" i\'", " I'"),
        ("\\'", "'"),
        ("\n ", "\n"),
        ("\u2019", "'"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text.strip()


def iter_writing_prompts(spec: Mapping[str, Any]) -> Iterator[SourceExample]:
    source_path = Path(str(spec["source_file"]))
    target_path = Path(str(spec["target_file"]))
    if not source_path.exists() or not target_path.exists():
        raise FileNotFoundError(
            "WritingPrompts files were not found. Expected "
            f"{source_path} and {target_path}; override them in --source-manifest."
        )
    # Keep this form compatible with Python 3.8/3.9. Parenthesized context
    # managers (``with (...):``) require Python 3.10 or newer.
    with source_path.open("r", encoding="utf-8") as source_handle, \
            target_path.open("r", encoding="utf-8") as target_handle:
        for index, (prompt, story) in enumerate(zip(source_handle, target_handle)):
            prompt = process_writing_prompt(prompt)
            text = process_writing_spaces(f"{prompt} {story}")
            if "nsfw" in text.lower():
                continue
            yield SourceExample(index, str(index), normalize_text(text))


def nested_value(row: Mapping[str, Any], dotted_field: str) -> Any:
    value: Any = row
    for part in dotted_field.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def local_files(pattern: str) -> list[Path]:
    paths = [Path(path) for path in sorted(glob.glob(pattern))]
    if not paths:
        raise FileNotFoundError(
            f"no local source files match {pattern!r}; "
            "the selector never downloads datasets"
        )
    return paths


def iter_json_rows(path: Path) -> Iterator[Mapping[str, Any]]:
    """Read either JSONL/NDJSON or a JSON array/object from a local file."""
    if not path.exists():
        raise FileNotFoundError(
            f"local JSON source not found: {path}; "
            "the selector never downloads datasets"
        )

    # C4's local file is normally JSONL even though its suffix is .json.
    # Detect a parseable first record and then stream the remaining records.
    with path.open("r", encoding="utf-8") as handle:
        first_line = ""
        for line in handle:
            if line.strip():
                first_line = line
                break
        if first_line:
            try:
                first_value = json.loads(first_line)
            except json.JSONDecodeError:
                first_value = None
            if isinstance(first_value, Mapping):
                yield first_value
                for line_number, line in enumerate(handle, start=2):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"{path}:{line_number}: invalid JSONL record: {exc}"
                        ) from exc
                    if not isinstance(value, Mapping):
                        raise ValueError(
                            f"{path}:{line_number}: expected a JSON object"
                        )
                    yield value
                return
            if isinstance(first_value, list):
                for row in first_value:
                    if not isinstance(row, Mapping):
                        raise ValueError(
                            f"{path}: every source record must be a JSON object"
                        )
                    yield row
                return

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        rows = value
    elif isinstance(value, Mapping):
        rows = next(
            (
                child
                for key in ("data", "train", "records", "examples")
                if isinstance((child := value.get(key)), list)
            ),
            [value],
        )
    else:
        raise ValueError(f"{path}: expected JSONL, a JSON array, or a JSON object")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{path}: every source record must be a JSON object")
        yield row


def iter_parquet_rows(pattern: str) -> Iterator[Mapping[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "reading local Parquet files requires pyarrow; install it in the "
            "current environment with `pip install pyarrow`"
        ) from exc

    for path in local_files(pattern):
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                if isinstance(row, Mapping):
                    yield row


def text_from_row(row: Mapping[str, Any], fields: Sequence[str]) -> str:
    for field in fields:
        value = nested_value(row, field)
        if isinstance(value, str):
            text = normalize_text(value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            text = normalize_text(
                " ".join(item for item in value if isinstance(item, str))
            )
        else:
            text = ""
        if text:
            return text
    return ""


def iter_local_records(spec: Mapping[str, Any]) -> Iterator[SourceExample]:
    kind = str(spec["kind"])
    if kind == "json":
        rows = iter_json_rows(Path(str(spec["path"])))
    elif kind in {"parquet", "pubmed_qa_contexts"}:
        rows = iter_parquet_rows(str(spec["data_files"]))
    else:
        raise ValueError(f"unsupported local record kind: {kind}")

    if kind == "pubmed_qa_contexts":
        fields = [str(spec.get("field", "context.contexts"))]
    else:
        configured_fields = spec.get("fields", [spec.get("field", "text")])
        fields = [str(field) for field in configured_fields]
    id_fields = [str(field) for field in spec.get(
        "id_fields",
        ("id", "document_id", "url"),
    )]

    for index, row in enumerate(rows):
        text = text_from_row(row, fields)
        if not text:
            continue
        source_id = next(
            (
                str(value)
                for field in id_fields
                if (value := nested_value(row, field)) is not None
            ),
            str(index),
        )
        yield SourceExample(index, source_id, text)


def iter_source_examples(
    dataset: str,
    spec: Mapping[str, Any],
    cache_dir: str | None,
) -> Iterator[SourceExample]:
    del cache_dir  # Kept as a deprecated CLI option for command compatibility.
    kind = str(spec.get("kind", "json"))
    if kind == "writing_prompts":
        yield from iter_writing_prompts(spec)
    elif kind in {"json", "parquet", "pubmed_qa_contexts"}:
        yield from iter_local_records(spec)
    else:
        raise ValueError(f"unsupported source kind {kind!r} for {dataset}")


def old_paths_for_dataset(
    dataset: str,
    existing_raw_root: Path,
    existing_generated_root: Path,
) -> list[Path]:
    paths = [existing_raw_root / f"{dataset}.jsonl"]
    for model in MODELS:
        paths.append(existing_generated_root / model / f"{dataset}.jsonl")
    return paths


def select_dataset(args: argparse.Namespace, dataset: str) -> Path:
    manifest = load_manifest(args.source_manifest)
    spec = manifest[dataset]
    old_paths = old_paths_for_dataset(
        dataset,
        args.existing_raw_root,
        args.existing_generated_root,
    )
    if not any(path.exists() for path in old_paths):
        raise FileNotFoundError(
            f"no old data found for {dataset}; refusing to build an unverified split"
        )

    used_indices, old_full_hashes, old_prefix_hashes = collect_exclusions(old_paths)
    min_words = int(spec.get("min_words", 1))
    max_source_records = spec.get("max_source_records")
    max_source_records = (
        int(max_source_records) if max_source_records is not None else None
    )

    # Keep only the n smallest deterministic priorities. This avoids retaining
    # hundreds of thousands of long C4 documents in memory.
    candidates: list[tuple[int, int, SourceExample]] = []
    seen_candidate_full_hashes: set[str] = set()
    scanned = 0
    excluded_by_index = 0
    excluded_by_text = 0
    excluded_too_short = 0

    for example in iter_source_examples(dataset, spec, args.cache_dir):
        scanned += 1
        if max_source_records is not None and scanned > max_source_records:
            break
        if example.source_index in used_indices:
            excluded_by_index += 1
            continue
        words = normalize_text(example.source_text).split()
        if len(words) < min_words:
            excluded_too_short += 1
            continue
        full = text_hash(example.source_text)
        prefix = prefix_hash(example.source_text)
        if (
            full in old_full_hashes
            or prefix in old_prefix_hashes
            or full in seen_candidate_full_hashes
        ):
            excluded_by_text += 1
            continue
        seen_candidate_full_hashes.add(full)
        priority = stable_priority(
            args.seed,
            dataset,
            example.source_id,
            example.source_text,
        )
        heap_item = (-priority, scanned, example)
        if len(candidates) < args.n_samples:
            heapq.heappush(candidates, heap_item)
        elif priority < -candidates[0][0]:
            heapq.heapreplace(candidates, heap_item)

    selected = [
        example
        for _, _, example in sorted(
            candidates,
            key=lambda item: (-item[0], item[1]),
        )
    ]
    if len(selected) != args.n_samples:
        raise RuntimeError(
            f"{dataset}: found only {len(selected)} unused eligible documents; "
            f"need {args.n_samples}. scanned={scanned}, "
            f"excluded_index={excluded_by_index}, "
            f"excluded_text={excluded_by_text}, "
            f"excluded_too_short={excluded_too_short}, "
            f"min_words={min_words}. Do not silently switch datasets. Supply the "
            "same larger source pool/revision used by the old experiment through "
            "--source-manifest."
        )

    rows: list[dict[str, Any]] = []
    for test_index, example in enumerate(selected):
        rows.append(
            {
                "record_id": f"{dataset}-test-{test_index:04d}",
                "dataset": dataset,
                "role": "independent_test",
                "test_index": test_index,
                "source_index": example.source_index,
                "source_id": example.source_id,
                "source_text": example.source_text,
                "source_text_sha256": text_hash(example.source_text),
                "source_prefix_sha256": prefix_hash(example.source_text),
                "split_seed": args.seed,
            }
        )

    output_path = args.output_root / f"{dataset}.jsonl"
    if output_path.exists() and not args.overwrite:
        existing = read_jsonl(output_path)
        if existing != rows:
            raise FileExistsError(
                f"{output_path} exists with different content; pass --overwrite "
                "only before the test set is used"
            )
    else:
        atomic_write_jsonl(output_path, rows)

    audit = {
        "dataset": dataset,
        "n_selected": len(rows),
        "seed": args.seed,
        "prefix_words": TEXT_PREFIX_WORDS,
        "source_spec": spec,
        "old_files_checked": [str(path) for path in old_paths if path.exists()],
        "used_source_indices": len(used_indices),
        "old_full_text_hashes": len(old_full_hashes),
        "old_prefix_hashes": len(old_prefix_hashes),
        "source_records_scanned": scanned,
        "excluded_by_source_index": excluded_by_index,
        "excluded_by_text_fingerprint": excluded_by_text,
        "excluded_too_short": excluded_too_short,
        "selected_manifest_sha256": text_hash(
            "\n".join(row["source_text_sha256"] for row in rows)
        ),
    }
    atomic_write_json(output_path.with_suffix(".audit.json"), audit)
    print(
        f"[select] {dataset}: wrote {len(rows)} records to {output_path}; "
        f"scanned={scanned}, excluded_index={excluded_by_index}, "
        f"excluded_text={excluded_by_text}"
    )
    return output_path


def calibration_texts_from_old_file(path: Path, expected: int) -> list[str]:
    rows = read_jsonl(path)
    texts: list[str] = []
    for row in rows:
        value = row.get("calibration_text")
        if isinstance(value, str) and normalize_text(value):
            texts.append(normalize_text(value))
    if len(texts) != expected:
        raise ValueError(
            f"{path}: expected exactly {expected} calibration_text values, "
            f"found {len(texts)}"
        )
    if len({text_hash(text) for text in texts}) != expected:
        raise ValueError(f"{path}: calibration_text contains duplicates")
    return texts


def model_dtype(name: str):
    import torch

    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def prepare_human_and_prompt(
    source_text: str,
    dataset: str,
    tokenizer: Any,
    prompt_tokens: int,
) -> tuple[str, str]:
    human = normalize_text(source_text)
    encoded = tokenizer(
        human,
        add_special_tokens=False,
        truncation=True,
        max_length=prompt_tokens,
    )["input_ids"]
    if not encoded:
        raise ValueError("empty prompt after tokenization")
    prompt = tokenizer.decode(encoded, skip_special_tokens=True)
    return human, normalize_text(prompt)


def generation_kwargs(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    if args.exact_cond_prob:
        return "exact_cond_prob", {"temperature": 1.0, "top_p": 1.0, "top_k": 0}
    if args.do_top_p:
        return "top_p", {"top_p": args.top_p}
    if args.do_top_k:
        return "top_k", {"top_k": args.top_k}
    if args.do_temperature:
        return "temperature", {"temperature": args.temperature}
    raise AssertionError("argparse should require one sampling mode")


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_one(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_length: int,
    min_length: int,
    sampling_kwargs: Mapping[str, Any],
    seed: int,
    max_attempts: int,
    min_words: int,
) -> str:
    import torch

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_length - 1,
    )
    input_length = int(inputs["input_ids"].shape[1])
    if input_length >= max_length:
        raise ValueError(
            f"prompt has {input_length} tokens but max_length is {max_length}"
        )
    device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    effective_min_length = max(input_length + 1, min(min_length, max_length))

    for attempt in range(max_attempts):
        seed_everything(seed + attempt)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                min_length=effective_min_length,
                max_length=max_length,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **dict(sampling_kwargs),
            )
        decoded = normalize_text(tokenizer.decode(output[0], skip_special_tokens=True))
        if len(decoded.split()) >= min_words:
            return decoded
    raise RuntimeError(
        f"generation did not reach {min_words} words after {max_attempts} attempts"
    )


def generate_batch(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    max_length: int,
    min_length: int,
    sampling_kwargs: Mapping[str, Any],
    batch_seed: int,
) -> list[str]:
    import torch

    inputs = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length - 1,
    )
    if int(inputs["input_ids"].shape[1]) >= max_length:
        raise ValueError("at least one prompt leaves no room for generation")
    device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    seed_everything(batch_seed)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            min_length=min(min_length, max_length),
            max_length=max_length,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **dict(sampling_kwargs),
        )
    return [
        normalize_text(tokenizer.decode(output, skip_special_tokens=True))
        for output in outputs
    ]


def trim_pair(human: str, machine: str) -> tuple[str, str]:
    human_words = normalize_text(human).split()
    machine_words = normalize_text(machine).split()
    shorter = min(len(human_words), len(machine_words))
    if shorter == 0:
        raise ValueError("cannot trim an empty human/machine pair")
    return " ".join(human_words[:shorter]), " ".join(machine_words[:shorter])


def load_existing_output_for_resume(
    output_path: Path,
    source_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not output_path.exists():
        return []
    rows = read_jsonl(output_path)
    if len(rows) > len(source_rows):
        raise ValueError(f"{output_path}: contains more rows than the source split")
    for index, row in enumerate(rows):
        if row.get("record_id") != source_rows[index].get("record_id"):
            raise ValueError(
                f"{output_path}: row {index} does not match the frozen source split"
            )
    return rows


def generate_dataset(args: argparse.Namespace) -> Path:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = args.model_path or DEFAULT_MODEL_PATHS[args.model_label]
    if not model_path.exists():
        raise FileNotFoundError(
            f"model path not found: {model_path}. "
            "Pass --model-path to override the built-in Water model path."
        )

    source_path = args.source_root / f"{args.dataset}.jsonl"
    if not source_path.exists():
        raise FileNotFoundError(
            f"{source_path} not found; run the select subcommand first"
        )
    source_rows = read_jsonl(source_path)
    if len(source_rows) != args.n_samples:
        raise ValueError(
            f"{source_path}: expected {args.n_samples} source rows, "
            f"found {len(source_rows)}"
        )

    old_path = (
        args.old_generated_root / args.model_label / f"{args.dataset}.jsonl"
    )
    calibration_texts = calibration_texts_from_old_file(old_path, args.n_samples)

    sampling_mode, sampling_kwargs = generation_kwargs(args)
    min_generated_words = (
        args.min_generated_words
        if args.min_generated_words is not None
        else 55
    )
    min_length = args.min_length
    config = GenerationConfig(
        prompt_tokens=args.prompt_tokens,
        min_length=min_length,
        max_length=args.max_length,
        min_generated_words=min_generated_words,
        batch_size=args.batch_size,
        sampling_mode=sampling_mode,
        temperature=sampling_kwargs.get("temperature"),
        top_p=sampling_kwargs.get("top_p"),
        top_k=sampling_kwargs.get("top_k"),
        seed=args.seed,
    )

    output_path = (
        args.output_root / args.model_label / f"{args.dataset}.jsonl"
    )
    if output_path.exists() and args.overwrite:
        output_path.unlink()
    completed = load_existing_output_for_resume(output_path, source_rows)
    if len(completed) == args.n_samples:
        print(f"[generate] {output_path} is already complete")
        return output_path

    print(f"[generate] loading tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=not args.slow_tokenizer,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"[generate] loading model: {model_path}")
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
        "torch_dtype": model_dtype(args.dtype),
        "local_files_only": True,
    }
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    if not args.device_map:
        device = torch.device(args.device)
        model.to(device)
    model.eval()

    rows = list(completed)
    while len(rows) < args.n_samples:
        start = len(rows)
        stop = min(start + args.batch_size, args.n_samples)
        batch_sources = source_rows[start:stop]
        prepared = [
            prepare_human_and_prompt(
                str(source_row["source_text"]),
                args.dataset,
                tokenizer,
                args.prompt_tokens,
            )
            for source_row in batch_sources
        ]
        humans = [item[0] for item in prepared]
        prompts = [item[1] for item in prepared]
        machines = generate_batch(
            model,
            tokenizer,
            prompts,
            max_length=args.max_length,
            min_length=min_length,
            sampling_kwargs=sampling_kwargs,
            batch_seed=args.seed + start * 10_000,
        )

        # Retry only unusually short outputs. This preserves batched throughput
        # while retaining the same minimum-word criterion as the old builder.
        for offset, machine in enumerate(machines):
            if len(machine.split()) >= min_generated_words:
                continue
            index = start + offset
            machines[offset] = generate_one(
                model,
                tokenizer,
                prompts[offset],
                max_length=args.max_length,
                min_length=min_length,
                sampling_kwargs=sampling_kwargs,
                seed=args.seed + index * 10_000,
                max_attempts=args.max_attempts,
                min_words=min_generated_words,
            )

        for offset, (source_row, human, machine) in enumerate(
            zip(batch_sources, humans, machines)
        ):
            index = start + offset
            human, machine = trim_pair(human, machine)
            rows.append(
                {
                    "record_id": source_row["record_id"],
                    "dataset": args.dataset,
                    "role": "independent_test",
                    "test_index": index,
                    "source_index": source_row["source_index"],
                    "source_id": source_row["source_id"],
                    "source_text_sha256": source_row["source_text_sha256"],
                    "human_text": human,
                    "machine_text": machine,
                    # Reuse the untouched calibration set. It must not be
                    # regenerated or replaced by a new random sample.
                    "calibration_text": calibration_texts[index],
                    "generator_model": args.model_label,
                    "generation": asdict(config),
                }
            )
        atomic_write_jsonl(output_path, rows)
        if len(rows) % args.log_every < args.batch_size or len(rows) == args.n_samples:
            print(
                f"[generate] {args.model_label}/{args.dataset}: "
                f"{len(rows)}/{args.n_samples}"
            )

    metadata = {
        "schema_version": 1,
        "purpose": "independent_frozen_whitebox_test",
        "model_label": args.model_label,
        "model_path": str(model_path),
        "dataset": args.dataset,
        "n_samples": args.n_samples,
        "source_file": str(source_path),
        "old_calibration_file": str(old_path),
        "generation": asdict(config),
        "calibration_manifest_sha256": text_hash(
            "\n".join(text_hash(text) for text in calibration_texts)
        ),
    }
    atomic_write_json(output_path.with_suffix(".metadata.json"), metadata)
    return output_path


def ensure_unique(values: Sequence[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicates")


def validate_cell(
    *,
    dataset: str,
    model: str,
    n_samples: int,
    source_root: Path,
    existing_raw_root: Path,
    existing_generated_root: Path,
    new_generated_root: Path,
) -> None:
    source_path = source_root / f"{dataset}.jsonl"
    old_generated_path = existing_generated_root / model / f"{dataset}.jsonl"
    new_path = new_generated_root / model / f"{dataset}.jsonl"
    for path in (source_path, old_generated_path, new_path):
        if not path.exists():
            raise FileNotFoundError(path)

    source_rows = read_jsonl(source_path)
    new_rows = read_jsonl(new_path)
    if len(source_rows) != n_samples or len(new_rows) != n_samples:
        raise ValueError(
            f"{model}/{dataset}: expected {n_samples} rows; "
            f"source={len(source_rows)}, generated={len(new_rows)}"
        )

    old_paths = old_paths_for_dataset(
        dataset,
        existing_raw_root,
        existing_generated_root,
    )
    used_indices, old_full_hashes, old_prefix_hashes = collect_exclusions(old_paths)
    new_source_indices = [int(row["source_index"]) for row in source_rows]
    new_source_full_hashes = [
        text_hash(str(row["source_text"])) for row in source_rows
    ]
    new_source_prefix_hashes = [
        prefix_hash(str(row["source_text"])) for row in source_rows
    ]
    ensure_unique([str(index) for index in new_source_indices], "new source indices")
    ensure_unique(new_source_full_hashes, "new source texts")
    if used_indices.intersection(new_source_indices):
        raise ValueError(f"{model}/{dataset}: source_index overlap with old data")
    if old_full_hashes.intersection(new_source_full_hashes):
        raise ValueError(f"{model}/{dataset}: full-text overlap with old data")
    if old_prefix_hashes.intersection(new_source_prefix_hashes):
        raise ValueError(f"{model}/{dataset}: text-prefix overlap with old data")

    old_calibration = calibration_texts_from_old_file(
        old_generated_path,
        n_samples,
    )
    new_calibration = [normalize_text(str(row["calibration_text"])) for row in new_rows]
    if old_calibration != new_calibration:
        raise ValueError(
            f"{model}/{dataset}: new file does not preserve the old calibration set"
        )

    for index, (source_row, new_row) in enumerate(zip(source_rows, new_rows)):
        if source_row["record_id"] != new_row.get("record_id"):
            raise ValueError(f"{model}/{dataset}: record_id mismatch at row {index}")
        if new_row.get("role") != "independent_test":
            raise ValueError(f"{model}/{dataset}: wrong role at row {index}")
        if not normalize_text(str(new_row.get("human_text", ""))):
            raise ValueError(f"{model}/{dataset}: empty human_text at row {index}")
        if not normalize_text(str(new_row.get("machine_text", ""))):
            raise ValueError(f"{model}/{dataset}: empty machine_text at row {index}")

    ensure_unique(
        [text_hash(str(row["human_text"])) for row in new_rows],
        f"{model}/{dataset} human_text",
    )
    ensure_unique(
        [text_hash(str(row["machine_text"])) for row in new_rows],
        f"{model}/{dataset} machine_text",
    )
    print(f"[validate] PASS {model}/{dataset}")


def add_common_dataset_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a frozen independent white-box test set for Water"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    select_parser = subparsers.add_parser(
        "select",
        help="select new non-overlapping human source documents",
    )
    add_common_dataset_argument(select_parser)
    select_parser.add_argument(
        "--existing-raw-root",
        type=Path,
        default=Path("data/train/sources"),
    )
    select_parser.add_argument(
        "--existing-generated-root",
        type=Path,
        default=Path("data/train/text"),
    )
    select_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/test/sources"),
    )
    select_parser.add_argument("--source-manifest", type=Path)
    select_parser.add_argument(
        "--cache-dir",
        help="deprecated compatibility option; all source datasets are local",
    )
    select_parser.add_argument("--n-samples", type=int, default=500)
    select_parser.add_argument("--seed", type=int, default=20260729)
    select_parser.add_argument("--overwrite", action="store_true")

    generate_parser = subparsers.add_parser(
        "generate",
        help="generate one model/dataset cell from the frozen source split",
    )
    generate_parser.add_argument("--dataset", choices=DATASETS, required=True)
    generate_parser.add_argument("--model-label", choices=MODELS, required=True)
    generate_parser.add_argument(
        "--model-path",
        type=Path,
        help=(
            "local model directory; defaults by --model-label to the original "
            "local model paths or Hugging Face model identifiers"
        ),
    )
    generate_parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/test/sources"),
    )
    generate_parser.add_argument(
        "--old-generated-root",
        type=Path,
        default=Path("data/train/text"),
    )
    generate_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/test/text"),
    )
    generate_parser.add_argument("--n-samples", type=int, default=500)
    generate_parser.add_argument("--prompt-tokens", type=int, required=True)
    generate_parser.add_argument(
        "--min-length",
        type=int,
        default=150,
        help="minimum total generated sequence length in tokens",
    )
    generate_parser.add_argument("--max-length", type=int, required=True)
    generate_parser.add_argument("--min-generated-words", type=int)
    sampling = generate_parser.add_mutually_exclusive_group(required=True)
    sampling.add_argument("--exact-cond-prob", action="store_true")
    sampling.add_argument("--do-top-p", action="store_true")
    sampling.add_argument("--do-top-k", action="store_true")
    sampling.add_argument("--do-temperature", action="store_true")
    generate_parser.add_argument("--top-p", type=float, default=0.96)
    generate_parser.add_argument("--top-k", type=int, default=40)
    generate_parser.add_argument("--temperature", type=float, default=0.8)
    generate_parser.add_argument("--seed", type=int, default=20260729)
    generate_parser.add_argument("--batch-size", type=int, default=8)
    generate_parser.add_argument("--max-attempts", type=int, default=20)
    generate_parser.add_argument("--log-every", type=int, default=10)
    generate_parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    generate_parser.add_argument("--device", default="cuda")
    generate_parser.add_argument("--device-map")
    generate_parser.add_argument("--trust-remote-code", action="store_true")
    generate_parser.add_argument("--slow-tokenizer", action="store_true")
    generate_parser.add_argument("--overwrite", action="store_true")

    validate_parser = subparsers.add_parser(
        "validate",
        help="validate completed test cells and prove split/calibration separation",
    )
    add_common_dataset_argument(validate_parser)
    validate_parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=list(MODELS),
    )
    validate_parser.add_argument("--n-samples", type=int, default=500)
    validate_parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/test/sources"),
    )
    validate_parser.add_argument(
        "--existing-raw-root",
        type=Path,
        default=Path("data/train/sources"),
    )
    validate_parser.add_argument(
        "--existing-generated-root",
        type=Path,
        default=Path("data/train/text"),
    )
    validate_parser.add_argument(
        "--new-generated-root",
        type=Path,
        default=Path("data/test/text"),
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if getattr(args, "n_samples", 1) <= 0:
        raise ValueError("--n-samples must be positive")
    if args.command == "generate":
        if args.prompt_tokens <= 0:
            raise ValueError("--prompt-tokens must be positive")
        if args.max_length <= args.prompt_tokens:
            raise ValueError("--max-length must exceed --prompt-tokens")
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be positive")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    if args.command == "select":
        for dataset in args.datasets:
            select_dataset(args, dataset)
        print(
            "\nThe source split is now frozen. Commit/archive "
            "test_raw_sources/*.jsonl and do not rerun selection after viewing "
            "test results."
        )
    elif args.command == "generate":
        output = generate_dataset(args)
        print(f"[generate] complete: {output}")
    elif args.command == "validate":
        for model in args.models:
            for dataset in args.datasets:
                validate_cell(
                    dataset=dataset,
                    model=model,
                    n_samples=args.n_samples,
                    source_root=args.source_root,
                    existing_raw_root=args.existing_raw_root,
                    existing_generated_root=args.existing_generated_root,
                    new_generated_root=args.new_generated_root,
                )
        print("[validate] all requested cells passed")
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
