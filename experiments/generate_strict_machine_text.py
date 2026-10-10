#!/usr/bin/env python3
"""
Generate strict Table-1-style human/machine pairs from raw_sources/*.jsonl.

Output layout:
  generated_strict/<model>/<dataset>.jsonl

Protocol:
  - 500 paired source records per dataset
  - 500 independent human calibration records per dataset
  - 120-token human prefix
  - 150-200 total tokens
  - raw prefix continuation, without chat template
  - multinomial sampling
  - human/calibration text matched to machine length

The script is resumable. Each completed batch is written atomically.

Tokenizer note:
  Retokenizing decode(prompt_ids) and decode(full_ids) need not preserve an
  exact token-prefix relation for byte-level BPE tokenizers such as OPT.
  Therefore character-level prefix consistency is the hard integrity check;
  retokenized token-prefix consistency is diagnostic only.
"""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_PATHS = {
    "llama-2-7b": "models/llama-2-7b",
    "opt-2.7b": "models/opt-2.7b",
    "opt-6.7b": "models/opt-6.7b",
    "falcon-7b": "models/falcon-7b",
}

DEFAULT_DATASETS = ["c4", "squad", "xsum", "writing", "pubmed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/raw"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/generated"),
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated dataset names.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt-tokens", type=int, default=120)
    parser.add_argument("--min-total-tokens", type=int, default=150)
    parser.add_argument("--max-total-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional smoke-test limit per dataset. Omit for all 500 records.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing outputs for this model before generating.",
    )
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def stable_seed(base_seed: int, *parts: Any) -> int:
    payload = "|".join([str(base_seed)] + [str(part) for part in parts])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected JSON object in {path} at line {line_number}"
                )
            rows.append(row)
    return rows


def atomic_write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def atomic_write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def load_source_pools(
    source_path: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = read_jsonl(source_path)
    paired = sorted(
        [row for row in rows if row.get("role") == "paired"],
        key=lambda row: str(row.get("record_id", "")),
    )
    calibration = sorted(
        [row for row in rows if row.get("role") == "calibration"],
        key=lambda row: str(row.get("record_id", "")),
    )

    if len(paired) != 500:
        raise ValueError(
            f"{source_path}: expected 500 paired rows, found {len(paired)}"
        )
    if len(calibration) != 500:
        raise ValueError(
            f"{source_path}: expected 500 calibration rows, "
            f"found {len(calibration)}"
        )

    paired_ids = [str(row.get("record_id")) for row in paired]
    calibration_ids = [str(row.get("record_id")) for row in calibration]

    if len(set(paired_ids)) != len(paired_ids):
        raise ValueError(f"{source_path}: duplicate paired record_id")
    if len(set(calibration_ids)) != len(calibration_ids):
        raise ValueError(f"{source_path}: duplicate calibration record_id")

    return paired, calibration


def load_tokenizer(model_path: str):
    common = {
        "local_files_only": True,
        "trust_remote_code": True,
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            **common,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            **common,
        )

    if tokenizer.eos_token_id is None:
        raise ValueError(f"Tokenizer at {model_path} has no eos_token_id")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    return tokenizer


def load_model(model_path: str, device: str, dtype: torch.dtype):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    return model


def encode_source(
    tokenizer,
    text: str,
    required_tokens: int,
    record_id: str,
) -> List[int]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{record_id}: missing non-empty text")

    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    ids = list(encoded["input_ids"])

    if len(ids) < required_tokens:
        raise ValueError(
            f"{record_id}: only {len(ids)} tokens under the current model; "
            f"{required_tokens} required"
        )
    return ids


def normalize_eos_ids(eos_token_id: Any) -> List[int]:
    if eos_token_id is None:
        return []
    if isinstance(eos_token_id, int):
        return [eos_token_id]
    return [int(value) for value in eos_token_id]


def strip_generated_tail(
    output_ids: Sequence[int],
    prompt_length: int,
    eos_ids: Sequence[int],
    pad_token_id: Optional[int],
) -> List[int]:
    eos_set = set(int(value) for value in eos_ids)
    new_ids: List[int] = []

    for token_id in output_ids[prompt_length:]:
        token_id = int(token_id)
        if token_id in eos_set:
            break
        if pad_token_id is not None and token_id == int(pad_token_id):
            break
        new_ids.append(token_id)

    return new_ids


def decode_ids(tokenizer, ids: Sequence[int]) -> str:
    return tokenizer.decode(
        list(ids),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def retokenized_prefix_diagnostic(
    tokenizer,
    prompt_text: str,
    candidate_text: str,
) -> Tuple[bool, int, int]:
    """Diagnostic only; byte-level BPE need not be prefix-stable."""
    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=True,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    candidate_ids = tokenizer(
        candidate_text,
        add_special_tokens=True,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]

    match = (
        len(candidate_ids) >= len(prompt_ids)
        and list(candidate_ids[: len(prompt_ids)]) == list(prompt_ids)
    )
    return bool(match), len(prompt_ids), len(candidate_ids)


def manifest_config(
    args: argparse.Namespace,
    model_path: str,
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "model_name": args.model_name,
        "model_path": str(Path(model_path)),
        "datasets": [
            value.strip()
            for value in args.datasets.split(",")
            if value.strip()
        ],
        "batch_size": args.batch_size,
        "seed": args.seed,
        "prompt_tokens": args.prompt_tokens,
        "min_total_tokens": args.min_total_tokens,
        "max_total_tokens": args.max_total_tokens,
        "sampling": {
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "num_beams": 1,
        },
        "dtype": args.dtype,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
    }


def validate_existing_rows(
    rows: Sequence[Dict[str, Any]],
    dataset: str,
    model_name: str,
) -> None:
    for expected_index, row in enumerate(rows):
        if row.get("dataset") != dataset:
            raise ValueError(
                f"Existing output has dataset={row.get('dataset')}, "
                f"expected {dataset}"
            )
        if row.get("generator_model") != model_name:
            raise ValueError(
                "Existing output was generated by a different model: "
                f"{row.get('generator_model')}"
            )
        if int(row.get("pair_index", -1)) != expected_index:
            raise ValueError(
                "Existing output is not a contiguous prefix: "
                f"expected pair_index={expected_index}, "
                f"found {row.get('pair_index')}"
            )


def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not (
        0 < args.prompt_tokens
        < args.min_total_tokens
        <= args.max_total_tokens
    ):
        raise ValueError(
            "Require 0 < prompt_tokens < min_total_tokens "
            "<= max_total_tokens"
        )

    if args.max_records is not None:
        if not 1 <= args.max_records <= 500:
            raise ValueError("--max-records must be between 1 and 500")
        if (
            args.max_records != 500
            and args.max_records % args.batch_size != 0
        ):
            raise ValueError(
                "--max-records must be a multiple of --batch-size "
                "for deterministic resume"
            )

    model_path = args.model_path or DEFAULT_MODEL_PATHS.get(args.model_name)
    if model_path is None:
        raise ValueError(
            f"No default path for model {args.model_name}; pass --model-path"
        )
    if not Path(model_path).exists():
        raise FileNotFoundError(model_path)

    datasets = [
        value.strip()
        for value in args.datasets.split(",")
        if value.strip()
    ]
    if not datasets:
        raise ValueError("No datasets selected")

    model_output_dir = args.output_root / args.model_name
    manifest_path = model_output_dir / "generation_manifest.json"
    config = manifest_config(args, model_path)

    if args.overwrite and model_output_dir.exists():
        for dataset in datasets:
            output_path = model_output_dir / f"{dataset}.jsonl"
            if output_path.exists():
                output_path.unlink()
        if manifest_path.exists():
            manifest_path.unlink()

    if manifest_path.exists():
        existing_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if existing_manifest != config:
            raise ValueError(
                "Existing manifest differs; use original settings or --overwrite"
            )
    else:
        atomic_write_json(manifest_path, config)

    print("Loading tokenizer:", model_path)
    tokenizer = load_tokenizer(model_path)

    print("Loading model:", model_path)
    dtype = resolve_dtype(args.dtype)
    model = load_model(model_path, args.device, dtype)

    eos_ids = normalize_eos_ids(
        model.generation_config.eos_token_id
        if model.generation_config.eos_token_id is not None
        else tokenizer.eos_token_id
    )
    pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = pad_token_id

    target_records = args.max_records or 500

    for dataset in datasets:
        source_path = args.source_root / f"{dataset}.jsonl"
        if not source_path.exists():
            raise FileNotFoundError(source_path)

        paired_pool, calibration_pool = load_source_pools(source_path)
        output_path = model_output_dir / f"{dataset}.jsonl"
        existing_rows = read_jsonl(output_path) if output_path.exists() else []

        validate_existing_rows(
            existing_rows,
            dataset=dataset,
            model_name=args.model_name,
        )

        if len(existing_rows) >= target_records:
            print(
                f"[{args.model_name}/{dataset}] already has "
                f"{len(existing_rows)} records; skipping"
            )
            continue

        print(
            f"[{args.model_name}/{dataset}] "
            f"{len(existing_rows)}/{target_records}"
        )

        while len(existing_rows) < target_records:
            start = len(existing_rows)
            stop = min(start + args.batch_size, target_records)
            batch_pairs = paired_pool[start:stop]
            batch_calibration = calibration_pool[start:stop]

            batch_seed = stable_seed(
                args.seed,
                args.model_name,
                dataset,
                start,
                stop,
            )
            set_all_seeds(batch_seed)

            pair_source_ids: List[List[int]] = []
            calibration_source_ids: List[List[int]] = []

            for pair_row, calibration_row in zip(
                batch_pairs,
                batch_calibration,
            ):
                pair_source_ids.append(
                    encode_source(
                        tokenizer,
                        pair_row["text"],
                        args.max_total_tokens,
                        str(pair_row.get("record_id")),
                    )
                )
                calibration_source_ids.append(
                    encode_source(
                        tokenizer,
                        calibration_row["text"],
                        args.max_total_tokens,
                        str(calibration_row.get("record_id")),
                    )
                )

            prompt_ids = [
                ids[: args.prompt_tokens]
                for ids in pair_source_ids
            ]
            input_ids = torch.tensor(
                prompt_ids,
                dtype=torch.long,
                device=args.device,
            )
            attention_mask = torch.ones_like(
                input_ids,
                dtype=torch.long,
                device=args.device,
            )

            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    min_length=args.min_total_tokens,
                    max_length=args.max_total_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    num_beams=1,
                    num_return_sequences=1,
                    use_cache=True,
                    pad_token_id=pad_token_id,
                    eos_token_id=(
                        eos_ids if eos_ids else tokenizer.eos_token_id
                    ),
                )

            outputs_cpu = outputs.detach().cpu().tolist()
            new_rows: List[Dict[str, Any]] = []

            for offset, output_ids in enumerate(outputs_cpu):
                pair_index = start + offset
                pair_row = batch_pairs[offset]
                calibration_row = batch_calibration[offset]
                original_ids = pair_source_ids[offset]
                calibration_ids = calibration_source_ids[offset]

                generated_ids = strip_generated_tail(
                    output_ids,
                    prompt_length=args.prompt_tokens,
                    eos_ids=eos_ids,
                    pad_token_id=pad_token_id,
                )

                min_new_tokens = (
                    args.min_total_tokens - args.prompt_tokens
                )
                max_new_tokens = (
                    args.max_total_tokens - args.prompt_tokens
                )
                if not (
                    min_new_tokens
                    <= len(generated_ids)
                    <= max_new_tokens
                ):
                    raise RuntimeError(
                        f"{dataset} pair {pair_index}: generated "
                        f"{len(generated_ids)} new tokens, expected "
                        f"{min_new_tokens}-{max_new_tokens}"
                    )

                total_tokens = args.prompt_tokens + len(generated_ids)
                prompt_sequence = original_ids[: args.prompt_tokens]
                human_sequence = original_ids[:total_tokens]
                machine_sequence = prompt_sequence + generated_ids
                calibration_sequence = calibration_ids[:total_tokens]

                # Authoritative token-level integrity checks.
                if human_sequence[: args.prompt_tokens] != prompt_sequence:
                    raise RuntimeError(
                        f"{dataset} pair {pair_index}: human token prefix mismatch"
                    )
                if machine_sequence[: args.prompt_tokens] != prompt_sequence:
                    raise RuntimeError(
                        f"{dataset} pair {pair_index}: machine token prefix mismatch"
                    )

                prompt_text = decode_ids(tokenizer, prompt_sequence)
                human_text = decode_ids(tokenizer, human_sequence)
                machine_text = decode_ids(tokenizer, machine_sequence)
                calibration_text = decode_ids(
                    tokenizer,
                    calibration_sequence,
                )

                # Diagnostic only. The hard integrity check is the exact
                # token-prefix comparison above. For byte-level BPE tokenizers
                # such as OPT, decoding a token prefix separately need not
                # produce a literal character prefix of the decoded full
                # sequence, especially at a byte/token boundary.
                human_character_prefix_match = human_text.startswith(prompt_text)
                machine_character_prefix_match = machine_text.startswith(
                    prompt_text
                )

                # Diagnostic only; OPT byte-level BPE may legitimately fail.
                (
                    human_retokenized_prefix_match,
                    prompt_retokenized_n,
                    human_retokenized_n,
                ) = retokenized_prefix_diagnostic(
                    tokenizer,
                    prompt_text,
                    human_text,
                )
                (
                    machine_retokenized_prefix_match,
                    _,
                    machine_retokenized_n,
                ) = retokenized_prefix_diagnostic(
                    tokenizer,
                    prompt_text,
                    machine_text,
                )

                row = {
                    "schema_version": 2,
                    "pair_id": (
                        f"{dataset}:{args.model_name}:{pair_index:04d}"
                    ),
                    "pair_index": pair_index,
                    "dataset": dataset,
                    "generator_model": args.model_name,
                    "generator_model_path": str(Path(model_path)),
                    "source_record_id": pair_row.get("record_id"),
                    "calibration_record_id": calibration_row.get(
                        "record_id"
                    ),
                    "prompt_text": prompt_text,
                    "human_text": human_text,
                    "machine_text": machine_text,
                    "calibration_text": calibration_text,
                    "prompt_token_count": args.prompt_tokens,
                    "generated_token_count": len(generated_ids),
                    "total_token_count": total_tokens,
                    "human_token_count_generator_space": total_tokens,
                    "machine_token_count_generator_space": total_tokens,
                    "calibration_token_count_generator_space": total_tokens,
                    "human_character_prefix_match": (
                        human_character_prefix_match
                    ),
                    "machine_character_prefix_match": (
                        machine_character_prefix_match
                    ),
                    # Diagnostic fields retained for compatibility.
                    "human_prompt_prefix_match": (
                        human_retokenized_prefix_match
                    ),
                    "machine_prompt_prefix_match": (
                        machine_retokenized_prefix_match
                    ),
                    "retokenized_prompt_token_count": prompt_retokenized_n,
                    "retokenized_human_token_count": human_retokenized_n,
                    "retokenized_machine_token_count": machine_retokenized_n,
                    "generation_seed": batch_seed,
                    "generation_batch_start": start,
                    "generation_batch_stop": stop,
                    "sampling": {
                        "do_sample": True,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                        "num_beams": 1,
                        "min_total_tokens": args.min_total_tokens,
                        "max_total_tokens": args.max_total_tokens,
                    },
                }
                new_rows.append(row)

            existing_rows.extend(new_rows)
            atomic_write_jsonl(output_path, existing_rows)

            print(
                f"[{args.model_name}/{dataset}] "
                f"{len(existing_rows)}/{target_records} complete"
            )

    print("Generation complete for model:", args.model_name)


if __name__ == "__main__":
    main()
