#!/usr/bin/env python3
"""Generate one frozen fresh white-/black-box confirmation cell.

Both test humans and calibration humans undergo the same randomized length
matching rule: each is paired with an independently generated passage from the
same source model and trimmed to the shorter word count.  The calibration-side
machine passage is discarded and is never evaluated as a machine example.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from confirmatory_common import (
    PROTOCOL_NAME,
    SCHEMA_VERSION,
    atomic_json,
    atomic_jsonl,
    file_sha256,
    normalize_text,
    read_jsonl,
    text_sha256,
)


MODELS = ("falcon-7b", "llama-2-7b", "opt-2.7b", "opt-6.7b")
DATASETS = ("c4", "pubmed", "squad", "writing", "xsum")


def effective_trust_remote_code(model_label: str, requested: bool) -> bool:
    """Use the Transformers-native Falcon implementation.

    The legacy Python shipped inside the local Falcon repository calls
    ``get_head_mask``, which is absent from the installed Transformers base
    class.  Falcon is now natively supported, so remote code is both
    unnecessary and incompatible in this environment.
    """

    if model_label == "falcon-7b":
        return False
    return requested


def import_original(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("water_original_generator", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sampling_configuration(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    if args.exact_cond_prob:
        return "exact_cond_prob", {"temperature": 1.0, "top_p": 1.0, "top_k": 0}
    if args.do_top_p:
        return "top_p", {"temperature": 1.0, "top_p": args.top_p, "top_k": 0}
    if args.do_top_k:
        return "top_k", {"temperature": 1.0, "top_p": 1.0, "top_k": args.top_k}
    if args.do_temperature:
        return "temperature", {
            "temperature": args.temperature,
            "top_p": 1.0,
            "top_k": 0,
        }
    raise AssertionError("one sampling mode must be selected")


def template_configuration(path: Path) -> dict[str, Any]:
    if path.suffix == ".jsonl":
        rows = read_jsonl(path)
        if not rows or not isinstance(rows[0].get("generation"), dict):
            raise ValueError(f"{path}: first row has no generation configuration")
        # Some legacy JSONL rows omit inactive sampling fields even when the
        # sibling metadata records them.  Merge the two representations and
        # let a non-null row value take precedence.
        config: dict[str, Any] = {}
        metadata_path = path.with_suffix(".metadata.json")
        if metadata_path.is_file():
            with metadata_path.open("r", encoding="utf-8") as stream:
                metadata = json.load(stream)
            metadata_config = (
                metadata.get("generation") if isinstance(metadata, dict) else None
            )
            if isinstance(metadata_config, dict):
                config.update(metadata_config)
        config.update(
            {key: value for key, value in rows[0]["generation"].items() if value is not None}
        )
        return config
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    config = value.get("generation") if isinstance(value, dict) else None
    if not isinstance(config, dict):
        raise ValueError(f"{path}: no generation configuration")
    return dict(config)


def resolve_generation_configuration(
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]]:
    flags = (
        args.exact_cond_prob,
        args.do_top_p,
        args.do_top_k,
        args.do_temperature,
    )
    if args.template_generated_file is not None:
        if any(flags) or args.prompt_tokens is not None or args.max_length is not None:
            raise ValueError(
                "with --template-generated-file, do not pass sampling flags, "
                "--prompt-tokens, or --max-length"
            )
        config = template_configuration(args.template_generated_file.resolve())
        mode_value = config.get("sampling_mode")
        if mode_value is None:
            raise ValueError(
                f"{args.template_generated_file}: generation configuration "
                "has no sampling_mode"
            )
        mode = str(mode_value).strip().lower().replace("-", "_")
        args.prompt_tokens = int(config["prompt_tokens"])
        args.max_length = int(config["max_length"])
        args.min_length = int(config.get("min_length", 150))
        args.min_generated_words = int(config.get("min_generated_words", 55))
        if mode == "exact_cond_prob":
            kwargs = {"temperature": 1.0, "top_p": 1.0, "top_k": 0}
        elif mode == "top_p":
            kwargs = {
                "temperature": 1.0,
                "top_p": float(
                    args.top_p if config.get("top_p") is None else config["top_p"]
                ),
                "top_k": 0,
            }
        elif mode == "top_k":
            kwargs = {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": int(
                    args.top_k if config.get("top_k") is None else config["top_k"]
                ),
            }
        elif mode == "temperature":
            kwargs = {
                "temperature": float(
                    args.temperature
                    if config.get("temperature") is None
                    else config["temperature"]
                ),
                "top_p": 1.0,
                "top_k": 0,
            }
        else:
            raise ValueError(f"{args.template_generated_file}: unknown mode {mode}")
        return mode, kwargs
    if sum(bool(value) for value in flags) != 1:
        raise ValueError("select exactly one sampling mode or provide a template")
    if args.prompt_tokens is None or args.max_length is None:
        raise ValueError("--prompt-tokens and --max-length are required without a template")
    if args.min_length is None:
        args.min_length = 150
    if args.min_generated_words is None:
        args.min_generated_words = 55
    return sampling_configuration(args)


def validate_sources(
    test_rows: Sequence[Mapping[str, Any]],
    calibration_rows: Sequence[Mapping[str, Any]],
    expected: int,
) -> None:
    if len(test_rows) != expected or len(calibration_rows) != expected:
        raise ValueError(
            f"expected {expected} test and calibration rows; got "
            f"{len(test_rows)} and {len(calibration_rows)}"
        )
    test_hashes = {row.get("source_text_sha256") for row in test_rows}
    calibration_hashes = {
        row.get("source_text_sha256") for row in calibration_rows
    }
    if len(test_hashes) != expected or len(calibration_hashes) != expected:
        raise ValueError("source split contains duplicates or missing hashes")
    if test_hashes.intersection(calibration_hashes):
        raise ValueError("test and calibration source texts overlap")


def load_resume(
    path: Path,
    test_rows: Sequence[Mapping[str, Any]],
    calibration_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = read_jsonl(path)
    if len(rows) > len(test_rows):
        raise ValueError(f"{path}: too many rows")
    for index, row in enumerate(rows):
        if row.get("test_record_id") != test_rows[index].get("record_id"):
            raise ValueError(f"{path}: test source mismatch at row {index}")
        if row.get("calibration_record_id") != calibration_rows[index].get("record_id"):
            raise ValueError(f"{path}: calibration source mismatch at row {index}")
    return rows


def generate_passages(
    original,
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    max_length: int,
    min_length: int,
    min_words: int,
    sampling_kwargs: Mapping[str, Any],
    seed: int,
    max_attempts: int,
) -> list[str]:
    passages = original.generate_batch(
        model,
        tokenizer,
        prompts,
        max_length=max_length,
        min_length=min_length,
        sampling_kwargs=sampling_kwargs,
        batch_seed=seed,
    )
    for offset, passage in enumerate(passages):
        if len(normalize_text(passage).split()) >= min_words:
            continue
        passages[offset] = original.generate_one(
            model,
            tokenizer,
            prompts[offset],
            max_length=max_length,
            min_length=min_length,
            sampling_kwargs=sampling_kwargs,
            seed=seed + offset * 10_000,
            max_attempts=max_attempts,
            min_words=min_words,
        )
    return passages


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--model-label", choices=MODELS, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument(
        "--template-generated-file",
        type=Path,
        help="Old frozen JSONL/metadata whose generation distribution is copied.",
    )
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--prompt-tokens", type=int)
    parser.add_argument("--min-length", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--min-generated-words", type=int)
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--exact-cond-prob", action="store_true")
    sampling.add_argument("--do-top-p", action="store_true")
    sampling.add_argument("--do-top-k", action="store_true")
    sampling.add_argument("--do-temperature", action="store_true")
    parser.add_argument("--top-p", type=float, default=0.96)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--slow-tokenizer", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate inputs and print the effective generation lock without loading a model",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.n_samples <= 0 or args.batch_size <= 0:
        parser.error("sample and batch sizes must be positive")
    args.project_root = args.project_root.resolve()
    args.source_root = args.source_root.resolve()
    original_path = args.project_root / "experiments/generate_independent_whitebox_test.py"
    original = import_original(original_path)

    test_path = args.source_root / "test" / f"{args.dataset}.jsonl"
    calibration_path = args.source_root / "calibration" / f"{args.dataset}.jsonl"
    test_rows = read_jsonl(test_path)
    calibration_rows = read_jsonl(calibration_path)
    validate_sources(test_rows, calibration_rows, args.n_samples)

    model_path = args.model_path or Path(original.DEFAULT_MODEL_PATHS[args.model_label])
    model_path = model_path.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    use_remote_code = effective_trust_remote_code(
        args.model_label, args.trust_remote_code
    )
    if args.model_label == "falcon-7b" and args.trust_remote_code:
        print(
            "FALCON COMPATIBILITY: forcing trust_remote_code=False and using "
            "the Transformers-native Falcon implementation.",
            flush=True,
        )
    try:
        mode, sampling_kwargs = resolve_generation_configuration(args)
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    assert args.prompt_tokens is not None and args.max_length is not None
    assert args.min_length is not None and args.min_generated_words is not None
    if args.max_length <= args.prompt_tokens:
        parser.error("--max-length must exceed --prompt-tokens")
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "preflight_ok",
                    "dataset": args.dataset,
                    "model_label": args.model_label,
                    "model_path": str(model_path),
                    "test_source": str(test_path),
                    "calibration_source": str(calibration_path),
                    "n_samples": args.n_samples,
                    "generation": {
                        "sampling_mode": mode,
                        **sampling_kwargs,
                        "prompt_tokens": args.prompt_tokens,
                        "min_length": args.min_length,
                        "max_length": args.max_length,
                        "min_generated_words": args.min_generated_words,
                        "seed": args.seed,
                    },
                    "device": args.device,
                    "trust_remote_code": use_remote_code,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    output_path = args.output_root / args.model_label / f"{args.dataset}.jsonl"
    metadata_path = output_path.with_suffix(".metadata.json")
    base_metadata = {
        "schema_version": SCHEMA_VERSION,
        "protocol_name": PROTOCOL_NAME,
        "dataset": args.dataset,
        "model_label": args.model_label,
        "model_path": str(model_path),
        "generator_script_sha256": file_sha256(Path(__file__).resolve()),
        "imported_generator_sha256": file_sha256(original_path),
        "n_samples": args.n_samples,
        "test_source_file": str(test_path),
        "test_source_sha256": file_sha256(test_path),
        "calibration_source_file": str(calibration_path),
        "calibration_source_sha256": file_sha256(calibration_path),
        "generation_template_file": (
            None
            if args.template_generated_file is None
            else str(args.template_generated_file.resolve())
        ),
        "generation_template_sha256": (
            None
            if args.template_generated_file is None
            else file_sha256(args.template_generated_file.resolve())
        ),
        "output_file": str(output_path),
        "human_test_and_calibration_preprocessing_identical": True,
        "calibration_machine_reference_used_only_for_length_matching": True,
        "generation": {
            "sampling_mode": mode,
            **sampling_kwargs,
            "prompt_tokens": args.prompt_tokens,
            "min_length": args.min_length,
            "max_length": args.max_length,
            "min_generated_words": args.min_generated_words,
            "seed": args.seed,
        },
        "generation_runtime": {
            "batch_size": args.batch_size,
            "max_attempts": args.max_attempts,
            "dtype": args.dtype,
            "device": args.device,
            "device_map": args.device_map,
            "trust_remote_code": use_remote_code,
            "slow_tokenizer": args.slow_tokenizer,
        },
    }
    recorded: dict[str, Any] | None = None
    if metadata_path.exists():
        recorded = json.load(metadata_path.open(encoding="utf-8"))
        changed_keys = [
            key for key, value in base_metadata.items() if recorded.get(key) != value
        ]
        pristine_failed_start = (
            recorded.get("status") == "generation_in_progress"
            and int(recorded.get("completed", -1)) == 0
            and not output_path.exists()
        )
        if changed_keys and pristine_failed_start:
            # No text was generated, so replacing the pre-model-load lock is
            # safe and avoids requiring users to delete failed-start metadata.
            atomic_json(
                metadata_path,
                {**base_metadata, "status": "generation_in_progress", "completed": 0},
            )
            recorded = None
        elif changed_keys:
            raise ValueError(
                f"{metadata_path}: resume configuration changed at "
                f"{changed_keys[0]}"
            )
    elif output_path.exists():
        raise FileExistsError(
            f"{output_path} exists without its metadata lock; refuse unsafe resume"
        )
    else:
        atomic_json(
            metadata_path,
            {**base_metadata, "status": "generation_in_progress", "completed": 0},
        )
    if recorded is not None:
        status = recorded.get("status")
        if status not in {"generation_in_progress", "generated_and_frozen"}:
            raise ValueError(f"{metadata_path}: invalid status {status!r}")
        if not output_path.exists():
            if int(recorded.get("completed", 0)) != 0:
                raise FileNotFoundError(
                    f"{output_path}: missing partial output recorded by metadata"
                )
        else:
            hash_key = (
                "output_sha256"
                if status == "generated_and_frozen"
                else "partial_output_sha256"
            )
            if recorded.get(hash_key) != file_sha256(output_path):
                raise ValueError(
                    f"{output_path}: content differs from recorded {hash_key}"
                )
    rows = load_resume(output_path, test_rows, calibration_rows)
    if recorded is not None and int(recorded.get("completed", -1)) != len(rows):
        raise ValueError(
            f"{metadata_path}: completed count does not match output rows"
        )
    if recorded is not None and recorded.get("status") == "generated_and_frozen":
        if len(rows) != args.n_samples:
            raise ValueError(f"{metadata_path}: frozen output is incomplete")
    if len(rows) == args.n_samples:
        atomic_json(
            metadata_path,
            {
                **base_metadata,
                "status": "generated_and_frozen",
                "completed": len(rows),
                "output_sha256": file_sha256(output_path),
            },
        )
        print(f"COMPLETE {output_path}")
        return 0

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=use_remote_code,
        use_fast=not args.slow_tokenizer,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": use_remote_code,
        "low_cpu_mem_usage": True,
        "torch_dtype": original.model_dtype(args.dtype),
        "local_files_only": True,
    }
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(str(model_path), **model_kwargs)
    if not args.device_map:
        model.to(torch.device(args.device))
    model.eval()

    while len(rows) < args.n_samples:
        start = len(rows)
        stop = min(start + args.batch_size, args.n_samples)
        current_test = test_rows[start:stop]
        current_calibration = calibration_rows[start:stop]
        prepared_test = [
            original.prepare_human_and_prompt(
                str(row["source_text"]), args.dataset, tokenizer, args.prompt_tokens
            )
            for row in current_test
        ]
        prepared_calibration = [
            original.prepare_human_and_prompt(
                str(row["source_text"]), args.dataset, tokenizer, args.prompt_tokens
            )
            for row in current_calibration
        ]
        test_humans = [item[0] for item in prepared_test]
        test_prompts = [item[1] for item in prepared_test]
        calibration_humans = [item[0] for item in prepared_calibration]
        calibration_prompts = [item[1] for item in prepared_calibration]

        test_machines = generate_passages(
            original,
            model,
            tokenizer,
            test_prompts,
            max_length=args.max_length,
            min_length=args.min_length,
            min_words=args.min_generated_words,
            sampling_kwargs=sampling_kwargs,
            seed=args.seed + start * 20_000,
            max_attempts=args.max_attempts,
        )
        calibration_length_references = generate_passages(
            original,
            model,
            tokenizer,
            calibration_prompts,
            max_length=args.max_length,
            min_length=args.min_length,
            min_words=args.min_generated_words,
            sampling_kwargs=sampling_kwargs,
            seed=args.seed + 1_000_000_000 + start * 20_000,
            max_attempts=args.max_attempts,
        )

        for offset in range(stop - start):
            human, machine = original.trim_pair(
                test_humans[offset], test_machines[offset]
            )
            calibration, calibration_reference = original.trim_pair(
                calibration_humans[offset], calibration_length_references[offset]
            )
            if len(human.split()) != len(machine.split()):
                raise AssertionError("paired test lengths differ after trimming")
            rows.append(
                {
                    "record_id": f"{args.model_label}-{args.dataset}-{start + offset:04d}",
                    "protocol_name": PROTOCOL_NAME,
                    "dataset": args.dataset,
                    "generator_model": args.model_label,
                    "test_record_id": current_test[offset]["record_id"],
                    "calibration_record_id": current_calibration[offset]["record_id"],
                    "test_source_sha256": current_test[offset]["source_text_sha256"],
                    "calibration_source_sha256": current_calibration[offset]["source_text_sha256"],
                    "human_text": human,
                    "machine_text": machine,
                    "calibration_text": calibration,
                    "calibration_length_reference_sha256": text_sha256(
                        calibration_reference
                    ),
                    "human_words": len(human.split()),
                    "machine_words": len(machine.split()),
                    "calibration_words": len(calibration.split()),
                }
            )
        atomic_jsonl(output_path, rows)
        atomic_json(
            metadata_path,
            {
                **base_metadata,
                "status": "generation_in_progress",
                "completed": len(rows),
                "partial_output_sha256": file_sha256(output_path),
            },
        )
        if len(rows) == args.n_samples or len(rows) % args.log_every < args.batch_size:
            print(
                f"{args.model_label}/{args.dataset}: {len(rows)}/{args.n_samples}",
                flush=True,
            )

    atomic_json(
        metadata_path,
        {
            **base_metadata,
            "status": "generated_and_frozen",
            "completed": len(rows),
            "output_sha256": file_sha256(output_path),
        },
    )
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"FROZEN {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
