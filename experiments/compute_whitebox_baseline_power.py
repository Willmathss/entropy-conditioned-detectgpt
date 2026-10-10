#!/usr/bin/env python3
"""Independent full-text white-box evaluation for five DetectLLM baselines.

Each source model scores its own 500 human, 500 machine and 500 independent
human-calibration texts. Non-LLaMA cells use ``--input-root``; corrected LLaMA
cells always use ``--llama-input-root``. Checkpoint metadata binds scores to the
input SHA-256 and scoring checkpoint so incompatible partial runs cannot mix.
"""

import argparse
import csv
import gc
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_ROOT = (
    BASE / "data/test/text"
)
DEFAULT_LLAMA_INPUT_ROOT = (
    BASE / "data/test/text"
)

DEFAULT_OUTPUT_ROOT = (
    BASE / "results/whitebox/probability_rank"
)

MODEL_PATHS = {
    "llama-2-7b": BASE / "models/llama-2-7b",
    "opt-2.7b": BASE / "models/opt-2.7b",
    "opt-6.7b": BASE / "models/opt-6.7b",
    "falcon-7b": BASE / "models/falcon-7b",
}

DEFAULT_MODELS = (
    "falcon-7b",
    "llama-2-7b",
    "opt-2.7b",
    "opt-6.7b",
)

DEFAULT_DATASETS = (
    "c4",
    "squad",
    "xsum",
    "writing",
    "pubmed",
)

TEXT_FIELDS = (
    "human_text",
    "machine_text",
    "calibration_text",
)

METHODS = (
    "Likelihood",
    "Rank",
    "LogRank",
    "Entropy",
    "LRR",
)

ALPHAS = (0.01, 0.05)
PROTOCOL = "white-box/source-model/independent"
SCORE_FORMAT_VERSION = 3
LLAMA_MODEL_LABEL = "llama-2-7b"
# mean(log(rank)) == 0 means every observed token has rank 1, so LRR is
# +infinity in the extended real line. JSON has no portable infinity; this
# finite ordering sentinel preserves the intended ordering in sklearn/NumPy.
LRR_ZERO_LOGRANK_SENTINEL = 1.0e300


def parse_csv(value):
    return tuple(x.strip() for x in value.split(",") if x.strip())


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
    )
    parser.add_argument(
        "--llama-input-root",
        type=Path,
        default=DEFAULT_LLAMA_INPUT_ROOT,
        help=(
            "修正后的 LLaMA full-text 数据根目录；LLaMA 不会从 "
            "--input-root 读取"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--models",
        type=parse_csv,
        default=DEFAULT_MODELS,
    )
    parser.add_argument(
        "--datasets",
        type=parse_csv,
        default=DEFAULT_DATASETS,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-pairs", type=int, default=500)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="每完成 N 对打印一次进度",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="重新计算已有的逐样本分数",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="某个 cell 失败后继续运行下一项",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只验证模型路径、输入版本、500/500/500 和字段",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        help="仅用于 smoke test；每个 cell 取前 N 对",
    )

    return parser.parse_args()


def get_dtype(name):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def input_path(args, model_name, dataset):
    root = (
        args.llama_input_root
        if model_name == LLAMA_MODEL_LABEL
        else args.input_root
    )
    return root / model_name / f"{dataset}.jsonl"


def input_version(model_name):
    if model_name == LLAMA_MODEL_LABEL:
        return "corrected-llama/full-text"
    return "strict-nonllama/full-text"


def read_rows(path, expected_pairs, limit_samples=None):
    rows = []

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc

            if not isinstance(row, dict):
                raise ValueError(
                    f"{path}:{line_number}: expected a JSON object"
                )

            for field in TEXT_FIELDS:
                value = row.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"{path}:{line_number}: "
                        f"missing or invalid {field}"
                    )

            rows.append(row)

    if expected_pairs and len(rows) != expected_pairs:
        raise ValueError(
            f"{path}: expected {expected_pairs} rows, "
            f"found {len(rows)}"
        )

    if limit_samples is not None:
        rows = rows[:limit_samples]
    if not rows:
        raise ValueError(f"{path}: no usable rows")

    return rows


def validate_args_and_inputs(args):
    if args.expected_pairs < 0:
        raise ValueError("--expected-pairs cannot be negative")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be positive")
    if args.limit_samples is not None and args.limit_samples < 1:
        raise ValueError("--limit-samples must be positive")
    if LLAMA_MODEL_LABEL in args.models and not args.llama_input_root.is_dir():
        raise FileNotFoundError(
            f"corrected LLaMA input root not found: {args.llama_input_root}"
        )
    if any(model != LLAMA_MODEL_LABEL for model in args.models):
        if not args.input_root.is_dir():
            raise FileNotFoundError(
                f"non-LLaMA input root not found: {args.input_root}"
            )

    for model_name in args.models:
        if model_name not in MODEL_PATHS:
            raise ValueError(f"unknown model: {model_name}")
        model_path = MODEL_PATHS[model_name]
        if not model_path.is_dir():
            raise FileNotFoundError(model_path)
        if not (model_path / "config.json").is_file():
            raise FileNotFoundError(model_path / "config.json")
        for dataset in args.datasets:
            path = input_path(args, model_name, dataset)
            rows = read_rows(
                path,
                args.expected_pairs,
                args.limit_samples,
            )
            print(
                f"OK {model_name}/{dataset}: {len(rows)} pairs; "
                f"version={input_version(model_name)}; path={path}",
                flush=True,
            )


@torch.inference_mode()
def score_text(text, tokenizer, model, device):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=False,
    ).to(device)

    sequence_length = encoded["input_ids"].shape[1]
    max_context = getattr(
        model.config,
        "max_position_embeddings",
        None,
    )

    if max_context and sequence_length > max_context:
        raise ValueError(
            f"text has {sequence_length} tokens, "
            f"but model context is {max_context}"
        )

    if sequence_length < 2:
        raise ValueError("text contains fewer than two tokens")

    logits = model(
        **encoded,
        use_cache=False,
    ).logits[0, :-1].float()

    labels = encoded["input_ids"][0, 1:]

    log_probs = torch.log_softmax(logits, dim=-1)

    observed_log_probs = log_probs.gather(
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)

    likelihood = observed_log_probs.mean()

    observed_logits = logits.gather(
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)

    # 与完整 argsort 的1-indexed rank等价（不存在精确logit ties时）。
    ranks = (
        (logits > observed_logits.unsqueeze(-1))
        .sum(dim=-1)
        .float()
        + 1.0
    )

    mean_rank = ranks.mean()
    mean_logrank = torch.log(ranks).mean()

    probabilities = log_probs.exp()
    entropy = -(probabilities * log_probs).sum(dim=-1).mean()

    mean_logrank_value = float(mean_logrank.item())
    if mean_logrank_value < 0:
        raise ValueError("mean log-rank is negative")
    zero_logrank = mean_logrank_value == 0.0
    lrr = (
        LRR_ZERO_LOGRANK_SENTINEL
        if zero_logrank
        else float((-likelihood / mean_logrank).item())
    )

    # 所有分数统一为：越大越像machine。
    return {
        "Likelihood": float(likelihood.item()),
        "Rank": float((-mean_rank).item()),
        "LogRank": float((-mean_logrank).item()),
        "Entropy": float(entropy.item()),
        "LRR": lrr,
        "_n_tokens": int(labels.numel()),
        "_lrr_zero_logrank": zero_logrank,
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_file_fingerprint(model_path):
    records = []
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    ):
        path = model_path / name
        if path.is_file():
            records.append(
                {
                    "name": name,
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    for pattern in ("*.safetensors", "pytorch_model*.bin"):
        for path in sorted(model_path.glob(pattern)):
            records.append(
                {
                    "name": path.name,
                    "size": path.stat().st_size,
                }
            )
    return records


def expected_metadata(
    source_path,
    source_model,
    dataset,
    n_pairs,
    model_path,
    args,
):
    return {
        "score_format_version": SCORE_FORMAT_VERSION,
        "protocol": PROTOCOL,
        "scoring_region": "full-text",
        "source_model": source_model,
        "dataset": dataset,
        "input_version": input_version(source_model),
        "input_path": str(source_path.resolve()),
        "input_sha256": sha256_file(source_path),
        "n_pairs": n_pairs,
        "scoring_model_path": str(model_path.resolve()),
        "scoring_model_files": model_file_fingerprint(model_path),
        "methods": list(METHODS),
        "text_fields": list(TEXT_FIELDS),
        "dtype": args.dtype,
        "limit_samples": args.limit_samples,
        "lrr_zero_logrank_policy": {
            "meaning": "extended-real +infinity (all observed ranks are 1)",
            "json_sentinel": LRR_ZERO_LOGRANK_SENTINEL,
        },
        "trust_remote_code": False,
    }


def atomic_write_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path, value):
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
    )


def prepare_cell_checkpoint(output_dir, metadata, overwrite):
    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / "scores_with_calibration.jsonl"
    metadata_path = output_dir / "metadata.json"
    result_path = output_dir / "power_result.json"

    if overwrite:
        for path in (score_path, metadata_path, result_path):
            if path.exists():
                path.unlink()

    existing_payload = score_path.exists() or result_path.exists()
    if existing_payload and not metadata_path.is_file():
        raise ValueError(
            f"{output_dir}: checkpoint has no metadata; rerun this cell with "
            "--overwrite"
        )
    if metadata_path.is_file():
        old = json.loads(metadata_path.read_text(encoding="utf-8"))
        if old != metadata:
            changed = sorted(
                key
                for key in set(old) | set(metadata)
                if old.get(key) != metadata.get(key)
            )
            raise ValueError(
                f"stale/incompatible checkpoint in {output_dir}; changed "
                f"metadata fields={changed}. Rerun this cell with --overwrite"
            )
    else:
        atomic_write_json(metadata_path, metadata)
    return score_path


def load_partial_scores(path):
    if not path.exists():
        return []

    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue

            record = json.loads(line)

            if record.get("index") != len(records):
                raise ValueError(
                    f"{path}:{line_number}: "
                    "non-contiguous sample index"
                )

            for field in TEXT_FIELDS:
                values = record.get(field)
                if not isinstance(values, dict):
                    raise ValueError(
                        f"{path}:{line_number}: missing {field} scores"
                    )
                for method in METHODS:
                    if not np.isfinite(values.get(method, np.nan)):
                        raise ValueError(
                            f"{path}:{line_number}: invalid {field}/{method}"
                        )
                if int(values.get("_n_tokens", 0)) < 1:
                    raise ValueError(
                        f"{path}:{line_number}: invalid {field}/_n_tokens"
                    )
                if not isinstance(values.get("_lrr_zero_logrank"), bool):
                    raise ValueError(
                        f"{path}:{line_number}: missing "
                        f"{field}/_lrr_zero_logrank diagnostic"
                    )

            records.append(record)

    return records


def score_cell(
    rows,
    score_path,
    tokenizer,
    model,
    device,
    progress_every,
):
    score_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_partial_scores(score_path)

    if len(records) > len(rows):
        raise ValueError(
            f"{score_path} contains too many records"
        )

    if len(records) == len(rows):
        print(f"SKIP completed: {score_path}")
        return records

    with score_path.open("a", encoding="utf-8") as f:
        for index in range(len(records), len(rows)):
            row = rows[index]

            record = {
                "index": index,
            }

            for field in TEXT_FIELDS:
                record[field] = score_text(
                    row[field],
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                )

            f.write(
                json.dumps(record, ensure_ascii=False)
                + "\n"
            )
            f.flush()

            records.append(record)

            if (
                (index + 1) % progress_every == 0
                or index + 1 == len(rows)
            ):
                print(
                    f"{score_path.parent}: "
                    f"{index + 1}/{len(rows)}"
                )

    return records


def conformal_p_values(calibration_scores, test_scores):
    """
    Larger score means more machine-like.

    p(x) = (1 + #{calibration score >= test score})
           / (n_calibration + 1)
    """
    calibration = np.sort(
        np.asarray(calibration_scores, dtype=np.float64)
    )
    test = np.asarray(test_scores, dtype=np.float64)

    first_greater_equal = np.searchsorted(
        calibration,
        test,
        side="left",
    )

    number_greater_equal = (
        len(calibration) - first_greater_equal
    )

    return (
        1.0 + number_greater_equal
    ) / float(len(calibration) + 1)


def evaluate_method(records, method):
    calibration = np.asarray(
        [
            record["calibration_text"][method]
            for record in records
        ],
        dtype=np.float64,
    )

    human = np.asarray(
        [
            record["human_text"][method]
            for record in records
        ],
        dtype=np.float64,
    )

    machine = np.asarray(
        [
            record["machine_text"][method]
            for record in records
        ],
        dtype=np.float64,
    )

    for name, values in (
        ("calibration", calibration),
        ("human", human),
        ("machine", machine),
    ):
        if not np.isfinite(values).all():
            raise ValueError(
                f"{method}: non-finite {name} scores"
            )

    labels = np.concatenate(
        [
            np.zeros(len(human), dtype=int),
            np.ones(len(machine), dtype=int),
        ]
    )

    test_scores = np.concatenate([human, machine])

    result = {
        "auroc": float(
            roc_auc_score(labels, test_scores)
        ),
        "n_calibration": len(calibration),
        "n_human_test": len(human),
        "n_machine_test": len(machine),
    }

    human_p = conformal_p_values(
        calibration,
        human,
    )
    machine_p = conformal_p_values(
        calibration,
        machine,
    )

    for alpha in ALPHAS:
        suffix = f"{alpha:.2f}"

        result[f"type_i_error_at_{suffix}"] = float(
            np.mean(human_p <= alpha)
        )
        result[f"power_at_{suffix}"] = float(
            np.mean(machine_p <= alpha)
        )

    return result


def write_cell_result(
    output_dir,
    source_model,
    dataset,
    records,
    metadata,
):
    methods = {
        method: evaluate_method(records, method)
        for method in METHODS
    }

    result = {
        "protocol": PROTOCOL,
        "score_format_version": SCORE_FORMAT_VERSION,
        "scoring_region": "full-text",
        "source_model": source_model,
        "dataset": dataset,
        "n_pairs": len(records),
        "score_direction": "larger means more machine-like",
        "calibration_protocol": (
            "independent human calibration; "
            "finite-sample conformal p-values"
        ),
        "metadata": metadata,
        "lrr_zero_logrank_counts": {
            field: sum(
                bool(record[field]["_lrr_zero_logrank"])
                for record in records
            )
            for field in TEXT_FIELDS
        },
        "methods": methods,
    }

    path = output_dir / "power_result.json"

    atomic_write_json(path, result)

    return result


def collect_results(output_root):
    rows = []

    for source_model in DEFAULT_MODELS:
        for dataset in DEFAULT_DATASETS:
            path = (
                output_root
                / source_model
                / dataset
                / "power_result.json"
            )

            if not path.exists():
                continue

            result = json.loads(
                path.read_text(encoding="utf-8")
            )

            if result.get("protocol") != PROTOCOL:
                continue
            if result.get("score_format_version") != SCORE_FORMAT_VERSION:
                continue

            for method, metrics in result["methods"].items():
                rows.append(
                    {
                        "source_model": source_model,
                        "dataset": dataset,
                        "method": method,
                        **metrics,
                    }
                )

    return rows


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_summaries(output_root):
    rows = collect_results(output_root)

    detailed_fields = (
        "source_model",
        "dataset",
        "method",
        "auroc",
        "type_i_error_at_0.01",
        "power_at_0.01",
        "type_i_error_at_0.05",
        "power_at_0.05",
        "n_calibration",
        "n_human_test",
        "n_machine_test",
    )

    write_csv(
        output_root / "summary_power.csv",
        rows,
        detailed_fields,
    )

    grouped = defaultdict(list)

    for row in rows:
        grouped[row["method"]].append(row)

    macro_rows = []

    for method in METHODS:
        method_rows = grouped.get(method, [])

        if not method_rows:
            continue

        macro_rows.append(
            {
                "method": method,
                "cells": len(method_rows),
                "auroc": np.mean(
                    [row["auroc"] for row in method_rows]
                ),
                "type_i_error_at_0.01": np.mean(
                    [
                        row["type_i_error_at_0.01"]
                        for row in method_rows
                    ]
                ),
                "power_at_0.01": np.mean(
                    [
                        row["power_at_0.01"]
                        for row in method_rows
                    ]
                ),
                "type_i_error_at_0.05": np.mean(
                    [
                        row["type_i_error_at_0.05"]
                        for row in method_rows
                    ]
                ),
                "power_at_0.05": np.mean(
                    [
                        row["power_at_0.05"]
                        for row in method_rows
                    ]
                ),
            }
        )

    macro_fields = (
        "method",
        "cells",
        "auroc",
        "type_i_error_at_0.01",
        "power_at_0.01",
        "type_i_error_at_0.05",
        "power_at_0.05",
    )

    write_csv(
        output_root / "table1_macro.csv",
        macro_rows,
        macro_fields,
    )

    print("\n===== Table I macro results =====")

    for row in macro_rows:
        print(
            f"{row['method']:10s} "
            f"cells={row['cells']:2d}  "
            f"AUC={row['auroc']:.4f}  "
            f"FPR@1%={row['type_i_error_at_0.01']:.4f}  "
            f"Power@1%={row['power_at_0.01']:.4f}  "
            f"FPR@5%={row['type_i_error_at_0.05']:.4f}  "
            f"Power@5%={row['power_at_0.05']:.4f}"
        )


def main():
    args = parse_args()
    validate_args_and_inputs(args)
    if args.check_only:
        print("All requested full-text independent inputs passed validation.")
        return

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    failures = []

    for model_name in args.models:
        model_path = MODEL_PATHS[model_name]
        print(f"\n===== Loading {model_name} =====", flush=True)
        print(f"Path: {model_path}", flush=True)

        tokenizer = None
        model = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                use_fast=False,
                local_files_only=True,
                trust_remote_code=False,
            )

            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=get_dtype(args.dtype),
                low_cpu_mem_usage=True,
                local_files_only=True,
                trust_remote_code=False,
            ).to(args.device)
            model.eval()

            for dataset in args.datasets:
                print(
                    f"\n===== {model_name} / {dataset} =====",
                    flush=True,
                )
                try:
                    source_path = input_path(args, model_name, dataset)
                    rows = read_rows(
                        source_path,
                        args.expected_pairs,
                        args.limit_samples,
                    )
                    output_dir = args.output_root / model_name / dataset
                    metadata = expected_metadata(
                        source_path=source_path,
                        source_model=model_name,
                        dataset=dataset,
                        n_pairs=len(rows),
                        model_path=model_path,
                        args=args,
                    )
                    score_path = prepare_cell_checkpoint(
                        output_dir,
                        metadata,
                        args.overwrite,
                    )
                    records = score_cell(
                        rows=rows,
                        score_path=score_path,
                        tokenizer=tokenizer,
                        model=model,
                        device=args.device,
                        progress_every=args.progress_every,
                    )
                    write_cell_result(
                        output_dir=output_dir,
                        source_model=model_name,
                        dataset=dataset,
                        records=records,
                        metadata=metadata,
                    )
                    # 每完成一个 cell 就原子更新汇总，保留断点进度。
                    write_summaries(args.output_root)
                except Exception as exc:
                    label = f"{model_name}/{dataset}"
                    message = f"{label}: {type(exc).__name__}: {exc}"
                    failures.append(message)
                    print(f"FAILED {message}", flush=True)
                    if not args.continue_on_error:
                        raise
        finally:
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_summaries(args.output_root)
    if failures:
        print("Failures:")
        for failure in failures:
            print(f"  {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
