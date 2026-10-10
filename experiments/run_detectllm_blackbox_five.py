#!/usr/bin/env python3
"""Independent black-box evaluation for five DetectLLM score baselines.

The source/generator model is never loaded. It is only a directory label for
already generated machine text. Every text is scored with one fixed local
GPT-Neo-2.7B surrogate using five perturbation-free statistics:

* likelihood: mean observed-token log probability;
* entropy: mean next-token predictive entropy;
* rank: negative mean observed-token rank;
* logrank: negative mean log observed-token rank;
* lrr: negative likelihood divided by mean log rank.

Expected input:
    INPUT_ROOT/<non-LLaMA-source-model>/<dataset>.jsonl
    LLAMA_INPUT_ROOT/llama-2-7b/<dataset>.jsonl

Every row must contain non-empty human_text, machine_text and calibration_text.
AUROC uses independent human_text versus machine_text. Operating points use
finite-sample upper-tail conformal p-values from calibration_text.
The complete text in each field is scored; over-context inputs fail instead of
being silently truncated.

The score JSONL is append-only and resumable. Per-cell metadata binds each
checkpoint to the SHA-256 of its input JSONL and all score-affecting settings,
so an updated SQuAD file cannot silently reuse stale scores.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    auc as sklearn_auc,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BASE = Path(".")
DEFAULT_INPUT_ROOT = (
    DEFAULT_BASE / "data/test/text"
)
DEFAULT_LLAMA_INPUT_ROOT = (
    DEFAULT_BASE / "data/test/text"
)
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_BASE / "results/blackbox/probability_rank"
)
DEFAULT_SCORING_MODEL = Path("models/gpt-neo-2.7b")
DEFAULT_SOURCE_MODELS = (
    "falcon-7b",
    "llama-2-7b",
    "opt-2.7b",
    "opt-6.7b",
)
DEFAULT_DATASETS = ("c4", "pubmed", "squad", "writing", "xsum")
METHODS = ("likelihood", "entropy", "rank", "logrank", "lrr")
TEXT_FIELDS = ("human_text", "machine_text", "calibration_text")
ALPHAS = ((0.01, "001", "1pct"), (0.05, "005", "5pct"))
SCORE_FORMAT_VERSION = 3
LLAMA_MODEL_LABEL = "llama-2-7b"
# If every observed token is rank 1, mean(log(rank)) is exactly zero and LRR
# is +infinity in the extended real line.  JSON has no portable infinity, so
# use a finite ordering sentinel that is far above every realizable nonzero-
# denominator LRR while remaining safe in NumPy/sklearn arithmetic.
LRR_ZERO_LOGRANK_SENTINEL = 1.0e300


@dataclass(frozen=True)
class TextScores:
    likelihood: float
    entropy: float
    rank: float
    logrank: float
    lrr: float
    n_tokens: int
    lrr_zero_logrank: bool


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument(
        "--llama-input-root",
        type=Path,
        default=DEFAULT_LLAMA_INPUT_ROOT,
        help=(
            "Root containing the corrected full-text LLaMA pairs. LLaMA is "
            "never read from --input-root."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--scoring-model-path", type=Path, default=DEFAULT_SCORING_MODEL
    )
    parser.add_argument(
        "--source-models",
        type=parse_csv,
        default=DEFAULT_SOURCE_MODELS,
        help="Comma-separated source/generator directory labels.",
    )
    parser.add_argument(
        "--datasets",
        type=parse_csv,
        default=DEFAULT_DATASETS,
        help="Comma-separated dataset names.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Pairs per field-specific forward. One pair requires three "
            "forwards: test human, machine, and calibration human."
        ),
    )
    parser.add_argument(
        "--token-chunk-size",
        type=int,
        default=16,
        help="Token positions processed at once after the model forward.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help=(
            "Hard full-text token limit. Inputs longer than this fail instead "
            "of being silently truncated."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print one progress line every N completed pairs.",
    )
    parser.add_argument("--expected-pairs", type=int, default=500)
    parser.add_argument(
        "--allow-online",
        action="store_true",
        help="Allow Transformers to contact the Hub for missing local files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard checkpoints for requested cells and recompute them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with the next cell after a cell-level failure.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate model/data paths and JSONL fields without loading weights.",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        help="Use only the first N pairs for a smoke test.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.token_chunk_size < 1:
        raise ValueError("--token-chunk-size must be positive")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be positive")
    if args.max_length < 2:
        raise ValueError("--max-length must be at least 2")
    if args.expected_pairs < 0:
        raise ValueError("--expected-pairs cannot be negative")
    if args.limit_samples is not None and args.limit_samples < 1:
        raise ValueError("--limit-samples must be positive")
    if not args.scoring_model_path.is_dir():
        raise FileNotFoundError(
            f"scoring model directory not found: {args.scoring_model_path}"
        )
    if not (args.scoring_model_path / "config.json").is_file():
        raise FileNotFoundError(
            "config.json not found in scoring model: "
            f"{args.scoring_model_path}"
        )
    if LLAMA_MODEL_LABEL in args.source_models:
        if not args.llama_input_root.is_dir():
            raise FileNotFoundError(
                "corrected LLaMA input root not found: "
                f"{args.llama_input_root}"
            )
    non_llama = [
        model for model in args.source_models if model != LLAMA_MODEL_LABEL
    ]
    if non_llama and not args.input_root.is_dir():
        raise FileNotFoundError(
            f"non-LLaMA input root not found: {args.input_root}"
        )


def input_path(args: argparse.Namespace, source_model: str, dataset: str) -> Path:
    root = (
        args.llama_input_root
        if source_model == LLAMA_MODEL_LABEL
        else args.input_root
    )
    return root / source_model / f"{dataset}.jsonl"


def input_version(source_model: str) -> str:
    if source_model == LLAMA_MODEL_LABEL:
        return "corrected-llama/full-text"
    return "strict-nonllama/full-text"


def read_rows(
    path: Path,
    expected_pairs: int,
    limit_samples: int | None,
) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
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
                        f"{path}:{line_number}: {field!r} must be non-empty text"
                    )
            rows.append({field: row[field] for field in TEXT_FIELDS})

    if expected_pairs and len(rows) != expected_pairs:
        raise ValueError(
            f"{path}: expected {expected_pairs} pairs, found {len(rows)}"
        )
    if limit_samples is not None:
        rows = rows[:limit_samples]
    if not rows:
        raise ValueError(f"{path}: no usable rows")
    return rows


def validate_inputs(args: argparse.Namespace) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for source_model in args.source_models:
        for dataset in args.datasets:
            path = input_path(args, source_model, dataset)
            rows = read_rows(path, args.expected_pairs, args.limit_samples)
            counts[(source_model, dataset)] = len(rows)
            print(
                f"OK {source_model}/{dataset}: {len(rows)} pairs; "
                f"version={input_version(source_model)}; path={path}",
                flush=True,
            )
    return counts


def get_torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_model(args: argparse.Namespace):
    local_only = not args.allow_online
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.scoring_model_path),
        use_fast=True,
        local_files_only=local_only,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        str(args.scoring_model_path),
        torch_dtype=get_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        local_files_only=local_only,
        trust_remote_code=False,
    )
    model.to(args.device)
    model.eval()
    model.config.use_cache = False
    print(
        "Scoring model loaded once: "
        f"{type(model).__module__}.{type(model).__name__} on {args.device}"
    )
    return tokenizer, model


@torch.inference_mode()
def score_text_batch(
    texts: Sequence[str],
    *,
    tokenizer,
    model,
    device: str,
    max_length: int,
    token_chunk_size: int,
) -> list[TextScores]:
    encoded = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    if input_ids.shape[1] < 2:
        raise ValueError("a tokenized text has fewer than two tokens")
    full_lengths = attention_mask.sum(dim=1)
    over_limit = (full_lengths > max_length).nonzero(as_tuple=False).flatten()
    if len(over_limit):
        details = ", ".join(
            f"batch[{int(index)}]={int(full_lengths[index])}"
            for index in over_limit.tolist()
        )
        raise ValueError(
            "full-text scoring refuses silent truncation: token lengths exceed "
            f"--max-length={max_length}: {details}"
        )
    model_context = getattr(model.config, "max_position_embeddings", None)
    if model_context is not None:
        over_context = (full_lengths > int(model_context)).nonzero(
            as_tuple=False
        ).flatten()
        if len(over_context):
            details = ", ".join(
                f"batch[{int(index)}]={int(full_lengths[index])}"
                for index in over_context.tolist()
            )
            raise ValueError(
                "full text exceeds scoring-model context "
                f"({model_context} tokens): {details}"
            )

    # Keep full logits in model dtype. Only a small token slice is promoted to
    # FP32, avoiding full [batch,tokens,vocab] FP32 probability tensors.
    model_output = model(**encoded, use_cache=False)
    logits = model_output.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    valid = attention_mask[:, 1:].bool()
    batch, positions = labels.shape

    sums = {
        name: torch.zeros(batch, dtype=torch.float64, device=device)
        for name in ("likelihood", "entropy", "rank", "logrank")
    }
    counts = valid.sum(dim=1)
    if bool((counts < 1).any()):
        bad = (counts < 1).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"texts with no scored tokens: {bad}")

    for start in range(0, positions, token_chunk_size):
        stop = min(start + token_chunk_size, positions)
        chunk = logits[:, start:stop, :].float()
        chunk_labels = labels[:, start:stop]
        chunk_valid = valid[:, start:stop].to(torch.float64)

        observed_logits = chunk.gather(
            dim=-1, index=chunk_labels.unsqueeze(-1)
        ).squeeze(-1)
        log_normalizer = torch.logsumexp(chunk, dim=-1)
        observed_log_prob = observed_logits - log_normalizer

        ranks = (
            (chunk > observed_logits.unsqueeze(-1)).sum(dim=-1).to(torch.float64)
            + 1.0
        )
        log_ranks = torch.log(ranks)

        probabilities = torch.softmax(chunk, dim=-1)
        entropy = log_normalizer - (probabilities * chunk).sum(dim=-1)

        sums["likelihood"] += (
            observed_log_prob.to(torch.float64) * chunk_valid
        ).sum(dim=1)
        sums["entropy"] += (
            entropy.to(torch.float64) * chunk_valid
        ).sum(dim=1)
        sums["rank"] += (ranks * chunk_valid).sum(dim=1)
        sums["logrank"] += (log_ranks * chunk_valid).sum(dim=1)

        del chunk, probabilities, ranks, log_ranks

    outputs: list[TextScores] = []
    for index in range(batch):
        n_tokens = int(counts[index].item())
        likelihood = float((sums["likelihood"][index] / n_tokens).item())
        entropy = float((sums["entropy"][index] / n_tokens).item())
        mean_rank = float((sums["rank"][index] / n_tokens).item())
        mean_logrank = float((sums["logrank"][index] / n_tokens).item())
        if mean_logrank < 0:
            raise ValueError(
                f"text index {index}: negative mean log rank"
            )
        zero_logrank = mean_logrank == 0.0
        lrr = (
            LRR_ZERO_LOGRANK_SENTINEL
            if zero_logrank
            else -likelihood / mean_logrank
        )
        outputs.append(
            TextScores(
                likelihood=likelihood,
                entropy=entropy,
                rank=-mean_rank,
                logrank=-mean_logrank,
                lrr=lrr,
                n_tokens=n_tokens,
                lrr_zero_logrank=zero_logrank,
            )
        )

    del model_output, logits
    return outputs


def chunks(items: Sequence, size: int) -> Iterable[Sequence]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_file_fingerprint(model_path: Path) -> list[dict[str, int | str]]:
    score_config_names = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    )
    records: list[dict[str, int | str]] = []
    for name in score_config_names:
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
    *,
    source_path: Path,
    source_model: str,
    dataset: str,
    n_pairs: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "score_format_version": SCORE_FORMAT_VERSION,
        "protocol": "black-box/fixed-surrogate/independent",
        "scoring_region": "full-text",
        "source_model": source_model,
        "dataset": dataset,
        "input_version": input_version(source_model),
        "input_path": str(source_path.resolve()),
        "input_sha256": sha256_file(source_path),
        "n_pairs": n_pairs,
        "scoring_model_path": str(args.scoring_model_path.resolve()),
        "scoring_model_files": model_file_fingerprint(args.scoring_model_path),
        "methods": list(METHODS),
        "text_fields": list(TEXT_FIELDS),
        "max_length": args.max_length,
        "dtype": args.dtype,
        "limit_samples": args.limit_samples,
        "lrr_zero_logrank_policy": {
            "meaning": "extended-real +infinity (all observed ranks are 1)",
            "json_sentinel": LRR_ZERO_LOGRANK_SENTINEL,
        },
        "trust_remote_code": False,
    }


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path, json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    )


def prepare_cell_checkpoint(
    output_dir: Path,
    metadata: dict[str, Any],
    overwrite: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / "scores.jsonl"
    metadata_path = output_dir / "metadata.json"
    result_path = output_dir / "result.json"

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


def load_completed_scores(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: corrupt partial score file"
                ) from exc
            if row.get("index") != len(rows):
                raise ValueError(
                    f"{path}:{line_number}: non-contiguous index; use "
                    "--overwrite to recompute this cell"
                )
            for field in TEXT_FIELDS:
                values = row.get(field)
                if not isinstance(values, dict):
                    raise ValueError(f"{path}:{line_number}: missing {field}")
                for method in METHODS:
                    if not np.isfinite(values.get(method, np.nan)):
                        raise ValueError(
                            f"{path}:{line_number}: invalid {field}/{method}"
                        )
                if int(values.get("n_tokens", 0)) < 1:
                    raise ValueError(
                        f"{path}:{line_number}: invalid {field}/n_tokens"
                    )
                if not isinstance(values.get("lrr_zero_logrank"), bool):
                    raise ValueError(
                        f"{path}:{line_number}: missing "
                        f"{field}/lrr_zero_logrank diagnostic"
                    )
            rows.append(row)
    return rows


def score_cell(
    rows: list[dict[str, str]],
    *,
    score_path: Path,
    source_model: str,
    dataset: str,
    tokenizer,
    model,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    completed = load_completed_scores(score_path)
    if len(completed) > len(rows):
        raise ValueError(
            f"{score_path} has more rows than current input; use --overwrite"
        )
    if len(completed) == len(rows):
        print(f"SKIP complete {source_model}/{dataset}")
        return completed

    remaining = rows[len(completed) :]
    with score_path.open("a", encoding="utf-8") as stream:
        for row_batch in chunks(remaining, args.batch_size):
            # Scoring each field separately avoids tripling the logits peak.
            by_field: dict[str, list[TextScores]] = {}
            for field in TEXT_FIELDS:
                by_field[field] = score_text_batch(
                    [row[field] for row in row_batch],
                    tokenizer=tokenizer,
                    model=model,
                    device=args.device,
                    max_length=args.max_length,
                    token_chunk_size=args.token_chunk_size,
                )

            for offset in range(len(row_batch)):
                record: dict[str, Any] = {"index": len(completed)}
                for field in TEXT_FIELDS:
                    record[field] = asdict(by_field[field][offset])
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                completed.append(record)
            if (
                len(completed) % args.progress_every == 0
                or len(completed) == len(rows)
            ):
                print(
                    f"{source_model}/{dataset}: "
                    f"{len(completed)}/{len(rows)} pairs",
                    flush=True,
                )
    return completed


def conformal_pvalues(
    calibration: np.ndarray, test_scores: np.ndarray
) -> np.ndarray:
    ordered = np.sort(calibration)
    counts_ge = len(ordered) - np.searchsorted(
        ordered, test_scores, side="left"
    )
    return (1.0 + counts_ge) / (len(ordered) + 1.0)


def labels_and_scores(
    human: np.ndarray, machine: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.concatenate(
        [
            np.zeros(len(human), dtype=np.int8),
            np.ones(len(machine), dtype=np.int8),
        ]
    )
    return labels, np.concatenate([human, machine])


def normalized_partial_auc(
    human: np.ndarray, machine: np.ndarray, max_fpr: float
) -> float:
    labels, scores = labels_and_scores(human, machine)
    fpr, tpr, _ = roc_curve(labels, scores)
    if max_fpr not in fpr:
        index = int(np.searchsorted(fpr, max_fpr))
        left = max(index - 1, 0)
        right = min(index, len(fpr) - 1)
        if fpr[right] == fpr[left]:
            tpr_at = tpr[right]
        else:
            weight = (max_fpr - fpr[left]) / (fpr[right] - fpr[left])
            tpr_at = tpr[left] + weight * (tpr[right] - tpr[left])
        fpr = np.insert(fpr, index, max_fpr)
        tpr = np.insert(tpr, index, tpr_at)
    keep = fpr <= max_fpr
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(tpr[keep], fpr[keep]) / max_fpr)


def tpr_at_fpr(
    human: np.ndarray, machine: np.ndarray, max_fpr: float
) -> float:
    labels, scores = labels_and_scores(human, machine)
    fpr, tpr, _ = roc_curve(labels, scores)
    eligible = tpr[fpr <= max_fpr]
    return float(eligible.max()) if len(eligible) else 0.0


def higher_quantile(values: np.ndarray, probability: float) -> float:
    try:
        return float(np.quantile(values, probability, method="higher"))
    except TypeError:
        return float(np.quantile(values, probability, interpolation="higher"))


def stable_mean(values: np.ndarray) -> float:
    """Mean without overflowing when LRR contains the finite +inf sentinel."""
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return 0.0
    return float(np.mean(values / scale) * scale)


def metrics_for_method(
    records: Sequence[dict[str, Any]], method: str
) -> dict[str, float | int]:
    human = np.asarray(
        [record["human_text"][method] for record in records], dtype=np.float64
    )
    machine = np.asarray(
        [record["machine_text"][method] for record in records], dtype=np.float64
    )
    calibration = np.asarray(
        [record["calibration_text"][method] for record in records],
        dtype=np.float64,
    )
    for name, values in (
        ("human", human),
        ("machine", machine),
        ("calibration", calibration),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"{method}: non-finite {name} scores")

    labels, scores = labels_and_scores(human, machine)
    precision, recall, _ = precision_recall_curve(labels, scores)
    human_p = conformal_pvalues(calibration, human)
    machine_p = conformal_pvalues(calibration, machine)
    result: dict[str, float | int] = {
        "n_calibration": len(calibration),
        "n_human_test": len(human),
        "n_machine_test": len(machine),
        "auroc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(sklearn_auc(recall, precision)),
        "pauc_001": normalized_partial_auc(human, machine, 0.01),
        "pauc_005": normalized_partial_auc(human, machine, 0.05),
        "tpr_at_fpr_001": tpr_at_fpr(human, machine, 0.01),
        "tpr_at_fpr_005": tpr_at_fpr(human, machine, 0.05),
        "human_mean": stable_mean(human),
        "machine_mean": stable_mean(machine),
        "calibration_mean": stable_mean(calibration),
    }
    for alpha, conformal_suffix, legacy_suffix in ALPHAS:
        fpr = float(np.mean(human_p <= alpha))
        power = float(np.mean(machine_p <= alpha))
        result[f"conformal_fpr_{conformal_suffix}"] = fpr
        result[f"conformal_power_{conformal_suffix}"] = power
        result[f"test_fpr_{legacy_suffix}"] = fpr
        result[f"power_{legacy_suffix}"] = power
        result[f"threshold_{legacy_suffix}"] = higher_quantile(
            calibration, 1.0 - alpha
        )
    return result


def write_cell_result(
    output_dir: Path,
    *,
    source_model: str,
    dataset: str,
    records: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    definitions = {
        "likelihood": "mean observed-token log probability",
        "entropy": "mean predictive entropy",
        "rank": "negative mean observed-token rank",
        "logrank": "negative mean log observed-token rank",
        "lrr": "negative likelihood divided by mean log rank",
    }
    result = {
        "protocol": "black-box/fixed-surrogate/independent",
        "score_format_version": SCORE_FORMAT_VERSION,
        "scoring_region": "full-text",
        "score_direction": "larger means more machine-like",
        "calibration_protocol": (
            "independent human calibration with finite-sample upper-tail "
            "conformal p-values"
        ),
        "source_model": source_model,
        "dataset": dataset,
        "n_pairs": len(records),
        "metadata": metadata,
        "score_definitions": definitions,
        "lrr_zero_logrank_counts": {
            field: sum(
                bool(record[field]["lrr_zero_logrank"])
                for record in records
            )
            for field in TEXT_FIELDS
        },
        "methods": {
            method: metrics_for_method(records, method) for method in METHODS
        },
    }
    atomic_write_json(output_dir / "result.json", result)


SUMMARY_FIELDS = (
    "source_model",
    "dataset",
    "method",
    "n_pairs",
    "auroc",
    "pr_auc",
    "pauc_001",
    "pauc_005",
    "tpr_at_fpr_001",
    "tpr_at_fpr_005",
    "conformal_fpr_001",
    "conformal_power_001",
    "conformal_fpr_005",
    "conformal_power_005",
    "test_fpr_1pct",
    "power_1pct",
    "threshold_1pct",
    "test_fpr_5pct",
    "power_5pct",
    "threshold_5pct",
    "human_mean",
    "machine_mean",
    "calibration_mean",
    "n_calibration",
    "n_human_test",
    "n_machine_test",
)


def write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def result_sort_key(row: dict[str, Any]) -> tuple[int, int, int]:
    def index_or_end(values: Sequence[str], value: str) -> int:
        try:
            return values.index(value)
        except ValueError:
            return len(values)

    return (
        index_or_end(DEFAULT_SOURCE_MODELS, str(row["source_model"])),
        index_or_end(DEFAULT_DATASETS, str(row["dataset"])),
        index_or_end(METHODS, str(row["method"])),
    )


def collect_completed_rows(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_root.glob("*/*/result.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("protocol") != "black-box/fixed-surrogate/independent":
            continue
        methods = value.get("methods")
        if not isinstance(methods, dict):
            continue
        for method in METHODS:
            metrics = methods.get(method)
            if not isinstance(metrics, dict):
                continue
            rows.append(
                {
                    "source_model": value["source_model"],
                    "dataset": value["dataset"],
                    "method": method,
                    "n_pairs": value["n_pairs"],
                    **metrics,
                }
            )
    rows.sort(key=result_sort_key)
    return rows


def write_summaries(output_root: Path) -> list[dict[str, Any]]:
    rows = collect_completed_rows(output_root)
    write_csv(output_root / "summary.csv", rows, SUMMARY_FIELDS)

    macro_rows: list[dict[str, Any]] = []
    metric_fields = (
        "auroc",
        "pr_auc",
        "pauc_001",
        "pauc_005",
        "conformal_fpr_001",
        "conformal_power_001",
        "conformal_fpr_005",
        "conformal_power_005",
    )
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        macro: dict[str, Any] = {"method": method, "cells": len(subset)}
        for field in metric_fields:
            macro[field] = (
                float(np.mean([float(row[field]) for row in subset]))
                if subset
                else math.nan
            )
        macro_rows.append(macro)
    write_csv(
        output_root / "macro.csv",
        macro_rows,
        ("method", "cells", *metric_fields),
    )
    return rows


def run_cell(
    *,
    source_model: str,
    dataset: str,
    tokenizer,
    model,
    args: argparse.Namespace,
) -> None:
    source_path = input_path(args, source_model, dataset)
    rows = read_rows(source_path, args.expected_pairs, args.limit_samples)
    output_dir = args.output_root / source_model / dataset
    metadata = expected_metadata(
        source_path=source_path,
        source_model=source_model,
        dataset=dataset,
        n_pairs=len(rows),
        args=args,
    )
    score_path = prepare_cell_checkpoint(
        output_dir, metadata, overwrite=args.overwrite
    )
    records = score_cell(
        rows,
        score_path=score_path,
        source_model=source_model,
        dataset=dataset,
        tokenizer=tokenizer,
        model=model,
        args=args,
    )
    write_cell_result(
        output_dir,
        source_model=source_model,
        dataset=dataset,
        records=records,
        metadata=metadata,
    )
    write_summaries(args.output_root)
    print(f"SAVED {output_dir / 'result.json'}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    validate_inputs(args)
    if args.check_only:
        print("All requested independent inputs passed validation.")
        return

    if not args.allow_online:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    tokenizer, model = load_model(args)

    failures: list[str] = []
    for source_model in args.source_models:
        for dataset in args.datasets:
            print(f"RUN {source_model}/{dataset}", flush=True)
            try:
                run_cell(
                    source_model=source_model,
                    dataset=dataset,
                    tokenizer=tokenizer,
                    model=model,
                    args=args,
                )
            except Exception as exc:
                label = f"{source_model}/{dataset}"
                failures.append(f"{label}: {type(exc).__name__}: {exc}")
                print(f"FAILED {failures[-1]}", flush=True)
                if not args.continue_on_error:
                    raise

    summary_rows = write_summaries(args.output_root)
    completed_cells = len(
        {(row["source_model"], row["dataset"]) for row in summary_rows}
    )
    print(
        f"SAVED {args.output_root / 'summary.csv'} "
        f"({completed_cells} completed cells, {len(summary_rows)} method rows)"
    )
    if failures:
        print("Failures:")
        for failure in failures:
            print(f"  {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
