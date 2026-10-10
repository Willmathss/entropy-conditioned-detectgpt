#!/usr/bin/env python3
"""Run three detectors in AdaDetectGPT's simple black-box protocol.

The target/source model only supplies already-generated human/machine text.
GPT-J-6B is the sampling/reference model and GPT-Neo-2.7B is the scoring
model. The official Fast-DetectGPT and AdaDetectGPT scripts are invoked
without modification. Water Tensor-Rayleigh uses the same target texts and
proxy models, with hyperparameters selected only on the other domains.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


SAMPLING_NAME = "gpt-j-6B"
SCORING_NAME = "gpt-neo-2.7B"
# AdaDetectGPT/scripts/model.py expands the short CLI names before checking
# cache_dir/local.<model>. These links must use the expanded names.
SAMPLING_FULLNAME = "EleutherAI/gpt-j-6B"
SCORING_FULLNAME = "EleutherAI/gpt-neo-2.7B"
DEFAULT_PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_OFFICIAL_REPO = DEFAULT_PROJECT / "third_party/AdaDetectGPT"
DEFAULT_SAMPLING_DIR = DEFAULT_PROJECT / "models/gpt-j-6b"
DEFAULT_SCORING_DIR = DEFAULT_PROJECT / "models/gpt-neo-2.7b"
DEFAULT_DATA_DIR = DEFAULT_PROJECT / "data/test/text"
DEFAULT_TRAIN_DATA_DIR = DEFAULT_PROJECT / "data/train/text"
DEFAULT_DATASETS = ("c4", "pubmed", "squad", "writing", "xsum")
DEFAULT_SOURCE_MODELS = (
    "falcon-7b",
    "llama-2-7b",
    "opt-2.7b",
    "opt-6.7b",
)
METHOD_NAMES = {
    "fast": "Fast-DetectGPT",
    "ada": "AdaDetectGPT",
    "tensor": "Tensor-Rayleigh",
}
METHOD_ORDER = {method: index for index, method in enumerate(METHOD_NAMES)}


@dataclass(frozen=True)
class Job:
    dataset: str
    source_model: str
    method: str
    command: tuple[str, ...]
    result_file: Path
    log_file: Path


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def raw_data_path(prefix: Path) -> Path:
    return Path(f"{prefix}.raw_data.json")


def result_filename(prefix: Path, method: str) -> Path:
    if method == "fast":
        return Path(f"{prefix}.sampling_discrepancy_analytic.json")
    if method == "ada":
        return Path(f"{prefix}.AdaDetectGPT.bspline.json")
    if method == "tensor":
        return Path(f"{prefix}.TensorRayleigh.json")
    raise ValueError(f"Unknown method: {method}")


def discover_result_jobs(
    result_dir: Path,
    log_dir: Path,
    adapter_dir: Path,
    planned_jobs: Iterable[Job] = (),
) -> list[Job]:
    """Discover every completed result represented by persistent adapters.

    The adapters provide an unambiguous source-model/dataset index, so names
    containing underscores or hyphens do not need to be reverse-parsed from a
    result filename. Planned jobs are included as an additional index for
    compatibility with hand-built/test jobs.
    """
    indexed: dict[tuple[str, str, str], Job] = {}
    for job in planned_jobs:
        if job.result_file.is_file():
            indexed[(job.dataset, job.source_model, job.method)] = job

    if adapter_dir.is_dir():
        for source_dir in sorted(adapter_dir.iterdir()):
            if not source_dir.is_dir():
                continue
            source_model = source_dir.name
            for adapter_file in sorted(source_dir.glob("*.raw_data.json")):
                dataset = adapter_file.name[: -len(".raw_data.json")]
                prefix = (
                    result_dir
                    / f"{dataset}_{source_model}.{SAMPLING_NAME}_{SCORING_NAME}"
                )
                for method in METHOD_NAMES:
                    result_file = result_filename(prefix, method)
                    if not result_file.is_file():
                        continue
                    indexed[(dataset, source_model, method)] = Job(
                        dataset=dataset,
                        source_model=source_model,
                        method=method,
                        command=(),
                        result_file=result_file,
                        log_file=(
                            log_dir / f"{dataset}_{source_model}.{method}.log"
                        ),
                    )

    return sorted(
        indexed.values(),
        key=lambda job: (
            job.source_model,
            job.dataset,
            METHOD_ORDER[job.method],
        ),
    )


def validate_model_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} model directory does not exist: {path}")
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"{label} is missing config.json: {path}")

    weight_candidates = [
        *path.glob("model*.safetensors"),
        *path.glob("pytorch_model*.bin"),
    ]
    has_index = any(
        (path / name).is_file()
        for name in (
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
    )
    if not weight_candidates and not has_index:
        raise FileNotFoundError(
            f"{label} has no PyTorch/safetensors weights in {path}"
        )

    has_tokenizer = (path / "tokenizer.json").is_file() or (
        (path / "vocab.json").is_file() and (path / "merges.txt").is_file()
    )
    if not has_tokenizer:
        raise FileNotFoundError(
            f"{label} is missing tokenizer.json or vocab.json + merges.txt: {path}"
        )


def ensure_local_link(cache_dir: Path, model_fullname: str, target: Path) -> Path:
    """Create the exact local.<fullname> path checked by official model.py."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    link = cache_dir / f"local.{model_fullname.replace('/', '_')}"
    target = target.resolve()
    if link.is_symlink():
        if link.resolve() != target:
            raise FileExistsError(
                f"{link} already points to {link.resolve()}, expected {target}"
            )
        return link
    if link.exists():
        raise FileExistsError(
            f"{link} already exists and is not a symlink; refusing to replace it"
        )
    link.symlink_to(target, target_is_directory=True)
    return link


def check_tokenizer_compatibility(sampling_dir: Path, scoring_dir: Path) -> None:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required; install the AdaDetectGPT requirements first"
        ) from exc

    kwargs = {
        "local_files_only": True,
        "use_fast": True,
    }
    sampling = AutoTokenizer.from_pretrained(str(sampling_dir), **kwargs)
    scoring = AutoTokenizer.from_pretrained(str(scoring_dir), **kwargs)
    if sampling.vocab_size != scoring.vocab_size:
        raise ValueError(
            "Proxy tokenizer vocabulary sizes differ: "
            f"{sampling.vocab_size} vs {scoring.vocab_size}"
        )

    probes = [
        "AdaDetectGPT black-box tokenizer check.",
        "A short sentence with punctuation: 1, 2, 3.",
        "Leading spaces and\na newline are intentional.",
    ]
    for probe in probes:
        left = sampling.encode(probe, add_special_tokens=True)
        right = scoring.encode(probe, add_special_tokens=True)
        if left != right:
            raise ValueError(
                "GPT-J and GPT-Neo tokenizers produce different token IDs. "
                "Recheck the downloaded tokenizer files."
            )


def strict_data_path(
    data_dir: Path,
    source_model: str,
    dataset: str,
) -> Path:
    return data_dir / source_model / f"{dataset}.jsonl"


def load_strict_jsonl(path: Path) -> dict[str, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict] = []
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
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no JSONL records")

    fields = ("human_text", "machine_text", "calibration_text")
    for line_number, row in enumerate(rows, start=1):
        for field in fields:
            text = row.get(field)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"{path}:{line_number}: {field!r} must be non-empty text"
                )
    return {
        "original": [row["human_text"] for row in rows],
        "sampled": [row["machine_text"] for row in rows],
        "calibration": [row["calibration_text"] for row in rows],
    }


def prepare_official_adapter(
    source_path: Path,
    adapter_prefix: Path,
    limit_samples: int,
) -> Path:
    """Convert generated_strict JSONL to the official raw_data JSON interface."""
    data = load_strict_jsonl(source_path)
    if limit_samples > 0 and len(data["original"]) < limit_samples:
        raise ValueError(
            f"{source_path} has {len(data['original'])} samples, "
            f"fewer than --limit-samples={limit_samples}"
        )
    selected = (
        {
            key: values[:limit_samples]
            for key, values in data.items()
        }
        if limit_samples > 0
        else data
    )
    output = raw_data_path(adapter_prefix)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        try:
            with output.open("r", encoding="utf-8") as stream:
                existing = json.load(stream)
        except (json.JSONDecodeError, OSError):
            existing = None
        if existing == selected:
            return adapter_prefix
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(selected, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, output)
    return adapter_prefix


def build_jobs(
    *,
    python: str,
    project_root: Path,
    official_repo_root: Path | None = None,
    data_prefixes: dict[tuple[str, str], Path],
    train_data_prefixes: dict[tuple[str, str], Path] | None = None,
    result_dir: Path,
    log_dir: Path,
    datasets: Sequence[str],
    source_models: Sequence[str],
    methods: Sequence[str],
    cache_dir: Path,
    device: str,
    num_subsample: int,
    sampling_model_dir: Path | None = None,
    scoring_model_dir: Path | None = None,
    tensor_cache_root: Path | None = None,
    tensor_expectation_mode: str = "full_vocab",
    tensor_num_samples: int = 128,
    tensor_max_length: int = 2048,
    tensor_chunk_tokens: int = 16,
    tensor_document_batch_size: int = 2,
    tensor_request_batch_size: int = 15,
    tensor_sampling_device: str = "auto",
    tensor_scoring_device: str = "auto",
    tensor_structures: str = "4x4,6x6,8x6",
    tensor_smooth_lambdas: str = "0,1e-3,1e-2,1e-1,1",
    tensor_ridge_lambdas: str = "1e-5,1e-4,1e-3,1e-2",
    tensor_covariance_shrinkages: str = "0,0.05,0.2",
    tensor_overwrite_cache: bool = False,
) -> list[Job]:
    jobs: list[Job] = []
    if train_data_prefixes is None:
        train_data_prefixes = data_prefixes
    tensor_cache_rebuilt_for: set[str] = set()
    official_scripts = (
        official_repo_root if official_repo_root is not None else project_root
    ) / "scripts"
    work_scripts = project_root / "experiments"
    for dataset in datasets:
        train_domains = [item for item in datasets if item != dataset]
        if not train_domains:
            raise ValueError("Ada requires at least two datasets in total")
        for source_model in source_models:
            output_prefix = (
                result_dir
                / f"{dataset}_{source_model}.{SAMPLING_NAME}_{SCORING_NAME}"
            )
            for method in methods:
                common = [
                    "--sampling_model_name",
                    SAMPLING_NAME,
                    "--scoring_model_name",
                    SCORING_NAME,
                    "--dataset",
                    dataset,
                    "--dataset_file",
                    str(data_prefixes[(dataset, source_model)]),
                    "--output_file",
                    str(output_prefix),
                    "--device",
                    device,
                    "--cache_dir",
                    str(cache_dir),
                ]
                if method == "fast":
                    command = [
                        python,
                        str(official_scripts / "detect_gpt_fast.py"),
                        *common,
                        "--discrepancy_analytic",
                    ]
                elif method == "ada":
                    train_prefixes = [
                        str(train_data_prefixes[(domain, source_model)])
                        for domain in train_domains
                    ]
                    command = [
                        python,
                        str(official_scripts / "detect_gpt_ada.py"),
                        *common,
                        "--train_dataset",
                        "&".join(train_prefixes),
                        "--num_subsample",
                        str(num_subsample),
                    ]
                elif method == "tensor":
                    if sampling_model_dir is None or scoring_model_dir is None:
                        raise ValueError(
                            "Tensor-Rayleigh requires both local proxy model directories"
                        )
                    if tensor_cache_root is None:
                        raise ValueError("Tensor-Rayleigh requires tensor_cache_root")
                    command = [
                        python,
                        str(work_scripts / "detect_tensor_rayleigh_blackbox.py"),
                        "--dataset",
                        dataset,
                        "--source-model",
                        source_model,
                        "--dataset-file",
                        str(data_prefixes[(dataset, source_model)]),
                        "--output-file",
                        str(output_prefix),
                        "--sampling-model-dir",
                        str(sampling_model_dir),
                        "--scoring-model-dir",
                        str(scoring_model_dir),
                        "--cache-root",
                        str(tensor_cache_root),
                        "--expectation-mode",
                        tensor_expectation_mode,
                        "--num-samples",
                        str(tensor_num_samples),
                        "--max-length",
                        str(tensor_max_length),
                        "--chunk-tokens",
                        str(tensor_chunk_tokens),
                        "--document-batch-size",
                        str(tensor_document_batch_size),
                        "--request-batch-size",
                        str(tensor_request_batch_size),
                        "--sampling-device",
                        tensor_sampling_device,
                        "--scoring-device",
                        tensor_scoring_device,
                        "--structures",
                        tensor_structures,
                        "--smooth-lambdas",
                        tensor_smooth_lambdas,
                        "--ridge-lambdas",
                        tensor_ridge_lambdas,
                        "--covariance-shrinkages",
                        tensor_covariance_shrinkages,
                    ]
                    for domain in train_domains:
                        command.extend(
                            [
                                "--train-domain",
                                (
                                    f"{domain}="
                                    f"{train_data_prefixes[(domain, source_model)]}"
                                ),
                            ]
                        )
                    if (
                        tensor_overwrite_cache
                        and source_model not in tensor_cache_rebuilt_for
                    ):
                        command.append("--overwrite-cache")
                        tensor_cache_rebuilt_for.add(source_model)
                else:
                    raise ValueError(f"Unknown method: {method}")
                jobs.append(
                    Job(
                        dataset=dataset,
                        source_model=source_model,
                        method=method,
                        command=tuple(command),
                        result_file=result_filename(output_prefix, method),
                        log_file=(
                            log_dir
                            / f"{dataset}_{source_model}.{method}.log"
                        ),
                    )
                )
    return jobs


def run_job(job: Job, env: dict[str, str]) -> None:
    job.log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nRUN [{job.dataset}/{job.source_model}/{job.method}]")
    print(shlex.join(job.command))
    with job.log_file.open("w", encoding="utf-8") as log:
        log.write(f"$ {shlex.join(job.command)}\n\n")
        log.flush()
        process = subprocess.Popen(
            list(job.command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="")
                log.write(line)
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, list(job.command))
    if not job.result_file.is_file():
        raise FileNotFoundError(
            f"Detector exited successfully but did not create {job.result_file}"
        )


def read_result_row(job: Job) -> dict:
    with job.result_file.open("r", encoding="utf-8") as stream:
        result = json.load(stream)
    metrics = result.get("metrics", {})
    pr_metrics = result.get("pr_metrics", {})
    predictions = result.get("predictions", {})
    return {
        "dataset": job.dataset,
        "source_model": job.source_model,
        "sampling_model": SAMPLING_NAME,
        "scoring_model": SCORING_NAME,
        "method": METHOD_NAMES[job.method],
        "roc_auc": metrics.get("roc_auc", ""),
        "pr_auc": pr_metrics.get("pr_auc", ""),
        "n_human": len(predictions.get("real", [])),
        "n_machine": len(predictions.get("samples", [])),
        "result_file": str(job.result_file),
    }


def write_summary(jobs: Iterable[Job], output: Path) -> list[dict]:
    rows = [read_result_row(job) for job in jobs if job.result_file.is_file()]
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "source_model",
        "sampling_model",
        "scoring_model",
        "method",
        "roc_auc",
        "pr_auc",
        "n_human",
        "n_machine",
        "result_file",
    ]
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output)
    return rows


def merge_unique(existing: Iterable[str], current: Iterable[str]) -> list[str]:
    return list(dict.fromkeys([*existing, *current]))


def merge_commands(
    existing: Iterable[Sequence[str]],
    current: Iterable[Sequence[str]],
) -> list[list[str]]:
    merged: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for command in [*existing, *current]:
        key = tuple(str(part) for part in command)
        if key in seen:
            continue
        seen.add(key)
        merged.append(list(key))
    return merged


def write_merged_manifest(
    current: dict,
    output: Path,
    completed_jobs: Iterable[Job],
) -> dict:
    """Merge invocation history while rebuilding completion state from disk."""
    existing: dict = {}
    if output.is_file():
        try:
            with output.open("r", encoding="utf-8") as stream:
                loaded = json.load(stream)
            if isinstance(loaded, dict):
                existing = loaded
        except (json.JSONDecodeError, OSError):
            existing = {}

    completed = list(completed_jobs)
    merged = {**existing, **current}
    merged["manifest_version"] = 2
    merged["datasets"] = merge_unique(
        existing.get("datasets", []),
        [
            *current.get("datasets", []),
            *(job.dataset for job in completed),
        ],
    )
    merged["source_models"] = merge_unique(
        existing.get("source_models", []),
        [
            *current.get("source_models", []),
            *(job.source_model for job in completed),
        ],
    )
    merged["methods"] = merge_unique(
        existing.get("methods", []),
        [
            *current.get("methods", []),
            *(job.method for job in completed),
        ],
    )
    merged["input_counts"] = {
        **existing.get("input_counts", {}),
        **current.get("input_counts", {}),
    }
    merged["commands"] = merge_commands(
        existing.get("commands", []),
        current.get("commands", []),
    )
    merged["current_invocation"] = {
        "datasets": list(current.get("datasets", [])),
        "source_models": list(current.get("source_models", [])),
        "methods": list(current.get("methods", [])),
        "commands": [list(command) for command in current.get("commands", [])],
    }
    merged["completed_result_count"] = len(completed)
    merged["completed_results"] = [
        {
            "dataset": job.dataset,
            "source_model": job.source_model,
            "method": job.method,
            "result_file": str(job.result_file),
        }
        for job in completed
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(merged, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, output)
    return merged


def validate_project(
    project_root: Path,
    official_repo_root: Path,
    methods: Sequence[str],
) -> None:
    required = []
    if "fast" in methods:
        required.append(
            official_repo_root / "scripts" / "detect_gpt_fast.py"
        )
    if "ada" in methods:
        required.append(
            official_repo_root / "scripts" / "detect_gpt_ada.py"
        )
    if any(method in {"fast", "ada"} for method in methods):
        required.append(official_repo_root / "scripts" / "model.py")
    if "tensor" in methods:
        required.append(
            project_root / "experiments" / "detect_tensor_rayleigh_blackbox.py"
        )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing AdaDetectGPT files: {missing}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run AdaDetectGPT simple black-box evaluation using local models"
    )
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument(
        "--official-repo-root",
        type=Path,
        default=DEFAULT_OFFICIAL_REPO,
        help=(
            "Official AdaDetectGPT repository containing scripts/"
            "detect_gpt_fast.py, detect_gpt_ada.py, and model.py"
        ),
    )
    parser.add_argument("--sampling-model-dir", type=Path, default=DEFAULT_SAMPLING_DIR)
    parser.add_argument("--scoring-model-dir", type=Path, default=DEFAULT_SCORING_DIR)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=(
            "independent test text root containing "
            "<source-model>/<dataset>.jsonl"
        ),
    )
    parser.add_argument(
        "--train-data-dir",
        type=Path,
        default=DEFAULT_TRAIN_DATA_DIR,
        help=(
            "Frozen development text root used only for Ada/Tensor training; "
            "defaults to data/train/text."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument(
        "--source-models",
        default=",".join(DEFAULT_SOURCE_MODELS),
        help="Comma-separated target generators",
    )
    parser.add_argument(
        "--methods",
        default="fast,ada,tensor",
        help="Comma-separated subset of fast,ada,tensor",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--num-subsample",
        type=int,
        default=-1,
        help="Ada witness training pairs per source domain; -1 uses all",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=0,
        help="Use a project-local N-sample subset for a smoke test; 0 uses all",
    )
    parser.add_argument(
        "--tensor-expectation-mode",
        choices=("full_vocab", "sampled"),
        default="full_vocab",
        help="Default: exact full shared vocabulary; sampled is the 128-candidate ablation",
    )
    parser.add_argument("--tensor-num-samples", type=int, default=128)
    parser.add_argument("--tensor-max-length", type=int, default=2048)
    parser.add_argument("--tensor-chunk-tokens", type=int, default=16)
    parser.add_argument("--tensor-document-batch-size", type=int, default=2)
    parser.add_argument("--tensor-request-batch-size", type=int, default=15)
    parser.add_argument("--tensor-sampling-device", default="auto")
    parser.add_argument("--tensor-scoring-device", default="auto")
    parser.add_argument("--tensor-structures", default="4x4,6x6,8x6")
    parser.add_argument(
        "--tensor-smooth-lambdas",
        default="0,1e-3,1e-2,1e-1,1",
    )
    parser.add_argument(
        "--tensor-ridge-lambdas",
        default="1e-5,1e-4,1e-3,1e-2",
    )
    parser.add_argument(
        "--tensor-covariance-shrinkages",
        default="0,0.05,0.2",
    )
    parser.add_argument(
        "--tensor-overwrite-cache",
        action="store_true",
        help="Rebuild Tensor-Rayleigh's proxy-token cache",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--rebuild-summary-only",
        action="store_true",
        help=(
            "Rebuild summary.csv and manifest completion inventory from "
            "existing results without validating/loading models or running jobs"
        ),
    )
    parser.add_argument(
        "--skip-tokenizer-check",
        action="store_true",
        help="Skip the lightweight local tokenizer compatibility check",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later cells if one detector command fails",
    )
    args = parser.parse_args(argv)

    project_root = args.project_root.resolve()
    official_repo_root = args.official_repo_root.resolve()
    sampling_dir = args.sampling_model_dir.resolve()
    scoring_dir = args.scoring_model_dir.resolve()
    data_dir = args.data_dir.resolve()
    train_data_dir = (
        args.train_data_dir.resolve()
        if args.train_data_dir is not None
        else data_dir
    )
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else project_root / "outputs" / "blackbox"
    )
    cache_dir = (
        args.cache_dir.resolve()
        if args.cache_dir is not None
        else project_root / "cache"
    )

    if args.rebuild_summary_only:
        if args.limit_samples < 0:
            parser.error("--limit-samples must be non-negative")
        subset_tag = (
            f"smoke_n{args.limit_samples}" if args.limit_samples else "full"
        )
        run_root = output_root / subset_tag
        discovered = discover_result_jobs(
            run_root / "results",
            run_root / "logs",
            run_root / "input_adapters",
        )
        rows = write_summary(discovered, run_root / "summary.csv")
        write_merged_manifest(
            {
                "datasets": [],
                "source_models": [],
                "methods": [],
                "input_counts": {},
                "commands": [],
                "summary_only": True,
            },
            run_root / "manifest.json",
            discovered,
        )
        print(f"Rebuilt aggregate summary from {len(rows)} result files.")
        print(f"Summary: {run_root / 'summary.csv'}")
        return 0

    datasets = parse_csv(args.datasets)
    methods = parse_csv(args.methods)
    if len(datasets) < 2:
        parser.error("--datasets must contain at least two domains")
    if not datasets:
        parser.error("--datasets is empty")
    if any(method not in {"fast", "ada", "tensor"} for method in methods):
        parser.error("--methods accepts only fast,ada,tensor")
    if "tensor" in methods and len(datasets) < 3:
        parser.error(
            "Tensor-Rayleigh requires at least three datasets: "
            "one held-out test domain and two training domains"
        )
    if args.limit_samples < 0:
        parser.error("--limit-samples must be non-negative")

    if args.tensor_num_samples < 2:
        parser.error("--tensor-num-samples must be at least 2")
    if args.tensor_max_length < 2:
        parser.error("--tensor-max-length must be at least 2")
    if args.tensor_chunk_tokens < 1:
        parser.error("--tensor-chunk-tokens must be positive")
    if args.tensor_document_batch_size < 1:
        parser.error("--tensor-document-batch-size must be positive")
    if args.tensor_request_batch_size < 1:
        parser.error("--tensor-request-batch-size must be positive")

    validate_project(project_root, official_repo_root, methods)
    validate_model_dir(sampling_dir, "GPT-J-6B sampling")
    validate_model_dir(scoring_dir, "GPT-Neo-2.7B scoring")
    sampling_link = ensure_local_link(
        cache_dir, SAMPLING_FULLNAME, sampling_dir
    )
    scoring_link = ensure_local_link(
        cache_dir, SCORING_FULLNAME, scoring_dir
    )
    if not args.skip_tokenizer_check:
        check_tokenizer_compatibility(sampling_dir, scoring_dir)

    source_models = parse_csv(args.source_models)
    if not source_models:
        parser.error("--source-models is empty")

    subset_tag = f"smoke_n{args.limit_samples}" if args.limit_samples else "full"
    run_root = output_root / subset_tag
    result_dir = run_root / "results"
    log_dir = run_root / "logs"
    adapter_dir = run_root / "input_adapters"
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    data_prefixes: dict[tuple[str, str], Path] = {}
    train_data_prefixes: dict[tuple[str, str], Path] = {}
    counts: dict[tuple[str, str], int] = {}
    train_counts: dict[tuple[str, str], int] = {}
    for dataset in datasets:
        for source_model in source_models:
            source_path = strict_data_path(data_dir, source_model, dataset)
            data = load_strict_jsonl(source_path)
            counts[(dataset, source_model)] = len(data["original"])
            data_prefixes[(dataset, source_model)] = prepare_official_adapter(
                source_path,
                adapter_dir / source_model / dataset,
                args.limit_samples,
            )
            if train_data_dir == data_dir:
                train_counts[(dataset, source_model)] = counts[
                    (dataset, source_model)
                ]
                train_data_prefixes[(dataset, source_model)] = data_prefixes[
                    (dataset, source_model)
                ]
            else:
                train_source_path = strict_data_path(
                    train_data_dir, source_model, dataset
                )
                train_data = load_strict_jsonl(train_source_path)
                train_counts[(dataset, source_model)] = len(
                    train_data["original"]
                )
                train_data_prefixes[(dataset, source_model)] = (
                    prepare_official_adapter(
                        train_source_path,
                        adapter_dir / "train_development" / source_model / dataset,
                        args.limit_samples,
                    )
                )

    jobs = build_jobs(
        python=sys.executable,
        project_root=project_root,
        official_repo_root=official_repo_root,
        data_prefixes=data_prefixes,
        train_data_prefixes=train_data_prefixes,
        result_dir=result_dir,
        log_dir=log_dir,
        datasets=datasets,
        source_models=source_models,
        methods=methods,
        cache_dir=cache_dir,
        device=args.device,
        num_subsample=args.num_subsample,
        sampling_model_dir=sampling_dir,
        scoring_model_dir=scoring_dir,
        tensor_cache_root=run_root / "tensor_cache",
        tensor_expectation_mode=args.tensor_expectation_mode,
        tensor_num_samples=args.tensor_num_samples,
        tensor_max_length=args.tensor_max_length,
        tensor_chunk_tokens=args.tensor_chunk_tokens,
        tensor_document_batch_size=args.tensor_document_batch_size,
        tensor_request_batch_size=args.tensor_request_batch_size,
        tensor_sampling_device=args.tensor_sampling_device,
        tensor_scoring_device=args.tensor_scoring_device,
        tensor_structures=args.tensor_structures,
        tensor_smooth_lambdas=args.tensor_smooth_lambdas,
        tensor_ridge_lambdas=args.tensor_ridge_lambdas,
        tensor_covariance_shrinkages=(
            args.tensor_covariance_shrinkages
        ),
        tensor_overwrite_cache=args.tensor_overwrite_cache,
    )

    manifest = {
        "project_root": str(project_root),
        "official_repo_root": str(official_repo_root),
        "data_dir": str(data_dir),
        "train_data_dir": str(train_data_dir),
        "sampling_model_dir": str(sampling_dir),
        "scoring_model_dir": str(scoring_dir),
        "sampling_model_name": SAMPLING_NAME,
        "scoring_model_name": SCORING_NAME,
        "sampling_model_fullname": SAMPLING_FULLNAME,
        "scoring_model_fullname": SCORING_FULLNAME,
        "sampling_local_link": str(sampling_link),
        "scoring_local_link": str(scoring_link),
        "datasets": datasets,
        "source_models": source_models,
        "methods": methods,
        "input_counts": {
            f"{dataset}/{source}": count
            for (dataset, source), count in counts.items()
        },
        "train_input_counts": {
            f"{dataset}/{source}": count
            for (dataset, source), count in train_counts.items()
        },
        "input_schema": {
            "layout": "<data_dir>/<source_model>/<dataset>.jsonl",
            "fields": [
                "human_text",
                "machine_text",
                "calibration_text",
            ],
            "official_adapter_dir": str(adapter_dir),
            "training_adapter_dir": str(
                adapter_dir
                if train_data_dir == data_dir
                else adapter_dir / "train_development"
            ),
            "calibration_used_for_raw_auc": False,
        },
        "limit_samples": args.limit_samples,
        "num_subsample": args.num_subsample,
        "tensor": {
            "expectation_mode": args.tensor_expectation_mode,
            "num_samples": args.tensor_num_samples,
            "max_length": args.tensor_max_length,
            "chunk_tokens": args.tensor_chunk_tokens,
            "document_batch_size": args.tensor_document_batch_size,
            "request_batch_size": args.tensor_request_batch_size,
            "sampling_device": args.tensor_sampling_device,
            "scoring_device": args.tensor_scoring_device,
            "structures": args.tensor_structures,
            "smooth_lambdas": args.tensor_smooth_lambdas,
            "ridge_lambdas": args.tensor_ridge_lambdas,
            "covariance_shrinkages": (
                args.tensor_covariance_shrinkages
            ),
        },
        "commands": [list(job.command) for job in jobs],
    }
    manifest_path = run_root / "manifest.json"

    def refresh_outputs() -> list[dict]:
        discovered = discover_result_jobs(
            result_dir,
            log_dir,
            adapter_dir,
            jobs,
        )
        rows = write_summary(discovered, run_root / "summary.csv")
        write_merged_manifest(manifest, manifest_path, discovered)
        return rows

    write_merged_manifest(
        manifest,
        manifest_path,
        discover_result_jobs(result_dir, log_dir, adapter_dir, jobs),
    )

    print("Input check passed.")
    print(f"Work project: {project_root}")
    print(f"Official repository: {official_repo_root}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Source models: {', '.join(source_models)}")
    print(f"Jobs: {len(jobs)}")
    print(f"Output: {run_root}")
    if args.check_only:
        return 0

    if args.dry_run:
        for job in jobs:
            print(shlex.join(job.command))
        return 0

    environment = os.environ.copy()
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["HF_HUB_OFFLINE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    completed: list[Job] = []
    failures: list[tuple[Job, str]] = []
    for job in jobs:
        if job.result_file.is_file() and not args.overwrite:
            print(
                f"SKIP existing [{job.dataset}/{job.source_model}/{job.method}]: "
                f"{job.result_file}"
            )
            completed.append(job)
            continue
        try:
            run_job(job, environment)
            completed.append(job)
        except Exception as exc:
            failures.append((job, str(exc)))
            print(
                f"FAILED [{job.dataset}/{job.source_model}/{job.method}]: {exc}",
                file=sys.stderr,
            )
            if not args.continue_on_error:
                refresh_outputs()
                raise
        refresh_outputs()

    rows = refresh_outputs()
    print(f"\nCompleted result files in aggregate summary: {len(rows)}")
    print(f"Jobs requested by this invocation: {len(jobs)}")
    print(f"Summary: {run_root / 'summary.csv'}")
    if failures:
        print("Failures:", file=sys.stderr)
        for job, message in failures:
            print(
                f"  {job.dataset}/{job.source_model}/{job.method}: {message}",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
