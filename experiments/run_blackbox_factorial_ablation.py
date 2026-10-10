#!/usr/bin/env python3
"""Cache-first black-box H/S/C factorial ablation for Water/Tensor-Rayleigh.

This script is meant to be copied to ``AdaDetectGPT/scripts`` and run after
the black-box independent Tensor jobs have finished.  It reuses their token
and exact full-vocabulary feature caches.  It never loads a source generator.

The factorial is identical to the white-box definition:

* H: entropy conditioning.  H=0 is the exact partition-of-unity projection
  from B(z) x C(h) to B(z).
* S: spline smoothness selection.  S=0 fixes smooth_lambda=0.
* C: covariance shrinkage selection.  C=0 fixes shrinkage=0.

Every variant deploys the paper's final self-normalized score

    S_theta(X) = theta^T U_X / sqrt(max(theta^T V_X theta, epsilon)).

The independent test domain is never used for fitting or hyperparameter
selection.  The four development domains are selected by leave-one-domain-out
validation, exactly as in the main black-box Tensor experiment.  The existing
Full Tensor result is used verbatim as the factorial ``full`` cell and is
checked against the cached basis context.

Calibration-human texts are scored only after all directions are frozen.
Their exact features are content-addressed in the existing Tensor feature
store.  This is the only phase that may load GPT-J/GPT-Neo; all fitting and
test scoring are cache-only.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import itertools
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import (
    auc as sklearn_auc,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


VERSION = "2026.08.15-v1-blackbox-hsc-factorial"
DATASETS = ("c4", "pubmed", "squad", "writing", "xsum")
SOURCE_MODELS = ("falcon-7b", "llama-2-7b", "opt-2.7b", "opt-6.7b")
SAMPLING_NAME = "gpt-j-6B"
SCORING_NAME = "gpt-neo-2.7B"


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    entropy_conditioning: bool
    smoothness: bool
    covariance_shrinkage: bool


VARIANTS = (
    Variant("b0", "B0", False, False, False),
    Variant("h", "B0+H", True, False, False),
    Variant("s", "B0+S", False, True, False),
    Variant("c", "B0+C", False, False, True),
    Variant("hs", "B0+H+S", True, True, False),
    Variant("hc", "B0+H+C", True, False, True),
    Variant("sc", "B0+S+C", False, True, True),
    Variant("full", "Full Water", True, True, True),
)
VARIANT_BY_KEY = {item.key: item for item in VARIANTS}


@dataclass(frozen=True)
class AblationConfig:
    source_j_bases: int
    source_k_bases: int
    j_bases: int
    k_bases: int
    smooth_lambda: float
    ridge_lambda: float
    covariance_shrinkage: float

    @property
    def dimension(self) -> int:
        return self.j_bases * self.k_bases

    @property
    def source_structure(self) -> tuple[int, int]:
        return self.source_j_bases, self.source_k_bases


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_float_csv(value: str) -> list[float]:
    return [float(item) for item in parse_csv(value)]


def parse_structures(value: str) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for item in parse_csv(value):
        pieces = item.lower().split("x")
        if len(pieces) != 2:
            raise ValueError(f"invalid structure {item!r}; expected JxK")
        j_bases, k_bases = (int(piece) for piece in pieces)
        if j_bases < 3 or k_bases < 3:
            raise ValueError("each source spline direction needs >=3 bases")
        result.append((j_bases, k_bases))
    if not result:
        raise ValueError("at least one --structure is required")
    return result


def import_file(path: Path, module_name: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_digest(texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(b"water-blackbox-factorial-calibration-v1\0")
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def result_path(run_root: Path, dataset: str, source: str) -> Path:
    prefix = run_root / "results" / (
        f"{dataset}_{source}.{SAMPLING_NAME}_{SCORING_NAME}"
    )
    return Path(f"{prefix}.TensorRayleigh.json")


def adapter_path(
    run_root: Path, dataset: str, source: str, *, development: bool
) -> Path:
    root = run_root / "input_adapters"
    if development:
        root = root / "train_development"
    return root / source / f"{dataset}.raw_data.json"


def adapter_prefix(path: Path) -> Path:
    suffix = ".raw_data.json"
    value = str(path)
    if not value.endswith(suffix):
        raise ValueError(f"unexpected adapter filename: {path}")
    return Path(value[: -len(suffix)])


def load_adapter(path: Path) -> dict[str, list[str]]:
    value = load_json(path)
    result: dict[str, list[str]] = {}
    for key in ("original", "sampled", "calibration"):
        texts = value.get(key)
        if not isinstance(texts, list) or not texts:
            raise ValueError(f"{path}: missing non-empty {key!r} list")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError(f"{path}: invalid text in {key!r}")
        result[key] = texts
    if len(result["original"]) != len(result["sampled"]):
        raise ValueError(f"{path}: human/machine test counts differ")
    return result


def validate_inputs(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    for source in args.source_models_list:
        for dataset in args.datasets_list:
            base = result_path(args.run_root, dataset, source)
            if not base.is_file():
                errors.append(f"missing main Tensor result: {base}")
                continue
            try:
                result = load_json(base)
                real = result.get("predictions", {}).get("real")
                samples = result.get("predictions", {}).get("samples")
                if not isinstance(real, list) or not isinstance(samples, list):
                    raise ValueError("missing prediction arrays")
                if len(real) != len(samples) or not real:
                    raise ValueError("invalid prediction lengths")
                selected = result.get("selected_hyperparameters", {})
                for key in (
                    "j_bases",
                    "k_bases",
                    "smooth_lambda",
                    "ridge_lambda",
                    "covariance_shrinkage",
                ):
                    if key not in selected:
                        raise ValueError(f"missing selected hyperparameter {key}")
                theta = np.asarray(
                    result.get("model", {}).get("theta"), dtype=np.float64
                )
                if theta.ndim != 1 or not len(theta) or not np.isfinite(theta).all():
                    raise ValueError("invalid frozen theta")
            except Exception as exc:  # report all cells together
                errors.append(f"invalid main Tensor result {base}: {exc}")
            for development, domains in (
                (False, (dataset,)),
                (True, tuple(name for name in DATASETS if name != dataset)),
            ):
                for domain in domains:
                    path = adapter_path(
                        args.run_root, domain, source, development=development
                    )
                    if not path.is_file():
                        errors.append(f"missing adapter: {path}")
    progress = list(
        (args.run_root / "tensor_cache").rglob("build_progress_v2.json")
    ) if (args.run_root / "tensor_cache").is_dir() else []
    if progress:
        errors.append(
            "Tensor exact cache still has active/incomplete progress files; "
            "finish/resume the independent Tensor run first. Example: "
            f"{progress[0]}"
        )
    return errors


def build_store(tensor, args: argparse.Namespace):
    from transformers import AutoConfig

    sampling_config = AutoConfig.from_pretrained(
        str(args.sampling_model_dir),
        local_files_only=True,
        trust_remote_code=True,
    )
    scoring_config = AutoConfig.from_pretrained(
        str(args.scoring_model_dir),
        local_files_only=True,
        trust_remote_code=True,
    )
    common_vocab_size = min(
        int(sampling_config.vocab_size), int(scoring_config.vocab_size)
    )
    context_id, context = tensor.exact_model_context(
        args.sampling_model_dir,
        args.scoring_model_dir,
        sampling_dtype=args.sampling_dtype,
        scoring_dtype=args.scoring_dtype,
        max_length=args.max_length,
        common_vocab_size=common_vocab_size,
    )
    return tensor.ExactFeatureStore(
        args.run_root / "tensor_cache",
        model_context_id=context_id,
        model_context=context,
    )


def load_cell_context(
    tensor,
    store,
    args: argparse.Namespace,
    source: str,
    test_domain: str,
):
    training_domains = sorted(
        name for name in DATASETS if name != test_domain
    )

    entries = {
        test_domain: adapter_prefix(
            adapter_path(
                args.run_root,
                test_domain,
                source,
                development=False,
            )
        )
    }
    entries.update(
        {
            domain: adapter_prefix(
                adapter_path(
                    args.run_root,
                    domain,
                    source,
                    development=True,
                )
            )
            for domain in training_domains
        }
    )

    cache_roles = {test_domain: "test"}
    cache_roles.update(
        {domain: "train" for domain in training_domains}
    )

    raw_data = {
        domain.rsplit("/", 1)[-1]: tensor.load_raw_data(prefix)
        for domain, prefix in entries.items()
    }

    sampling_device, scoring_device = tensor.resolve_proxy_devices(
        args.sampling_device, args.scoring_device
    )
    original_loader = tensor.load_local_model

    def refuse_model_load(*_args, **_kwargs):
        raise RuntimeError(
            "required Tensor token cache is missing; this ablation refuses "
            "to build/overwrite it. Finish the independent Tensor run first."
        )

    tensor.load_local_model = refuse_model_load
    try:
        caches = tensor.ensure_token_caches(
            entries,
            source_model=source,
            cache_root=args.run_root / "tensor_cache" / "full_vocab_observed",
            sampling_model_dir=args.sampling_model_dir,
            scoring_model_dir=args.scoring_model_dir,
            sampling_dtype=args.sampling_dtype,
            scoring_dtype=args.scoring_dtype,
            num_samples=2,
            max_length=args.max_length,
            seed=args.seed,
            chunk_tokens=args.chunk_tokens,
            overwrite=False,
            sampling_device=sampling_device,
            scoring_device=scoring_device,
            document_batch_size=args.document_batch_size,
            cache_roles=cache_roles,
        )
    finally:
        tensor.load_local_model = original_loader

    caches = {key.rsplit("/", 1)[-1]: value for key, value in caches.items()}
    requests, lookup = tensor.build_global_exact_request_plan(
        caches, args.structures_list
    )
    return training_domains, raw_data, caches, requests, lookup


def project_entropy_independent(
    u: np.ndarray, v: np.ndarray, j_bases: int, k_bases: int
) -> tuple[np.ndarray, np.ndarray]:
    """Exactly project B(z) x C(h) moments to B(z) moments."""
    u_array = np.asarray(u)
    v_array = np.asarray(v)
    n_documents = u_array.shape[0]
    reduced_u = u_array.reshape(n_documents, j_bases, k_bases).sum(axis=2)
    reduced_v = v_array.reshape(
        n_documents, j_bases, k_bases, j_bases, k_bases
    ).sum(axis=(2, 4))
    return reduced_u, 0.5 * (reduced_v + reduced_v.transpose(0, 2, 1))


def effective_pair(
    pair,
    config: AblationConfig,
    entropy_conditioning: bool,
):
    if entropy_conditioning:
        return (
            (np.asarray(pair[0][0]), np.asarray(pair[0][1])),
            (np.asarray(pair[1][0]), np.asarray(pair[1][1])),
        )
    return (
        project_entropy_independent(
            pair[0][0], pair[0][1],
            config.source_j_bases, config.source_k_bases,
        ),
        project_entropy_independent(
            pair[1][0], pair[1][1],
            config.source_j_bases, config.source_k_bases,
        ),
    )


def effective_spec(config: AblationConfig):
    return SimpleNamespace(
        j_bases=config.j_bases,
        k_bases=config.k_bases,
        dimension=config.dimension,
    )


def tensor_config(tensor, config: AblationConfig):
    return tensor.Config(
        config.j_bases,
        config.k_bases,
        config.smooth_lambda,
        config.ridge_lambda,
        config.covariance_shrinkage,
    )


def candidate_configs(
    variant: Variant,
    structures: Sequence[tuple[int, int]],
    smooth_lambdas: Sequence[float],
    ridge_lambdas: Sequence[float],
    shrinkages: Sequence[float],
) -> list[AblationConfig]:
    source_structures = list(structures)
    if not variant.entropy_conditioning:
        first_for_j: dict[int, tuple[int, int]] = {}
        for structure in source_structures:
            first_for_j.setdefault(structure[0], structure)
        source_structures = list(first_for_j.values())
    smooth_grid = smooth_lambdas if variant.smoothness else [0.0]
    shrink_grid = shrinkages if variant.covariance_shrinkage else [0.0]
    result = []
    for (source_j, source_k), smooth, ridge, shrink in itertools.product(
        source_structures, smooth_grid, ridge_lambdas, shrink_grid
    ):
        effective_k = source_k if variant.entropy_conditioning else 1
        result.append(
            AblationConfig(
                source_j_bases=source_j,
                source_k_bases=source_k,
                j_bases=source_j,
                k_bases=effective_k,
                smooth_lambda=float(smooth),
                ridge_lambda=float(ridge),
                covariance_shrinkage=float(shrink),
            )
        )
    return result


def concatenate_features(parts: Iterable[tuple[np.ndarray, np.ndarray]]):
    values = list(parts)
    return (
        np.concatenate([np.asarray(value[0]) for value in values], axis=0),
        np.concatenate([np.asarray(value[1]) for value in values], axis=0),
    )


def config_key(config: AblationConfig) -> tuple[Any, ...]:
    return tuple(asdict(config).values())


def select_variant(
    tensor,
    store,
    variant: Variant,
    training_domains: Sequence[str],
    raw_data: dict[str, dict[str, list[str]]],
    requests,
    lookup,
    configs: Sequence[AblationConfig],
) -> tuple[AblationConfig, list[dict[str, Any]]]:
    records: dict[tuple[Any, ...], list[dict[str, Any]]] = {
        config_key(config): [] for config in configs
    }
    grouped: dict[tuple[int, int], list[AblationConfig]] = {}
    for config in configs:
        grouped.setdefault(config.source_structure, []).append(config)

    for validation_domain in training_domains:
        fit_domains = sorted(
            name for name in training_domains if name != validation_domain
        )
        fit_subset = tuple(fit_domains)
        for (source_j, source_k), group in grouped.items():
            request_id = lookup[(fit_subset, source_j, source_k)]
            request = requests[request_id]
            loaded = {
                domain: tensor.load_exact_domain_pair(
                    store, raw_data[domain], request_id, request
                )
                for domain in training_domains
            }
            template = group[0]
            transformed = {
                domain: effective_pair(
                    pair, template, variant.entropy_conditioning
                )
                for domain, pair in loaded.items()
            }
            human_fit = concatenate_features(
                transformed[domain][0] for domain in fit_domains
            )
            machine_fit = concatenate_features(
                transformed[domain][1] for domain in fit_domains
            )
            validation = transformed[validation_domain]
            for config in group:
                theta = tensor.fit_rayleigh(
                    human_fit[0], human_fit[1],
                    machine_fit[0], machine_fit[1],
                    effective_spec(config), tensor_config(tensor, config),
                )
                human_scores = tensor.score_documents(
                    validation[0][0], validation[0][1], theta
                )
                machine_scores = tensor.score_documents(
                    validation[1][0], validation[1][1], theta
                )
                records[config_key(config)].append(
                    {
                        "validation_domain": validation_domain,
                        "auroc": tensor.auc(human_scores, machine_scores),
                    }
                )
            del loaded, transformed, human_fit, machine_fit

    summaries: list[dict[str, Any]] = []
    by_key = {config_key(config): config for config in configs}
    for key, folds in records.items():
        values = [fold["auroc"] for fold in folds]
        summaries.append(
            {
                **asdict(by_key[key]),
                "dimension": by_key[key].dimension,
                "mean_validation_auroc": float(np.mean(values)),
                "std_validation_auroc": float(np.std(values)),
                "folds": folds,
            }
        )
    best_record = max(
        summaries,
        key=lambda item: (
            item["mean_validation_auroc"],
            -item["dimension"],
            -item["smooth_lambda"],
            -item["ridge_lambda"],
            -item["covariance_shrinkage"],
        ),
    )
    selected = AblationConfig(
        **{field: best_record[field] for field in asdict(configs[0])}
    )
    summaries.sort(
        key=lambda item: (
            item["mean_validation_auroc"],
            -item["dimension"],
            -item["smooth_lambda"],
            -item["ridge_lambda"],
            -item["covariance_shrinkage"],
        ),
        reverse=True,
    )
    return selected, summaries


def fit_and_score_variant(
    tensor,
    store,
    variant: Variant,
    selected: AblationConfig,
    training_domains: Sequence[str],
    test_domain: str,
    raw_data: dict[str, dict[str, list[str]]],
    requests,
    lookup,
):
    training_subset = tuple(sorted(training_domains))
    request_id = lookup[
        (
            training_subset,
            selected.source_j_bases,
            selected.source_k_bases,
        )
    ]
    request = requests[request_id]
    source_spec, sorted_entropy = request
    loaded_train = {
        domain: tensor.load_exact_domain_pair(
            store, raw_data[domain], request_id, request
        )
        for domain in training_domains
    }
    loaded_test = tensor.load_exact_domain_pair(
        store, raw_data[test_domain], request_id, request
    )
    transformed_train = {
        domain: effective_pair(pair, selected, variant.entropy_conditioning)
        for domain, pair in loaded_train.items()
    }
    transformed_test = effective_pair(
        loaded_test, selected, variant.entropy_conditioning
    )
    human_train = concatenate_features(
        transformed_train[domain][0] for domain in training_domains
    )
    machine_train = concatenate_features(
        transformed_train[domain][1] for domain in training_domains
    )
    theta = tensor.fit_rayleigh(
        human_train[0], human_train[1],
        machine_train[0], machine_train[1],
        effective_spec(selected), tensor_config(tensor, selected),
    )
    human_scores = tensor.score_documents(
        transformed_test[0][0], transformed_test[0][1], theta
    )
    machine_scores = tensor.score_documents(
        transformed_test[1][0], transformed_test[1][1], theta
    )
    return (
        np.asarray(theta, dtype=np.float64),
        np.asarray(human_scores, dtype=np.float64),
        np.asarray(machine_scores, dtype=np.float64),
        source_spec,
        np.asarray(sorted_entropy, dtype=np.float64),
        request_id,
    )


def labels_scores(human: np.ndarray, machine: np.ndarray):
    return (
        np.concatenate([np.zeros(len(human)), np.ones(len(machine))]),
        np.concatenate([human, machine]),
    )


def normalized_partial_auc(
    human: np.ndarray, machine: np.ndarray, max_fpr: float
) -> float:
    labels, scores = labels_scores(human, machine)
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
    labels, scores = labels_scores(human, machine)
    fpr, tpr, _ = roc_curve(labels, scores)
    eligible = tpr[fpr <= max_fpr]
    return float(eligible.max()) if len(eligible) else 0.0


def conformal_pvalues(calibration: np.ndarray, scores: np.ndarray):
    ordered = np.sort(calibration)
    counts_ge = len(ordered) - np.searchsorted(ordered, scores, side="left")
    return (1.0 + counts_ge) / (len(ordered) + 1.0)


def metrics_from_scores(
    human: np.ndarray,
    machine: np.ndarray,
    calibration: np.ndarray | None = None,
) -> dict[str, float | None]:
    labels, scores = labels_scores(human, machine)
    precision, recall, _ = precision_recall_curve(labels, scores)
    result: dict[str, float | None] = {
        "auroc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(sklearn_auc(recall, precision)),
        "pauc_001": normalized_partial_auc(human, machine, 0.01),
        "pauc_005": normalized_partial_auc(human, machine, 0.05),
        "tpr_at_fpr_001": tpr_at_fpr(human, machine, 0.01),
        "tpr_at_fpr_005": tpr_at_fpr(human, machine, 0.05),
        "conformal_fpr_001": None,
        "conformal_power_001": None,
        "conformal_fpr_005": None,
        "conformal_power_005": None,
    }
    if calibration is not None:
        human_p = conformal_pvalues(calibration, human)
        machine_p = conformal_pvalues(calibration, machine)
        result.update(
            {
                "conformal_fpr_001": float(np.mean(human_p <= 0.01)),
                "conformal_power_001": float(np.mean(machine_p <= 0.01)),
                "conformal_fpr_005": float(np.mean(human_p <= 0.05)),
                "conformal_power_005": float(np.mean(machine_p <= 0.05)),
            }
        )
    return result


def run_signature(
    args: argparse.Namespace,
    variant: Variant,
    base_result: Path,
    test_adapter: Path,
) -> str:
    value = {
        "version": VERSION,
        "variant": asdict(variant),
        "structures": args.structures_list,
        "smooth_lambdas": args.smooth_lambdas_list,
        "ridge_lambdas": args.ridge_lambdas_list,
        "shrinkages": args.shrinkages_list,
        "seed": args.seed,
        "max_length": args.max_length,
        "score": "self_normalized_S_theta",
        "base_result_sha256": sha256_file(base_result),
        "test_adapter_sha256": sha256_file(test_adapter),
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def save_frozen_result(
    output_dir: Path,
    result: dict[str, Any],
    theta: np.ndarray,
    sorted_entropy: np.ndarray,
    human_scores: np.ndarray,
    machine_scores: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_savez(
        output_dir / "frozen_model.npz",
        theta=np.asarray(theta, dtype=np.float64),
        sorted_entropy=np.asarray(sorted_entropy, dtype=np.float64),
        selected_json=np.asarray(
            [json.dumps(result["selected_hyperparameters"], sort_keys=True)]
        ),
        source_basis_spec_json=np.asarray(
            [json.dumps(result["source_basis_spec"], sort_keys=True)]
        ),
    )
    atomic_savez(
        output_dir / "scores.npz",
        human=np.asarray(human_scores, dtype=np.float64),
        machine=np.asarray(machine_scores, dtype=np.float64),
    )
    atomic_json(output_dir / "results.json", result)


def full_from_main(
    tensor,
    base: dict[str, Any],
    requests,
    lookup,
    training_domains: Sequence[str],
):
    selected_value = base["selected_hyperparameters"]
    selected = AblationConfig(
        source_j_bases=int(selected_value["j_bases"]),
        source_k_bases=int(selected_value["k_bases"]),
        j_bases=int(selected_value["j_bases"]),
        k_bases=int(selected_value["k_bases"]),
        smooth_lambda=float(selected_value["smooth_lambda"]),
        ridge_lambda=float(selected_value["ridge_lambda"]),
        covariance_shrinkage=float(selected_value["covariance_shrinkage"]),
    )
    request_id = lookup[
        (
            tuple(sorted(training_domains)),
            selected.source_j_bases,
            selected.source_k_bases,
        )
    ]
    source_spec, sorted_entropy = requests[request_id]
    normalized_spec = json.loads(json.dumps(asdict(source_spec)))
    if normalized_spec != base.get("basis_spec"):
        raise ValueError(
            "main Full basis_spec does not match the current training token cache"
        )
    theta = np.asarray(base["model"]["theta"], dtype=np.float64)
    human = np.asarray(base["predictions"]["real"], dtype=np.float64)
    machine = np.asarray(base["predictions"]["samples"], dtype=np.float64)
    if theta.shape != (selected.dimension,):
        raise ValueError("main Full theta dimension disagrees with selected basis")
    return selected, theta, human, machine, source_spec, sorted_entropy, request_id


def run_test_scoring(
    tensor,
    store,
    args: argparse.Namespace,
) -> None:
    for source in args.source_models_list:
        for test_domain in args.datasets_list:
            print(f"CELL {source}/{test_domain}: loading cache context", flush=True)
            (
                training_domains,
                raw_data,
                _caches,
                requests,
                lookup,
            ) = load_cell_context(
                tensor, store, args, source, test_domain
            )
            base_path = result_path(args.run_root, test_domain, source)
            base = load_json(base_path)
            test_adapter = adapter_path(
                args.run_root, test_domain, source, development=False
            )
            for variant in args.variants_list:
                output_dir = (
                    args.output_root / source / test_domain / variant.key
                )
                output_path = output_dir / "results.json"
                signature = run_signature(
                    args, variant, base_path, test_adapter
                )
                if output_path.is_file() and not args.overwrite:
                    existing = load_json(output_path)
                    if existing.get("run_signature") != signature:
                        raise ValueError(
                            f"{output_path}: protocol changed; use --overwrite"
                        )
                    if (
                        (output_dir / "frozen_model.npz").is_file()
                        and (output_dir / "scores.npz").is_file()
                    ):
                        print(
                            f"SKIP frozen {source}/{test_domain}/{variant.key}",
                            flush=True,
                        )
                        continue

                if variant.key == "full":
                    (
                        selected,
                        theta,
                        human_scores,
                        machine_scores,
                        source_spec,
                        sorted_entropy,
                        request_id,
                    ) = full_from_main(
                        tensor, base, requests, lookup, training_domains
                    )
                    validation = base.get("domain_validation", [])
                    candidate_count = len(validation)
                    provenance = "verbatim existing main Tensor result"
                else:
                    configs = candidate_configs(
                        variant,
                        args.structures_list,
                        args.smooth_lambdas_list,
                        args.ridge_lambdas_list,
                        args.shrinkages_list,
                    )
                    print(
                        f"RUN {source}/{test_domain}/{variant.key}: "
                        f"{len(configs)} candidates x 4 held-out dev domains",
                        flush=True,
                    )
                    selected, validation = select_variant(
                        tensor, store, variant, training_domains, raw_data,
                        requests, lookup, configs,
                    )
                    (
                        theta,
                        human_scores,
                        machine_scores,
                        source_spec,
                        sorted_entropy,
                        request_id,
                    ) = fit_and_score_variant(
                        tensor, store, variant, selected, training_domains,
                        test_domain, raw_data, requests, lookup,
                    )
                    candidate_count = len(configs)
                    provenance = "cache-only factorial refit"

                metrics = metrics_from_scores(human_scores, machine_scores)
                result = {
                    "schema_version": 1,
                    "builder_version": VERSION,
                    "run_signature": signature,
                    "status": "frozen_and_independent_test_scored",
                    "source_model": source,
                    "test_domain": test_domain,
                    "training_domains": training_domains,
                    "variant": variant.key,
                    "label": variant.label,
                    "modules": {
                        "entropy_conditioning": variant.entropy_conditioning,
                        "smoothness": variant.smoothness,
                        "covariance_shrinkage": variant.covariance_shrinkage,
                    },
                    "protocol": {
                        "conditional_moments": "exact full vocabulary",
                        "deployment_score": (
                            "theta^T U_X / sqrt(max(theta^T V_X theta, epsilon))"
                        ),
                        "test_domain_used_for_fitting_or_selection": False,
                        "calibration_used_for_fitting_or_selection": False,
                        "selection": "leave-one-development-domain-out mean AUROC",
                        "entropy_removal": (
                            "exact partition-of-unity projection"
                            if not variant.entropy_conditioning else None
                        ),
                        "full_variant_provenance": provenance,
                    },
                    "selected_hyperparameters": asdict(selected),
                    "source_basis_spec": asdict(source_spec),
                    "exact_request_id": request_id,
                    "hyperparameter_selection": {
                        "candidate_count": candidate_count,
                        "candidates": validation,
                    },
                    "n_human": len(human_scores),
                    "n_machine": len(machine_scores),
                    "n_calibration": 0,
                    "metrics": metrics,
                }
                save_frozen_result(
                    output_dir, result, theta, sorted_entropy,
                    human_scores, machine_scores,
                )
                print(
                    f"DONE {source}/{test_domain}/{variant.key}: "
                    f"AUROC={metrics['auroc']:.6f}",
                    flush=True,
                )
            del raw_data, requests, lookup
            gc.collect()


def collect_calibration_groups(args: argparse.Namespace):
    groups: dict[str, dict[str, Any]] = {}
    for source in args.source_models_list:
        for dataset in args.datasets_list:
            path = adapter_path(
                args.run_root, dataset, source, development=False
            )
            texts = load_adapter(path)["calibration"]
            digest = sequence_digest(texts)
            group = groups.setdefault(
                digest,
                {"texts": texts, "members": [], "requests": {}},
            )
            if group["texts"] != texts:
                raise RuntimeError("SHA-256 collision in calibration texts")
            group["members"].append((source, dataset))
    return groups


def load_frozen_context(tensor, args, source: str, dataset: str, variant: Variant):
    directory = args.output_root / source / dataset / variant.key
    result = load_json(directory / "results.json")
    with np.load(directory / "frozen_model.npz", allow_pickle=False) as arrays:
        theta = np.asarray(arrays["theta"], dtype=np.float64)
        sorted_entropy = np.asarray(
            arrays["sorted_entropy"], dtype=np.float64
        )
    source_spec = tensor.BasisSpec(**result["source_basis_spec"])
    request_id = tensor.exact_request_digest(source_spec, sorted_entropy)
    if request_id != result["exact_request_id"]:
        raise ValueError(
            f"{source}/{dataset}/{variant.key}: frozen request digest mismatch"
        )
    selected = AblationConfig(**result["selected_hyperparameters"])
    return directory, result, selected, theta, source_spec, sorted_entropy, request_id


def score_calibration(tensor, store, args: argparse.Namespace) -> None:
    groups = collect_calibration_groups(args)
    member_to_group: dict[tuple[str, str], dict[str, Any]] = {}
    for group in groups.values():
        for member in group["members"]:
            member_to_group[member] = group
            source, dataset = member
            for variant in args.variants_list:
                context = load_frozen_context(
                    tensor, args, source, dataset, variant
                )
                source_spec, sorted_entropy, request_id = (
                    context[4], context[5], context[6]
                )
                group["requests"].setdefault(
                    request_id, (source_spec, sorted_entropy)
                )

    pending_groups = []
    for digest, group in groups.items():
        missing = {
            request_id: request
            for request_id, request in group["requests"].items()
            if not store.is_valid(
                group["texts"], request_id, request[0], request[1]
            )
        }
        if missing:
            pending_groups.append((digest, group, missing))

    if pending_groups:
        sampling_device, scoring_device = tensor.resolve_proxy_devices(
            args.sampling_device, args.scoring_device
        )
        print(
            f"CALIBRATION: {len(pending_groups)} text groups need exact "
            "features; loading GPT-J/GPT-Neo once.",
            flush=True,
        )
        sampling_model, sampling_tokenizer = tensor.load_local_model(
            args.sampling_model_dir,
            args.sampling_dtype,
            sampling_device,
        )
        scoring_model, scoring_tokenizer = tensor.load_local_model(
            args.scoring_model_dir,
            args.scoring_dtype,
            scoring_device,
        )
        try:
            for digest, group, missing in pending_groups:
                tensor.write_exact_feature_set(
                    store,
                    group["texts"],
                    missing,
                    sampling_model=sampling_model,
                    sampling_tokenizer=sampling_tokenizer,
                    scoring_model=scoring_model,
                    scoring_tokenizer=scoring_tokenizer,
                    max_length=args.max_length,
                    chunk_tokens=args.chunk_tokens,
                    document_batch_size=args.document_batch_size,
                    request_batch_size=args.request_batch_size,
                    label=f"calibration/{digest[:12]}",
                    overwrite=False,
                )
        finally:
            del sampling_model, sampling_tokenizer, scoring_model, scoring_tokenizer
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
    else:
        print("CALIBRATION: exact feature cache hit; no model load.", flush=True)

    for source in args.source_models_list:
        for dataset in args.datasets_list:
            group = member_to_group[(source, dataset)]
            for variant in args.variants_list:
                (
                    directory,
                    result,
                    selected,
                    theta,
                    source_spec,
                    sorted_entropy,
                    request_id,
                ) = load_frozen_context(tensor, args, source, dataset, variant)
                scores_path = directory / "scores.npz"
                with np.load(scores_path, allow_pickle=False) as arrays:
                    human = np.asarray(arrays["human"], dtype=np.float64)
                    machine = np.asarray(arrays["machine"], dtype=np.float64)
                    if "calibration" in arrays and result.get("status") == "complete":
                        print(
                            f"SKIP calibration {source}/{dataset}/{variant.key}",
                            flush=True,
                        )
                        continue
                u, v = store.load(
                    group["texts"], request_id, source_spec, sorted_entropy
                )
                if variant.entropy_conditioning:
                    effective_u, effective_v = np.asarray(u), np.asarray(v)
                else:
                    effective_u, effective_v = project_entropy_independent(
                        u, v,
                        selected.source_j_bases,
                        selected.source_k_bases,
                    )
                calibration = tensor.score_documents(
                    effective_u, effective_v, theta
                )
                calibration = np.asarray(calibration, dtype=np.float64)
                result["metrics"] = metrics_from_scores(
                    human, machine, calibration
                )
                result["n_calibration"] = len(calibration)
                result["status"] = "complete"
                result["calibration_text_sha256"] = sequence_digest(
                    group["texts"]
                )
                atomic_savez(
                    scores_path,
                    human=human,
                    machine=machine,
                    calibration=calibration,
                )
                atomic_json(directory / "results.json", result)
                print(
                    f"DONE calibration {source}/{dataset}/{variant.key}",
                    flush=True,
                )


def summary_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for source in args.source_models_list:
        for dataset in args.datasets_list:
            for variant in args.variants_list:
                path = (
                    args.output_root / source / dataset /
                    variant.key / "results.json"
                )
                result = load_json(path)
                if not args.skip_calibration and result.get("status") != "complete":
                    raise ValueError(f"incomplete result: {path}")
                selected = result["selected_hyperparameters"]
                row = {
                    "source_model": source,
                    "dataset": dataset,
                    "variant": variant.key,
                    "H": int(variant.entropy_conditioning),
                    "S": int(variant.smoothness),
                    "C": int(variant.covariance_shrinkage),
                    "source_structure": (
                        f"{selected['source_j_bases']}x"
                        f"{selected['source_k_bases']}"
                    ),
                    "effective_dimension": (
                        int(selected["j_bases"]) * int(selected["k_bases"])
                    ),
                    "smooth_lambda": selected["smooth_lambda"],
                    "ridge_lambda": selected["ridge_lambda"],
                    "covariance_shrinkage": selected["covariance_shrinkage"],
                    "n_human": result["n_human"],
                    "n_machine": result["n_machine"],
                    "n_calibration": result["n_calibration"],
                }
                row.update(result["metrics"])
                rows.append(row)
    return rows


def bootstrap_interval(
    values: np.ndarray, *, seed: int, samples: int
) -> tuple[float, float]:
    if len(values) < 2 or samples < 1:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 500):
        stop = min(samples, start + 500)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def write_summaries(args: argparse.Namespace) -> None:
    rows = summary_rows(args)
    atomic_csv(args.output_root / "summary.csv", rows)
    metric_names = [
        "auroc", "pr_auc", "pauc_001", "pauc_005",
        "tpr_at_fpr_001", "tpr_at_fpr_005",
    ]
    if not args.skip_calibration:
        metric_names.extend(
            [
                "conformal_fpr_001", "conformal_power_001",
                "conformal_fpr_005", "conformal_power_005",
            ]
        )

    macro_rows = []
    for variant in args.variants_list:
        selected_rows = [row for row in rows if row["variant"] == variant.key]
        output = {
            "variant": variant.key,
            "H": int(variant.entropy_conditioning),
            "S": int(variant.smoothness),
            "C": int(variant.covariance_shrinkage),
            "n_cells": len(selected_rows),
        }
        for metric in metric_names:
            values = np.asarray([row[metric] for row in selected_rows], dtype=float)
            output[f"mean_{metric}"] = float(values.mean())
            output[f"std_{metric}"] = float(values.std())
        macro_rows.append(output)
    atomic_csv(args.output_root / "variant_macro.csv", macro_rows)

    if set(item.key for item in args.variants_list) != set(VARIANT_BY_KEY):
        print(
            "SUMMARY: subset run; factorial_effects.csv requires all 8 variants.",
            flush=True,
        )
        return

    by_cell = {
        (row["source_model"], row["dataset"], row["variant"]): row
        for row in rows
    }
    cells = list(itertools.product(args.source_models_list, args.datasets_list))
    contrasts = {
        "H": [("h", "b0"), ("hs", "s"), ("hc", "c"), ("full", "sc")],
        "S": [("s", "b0"), ("hs", "h"), ("sc", "c"), ("full", "hc")],
        "C": [("c", "b0"), ("hc", "h"), ("sc", "s"), ("full", "hs")],
    }
    effect_rows = []
    for module, pairs in contrasts.items():
        for metric_index, metric in enumerate(metric_names):
            cell_effects = []
            for source, dataset in cells:
                differences = [
                    float(by_cell[(source, dataset, on)][metric])
                    - float(by_cell[(source, dataset, off)][metric])
                    for on, off in pairs
                ]
                cell_effects.append(float(np.mean(differences)))
            values = np.asarray(cell_effects)
            low, high = bootstrap_interval(
                values,
                seed=args.bootstrap_seed + 100 * metric_index + ord(module),
                samples=args.bootstrap_samples,
            )
            effect_rows.append(
                {
                    "effect_kind": "balanced_factorial_main_effect",
                    "term": module,
                    "metric": metric,
                    "n_cells": len(values),
                    "mean_delta": float(values.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "positive_cells": int(np.sum(values > 0)),
                    "negative_cells": int(np.sum(values < 0)),
                }
            )

    leave_one_out = {
        "H": ("full", "sc"),
        "S": ("full", "hc"),
        "C": ("full", "hs"),
    }
    for module, (full_key, removed_key) in leave_one_out.items():
        for metric_index, metric in enumerate(metric_names):
            values = np.asarray(
                [
                    float(by_cell[(source, dataset, full_key)][metric])
                    - float(by_cell[(source, dataset, removed_key)][metric])
                    for source, dataset in cells
                ],
                dtype=np.float64,
            )
            low, high = bootstrap_interval(
                values,
                seed=args.bootstrap_seed + 1000 + 100 * metric_index + ord(module),
                samples=args.bootstrap_samples,
            )
            effect_rows.append(
                {
                    "effect_kind": "leave_one_component_out_from_full",
                    "term": module,
                    "metric": metric,
                    "n_cells": len(values),
                    "mean_delta": float(values.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "positive_cells": int(np.sum(values > 0)),
                    "negative_cells": int(np.sum(values < 0)),
                }
            )
    atomic_csv(args.output_root / "factorial_effects.csv", effect_rows)
    print(f"SUMMARY: {len(rows)} cell/variant rows", flush=True)
    print(f"  {args.output_root / 'summary.csv'}", flush=True)
    print(f"  {args.output_root / 'variant_macro.csv'}", flush=True)
    print(f"  {args.output_root / 'factorial_effects.csv'}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    script_project = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Black-box exact-cache H/S/C factorial ablation"
    )
    parser.add_argument("--project-root", type=Path, default=script_project)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--sampling-model-dir", type=Path)
    parser.add_argument("--scoring-model-dir", type=Path)
    parser.add_argument(
        "--source-models", default=",".join(SOURCE_MODELS)
    )
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument(
        "--variants", default=",".join(item.key for item in VARIANTS)
    )
    parser.add_argument("--structures", default="4x4,6x6,8x6")
    parser.add_argument("--smooth-lambdas", default="0,1e-3,1e-2,1e-1,1")
    parser.add_argument("--ridge-lambdas", default="1e-5,1e-4,1e-3,1e-2")
    parser.add_argument("--covariance-shrinkages", default="0,0.05,0.2")
    parser.add_argument("--sampling-dtype", default="float16")
    parser.add_argument("--scoring-dtype", default="float16")
    parser.add_argument("--sampling-device", default="auto")
    parser.add_argument("--scoring-device", default="auto")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--chunk-tokens", type=int, default=16)
    parser.add_argument("--document-batch-size", type=int, default=2)
    parser.add_argument("--request-batch-size", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--bootstrap-seed", type=int, default=20260815)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def prepare_args(args: argparse.Namespace) -> None:
    args.project_root = args.project_root.resolve()
    if args.run_root is None:
        args.run_root = (
            args.project_root
            / "cache/blackbox/independent"
        )
    args.run_root = args.run_root.resolve()
    if args.output_root is None:
        args.output_root = args.run_root / "tensor_factorial_ablation"
    args.output_root = args.output_root.resolve()
    if args.sampling_model_dir is None:
        args.sampling_model_dir = args.project_root / "models/gpt-j-6b"
    if args.scoring_model_dir is None:
        args.scoring_model_dir = args.project_root / "models/gpt-neo-2.7b"
    args.sampling_model_dir = args.sampling_model_dir.resolve()
    args.scoring_model_dir = args.scoring_model_dir.resolve()
    args.source_models_list = parse_csv(args.source_models)
    args.datasets_list = parse_csv(args.datasets)
    args.structures_list = parse_structures(args.structures)
    args.smooth_lambdas_list = parse_float_csv(args.smooth_lambdas)
    args.ridge_lambdas_list = parse_float_csv(args.ridge_lambdas)
    args.shrinkages_list = parse_float_csv(args.covariance_shrinkages)
    unknown_sources = set(args.source_models_list) - set(SOURCE_MODELS)
    unknown_datasets = set(args.datasets_list) - set(DATASETS)
    if unknown_sources:
        raise ValueError(f"unknown source models: {sorted(unknown_sources)}")
    if unknown_datasets:
        raise ValueError(f"unknown datasets: {sorted(unknown_datasets)}")
    variant_keys = parse_csv(args.variants)
    unknown_variants = set(variant_keys) - set(VARIANT_BY_KEY)
    if unknown_variants:
        raise ValueError(f"unknown variants: {sorted(unknown_variants)}")
    args.variants_list = [VARIANT_BY_KEY[key] for key in variant_keys]
    if not args.datasets_list:
        raise ValueError("at least one independent test domain is required")
    for name in ("sampling_model_dir", "scoring_model_dir"):
        if not getattr(args, name).is_dir():
            raise FileNotFoundError(getattr(args, name))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        prepare_args(args)
    except Exception as exc:
        parser.error(str(exc))

    errors = validate_inputs(args)
    if errors:
        print("INPUT CHECK FAILED:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        print(
            "Do not start the ablation yet. Finish/resume the main independent "
            "Tensor run, then rerun --check-only.",
            file=sys.stderr,
        )
        return 2
    expected_cells = len(args.source_models_list) * len(args.datasets_list)
    print(
        f"INPUT CHECK PASSED: {expected_cells} main Tensor cells; "
        f"{len(args.variants_list)} factorial variants.",
        flush=True,
    )
    if args.check_only:
        return 0

    tensor_path = args.project_root / "experiments/detect_tensor_rayleigh_blackbox.py"
    tensor = import_file(tensor_path, "water_tensor_blackbox_for_factorial")
    store = build_store(tensor, args)
    run_test_scoring(tensor, store, args)
    if not args.skip_calibration:
        score_calibration(tensor, store, args)
    write_summaries(args)
    print("BLACK-BOX FACTORIAL ABLATION COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
