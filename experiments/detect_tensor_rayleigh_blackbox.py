#!/usr/bin/env python3
"""Water Tensor-Rayleigh under AdaDetectGPT's simple black-box protocol.

The target generator contributes text only. GPT-J-6B supplies the reference
distribution q_r. GPT-Neo-2.7B supplies observed/candidate log probabilities
under q_s. Full-vocabulary moments are the default; Monte Carlo candidates are
retained as an ablation. For a held-out test domain, hyperparameters are
selected by leave-one-training-domain-out validation on the remaining domains.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import (
    auc as sklearn_auc,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import SplineTransformer


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_float_csv(value: str) -> list[float]:
    return [float(item) for item in parse_csv(value)]


def parse_structures(value: str) -> list[tuple[int, int]]:
    result = []
    for item in parse_csv(value):
        pieces = item.lower().split("x")
        if len(pieces) != 2:
            raise ValueError(f"Invalid structure {item!r}; expected JxK")
        j, k = (int(piece) for piece in pieces)
        if j < 3 or k < 3:
            raise ValueError("Each spline direction requires at least 3 bases")
        result.append((j, k))
    return result


def raw_data_path(prefix: Path) -> Path:
    return Path(f"{prefix}.raw_data.json")


def load_raw_data(prefix: Path) -> dict[str, list[str]]:
    path = raw_data_path(prefix)
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    for key in ("original", "sampled"):
        items = value.get(key)
        if not isinstance(items, list) or not items:
            raise ValueError(f"{path}: {key} must be a non-empty list")
        if not all(isinstance(item, str) and item.strip() for item in items):
            raise ValueError(f"{path}: {key} contains invalid text")
    if len(value["original"]) != len(value["sampled"]):
        raise ValueError(f"{path}: human/machine counts differ")
    return {"original": value["original"], "sampled": value["sampled"]}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@dataclass
class ReferenceCache:
    offsets: np.ndarray
    observed_ids: np.ndarray
    entropy: np.ndarray
    candidate_ids: np.ndarray
    num_samples: int

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp.npz")
        np.savez(
            temporary,
            offsets=self.offsets.astype(np.int64),
            observed_ids=self.observed_ids.astype(np.int32),
            entropy=self.entropy.astype(np.float32),
            candidate_ids=self.candidate_ids.astype(np.int32),
            num_samples=np.asarray([self.num_samples], dtype=np.int64),
        )
        os.replace(temporary, path)

    @classmethod
    def load(cls, path: Path) -> "ReferenceCache":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                offsets=np.asarray(data["offsets"], dtype=np.int64),
                observed_ids=np.asarray(data["observed_ids"], dtype=np.int64),
                entropy=np.asarray(data["entropy"], dtype=np.float64),
                candidate_ids=np.asarray(data["candidate_ids"], dtype=np.int64),
                num_samples=int(np.asarray(data["num_samples"])[0]),
            )


@dataclass
class TokenCache:
    offsets: np.ndarray
    z_observed: np.ndarray
    entropy: np.ndarray
    z_sampled: np.ndarray
    num_samples: int

    @property
    def num_documents(self) -> int:
        return len(self.offsets) - 1

    def document(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        start = int(self.offsets[index])
        stop = int(self.offsets[index + 1])
        return (
            self.z_observed[start:stop],
            self.entropy[start:stop],
            self.z_sampled[start:stop],
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp.npz")
        np.savez(
            temporary,
            offsets=self.offsets.astype(np.int64),
            z_observed=self.z_observed.astype(np.float32),
            entropy=self.entropy.astype(np.float32),
            z_sampled=self.z_sampled.astype(np.float16),
            num_samples=np.asarray([self.num_samples], dtype=np.int64),
        )
        os.replace(temporary, path)

    @classmethod
    def load(cls, path: Path) -> "TokenCache":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                offsets=np.asarray(data["offsets"], dtype=np.int64),
                z_observed=np.asarray(data["z_observed"], dtype=np.float64),
                entropy=np.asarray(data["entropy"], dtype=np.float64),
                z_sampled=np.asarray(data["z_sampled"], dtype=np.float64),
                num_samples=int(np.asarray(data["num_samples"])[0]),
            )


def concatenate_token_caches(caches: Iterable[TokenCache]) -> TokenCache:
    caches = list(caches)
    if not caches:
        raise ValueError("No token caches to concatenate")
    num_samples = caches[0].num_samples
    if any(cache.num_samples != num_samples for cache in caches):
        raise ValueError("Candidate sample counts differ across caches")
    offsets = [0]
    for cache in caches:
        lengths = np.diff(cache.offsets)
        offsets.extend((offsets[-1] + np.cumsum(lengths)).tolist())
    return TokenCache(
        offsets=np.asarray(offsets, dtype=np.int64),
        z_observed=np.concatenate([cache.z_observed for cache in caches]),
        entropy=np.concatenate([cache.entropy for cache in caches]),
        z_sampled=np.concatenate([cache.z_sampled for cache in caches]),
        num_samples=num_samples,
    )


def resolve_dtype(name: str):
    import torch

    table = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return table[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def resolve_proxy_devices(
    sampling_device: str,
    scoring_device: str,
) -> tuple[str, str]:
    """Resolve automatic placement without sharding either model across GPUs."""
    import torch

    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if sampling_device == "auto":
        sampling_device = "cuda:0" if count else "cpu"
    if scoring_device == "auto":
        scoring_device = "cuda:1" if count >= 2 else (
            "cuda:0" if count else "cpu"
        )
    return sampling_device, scoring_device


def load_local_model(path: Path, dtype_name: str, device: str = "auto"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(path),
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )
    model_kwargs = {
        "local_files_only": True,
        "trust_remote_code": True,
        "torch_dtype": resolve_dtype(dtype_name),
        "low_cpu_mem_usage": True,
    }
    if device == "auto":
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = {"": device}
    model = AutoModelForCausalLM.from_pretrained(str(path), **model_kwargs)
    model.eval()
    if tokenizer.eos_token_id is not None and tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def model_input_device(model):
    return model.get_input_embeddings().weight.device


def encode_text(tokenizer, text: str, max_length: int):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        return_token_type_ids=False,
        truncation=True,
        max_length=max_length,
    )
    if encoded.input_ids.shape[1] < 2:
        raise ValueError("A document tokenized to fewer than two tokens")
    return encoded


def encode_text_batch(tokenizer, texts: Sequence[str], max_length: int):
    encoded = tokenizer(
        list(texts),
        return_tensors="pt",
        return_token_type_ids=False,
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    if (encoded.attention_mask.sum(dim=1) < 2).any():
        raise ValueError("A document tokenized to fewer than two tokens")
    return encoded


def reference_statistics(
    model,
    tokenizer,
    texts: Sequence[str],
    *,
    num_samples: int,
    max_length: int,
    seed: int,
    chunk_tokens: int,
    common_vocab_size: int,
    document_batch_size: int = 1,
) -> ReferenceCache:
    import torch

    offsets = [0]
    observed_parts: list[np.ndarray] = []
    entropy_parts: list[np.ndarray] = []
    candidate_parts: list[np.ndarray] = []
    input_device = model_input_device(model)
    with torch.inference_mode():
        for batch_start in range(0, len(texts), document_batch_size):
            batch_stop = min(
                len(texts), batch_start + document_batch_size
            )
            encoded_cpu = encode_text_batch(
                tokenizer, texts[batch_start:batch_stop], max_length
            )
            encoded = encoded_cpu.to(input_device)
            output = model(**encoded, use_cache=False)
            for local_index, index in enumerate(
                range(batch_start, batch_stop)
            ):
                length = int(
                    encoded_cpu.attention_mask[local_index].sum()
                )
                logits = output.logits[
                    local_index, : length - 1, :common_vocab_size
                ]
                targets = encoded_cpu.input_ids[
                    local_index, 1:length
                ].to(logits.device)
                generator = torch.Generator(device=logits.device)
                generator.manual_seed(seed + 10007 * index)
                doc_entropy = []
                doc_candidates = []
                for start in range(0, len(targets), chunk_tokens):
                    stop = min(len(targets), start + chunk_tokens)
                    log_probs = torch.log_softmax(
                        logits[start:stop].float(), dim=-1
                    )
                    probs = torch.exp(log_probs)
                    doc_entropy.append(
                        (-(probs * log_probs).sum(dim=-1)).cpu().numpy()
                    )
                    doc_candidates.append(
                        torch.multinomial(
                            probs,
                            num_samples=num_samples,
                            replacement=True,
                            generator=generator,
                        )
                        .cpu()
                        .numpy()
                    )
                observed = targets.cpu().numpy()
                entropy = np.concatenate(doc_entropy)
                candidates = np.concatenate(doc_candidates)
                observed_parts.append(observed)
                entropy_parts.append(entropy)
                candidate_parts.append(candidates)
                offsets.append(offsets[-1] + len(observed))
            del output, encoded, encoded_cpu
            if batch_stop % 20 == 0 or batch_stop == len(texts):
                print(f"    GPT-J reference: {batch_stop}/{len(texts)}")
    return ReferenceCache(
        offsets=np.asarray(offsets, dtype=np.int64),
        observed_ids=np.concatenate(observed_parts),
        entropy=np.concatenate(entropy_parts),
        candidate_ids=np.concatenate(candidate_parts),
        num_samples=num_samples,
    )


def scoring_statistics(
    model,
    tokenizer,
    texts: Sequence[str],
    reference: ReferenceCache,
    *,
    max_length: int,
    chunk_tokens: int,
    common_vocab_size: int,
    document_batch_size: int = 1,
) -> TokenCache:
    import torch

    observed_parts: list[np.ndarray] = []
    sampled_parts: list[np.ndarray] = []
    input_device = model_input_device(model)
    with torch.inference_mode():
        for batch_start in range(0, len(texts), document_batch_size):
            batch_stop = min(
                len(texts), batch_start + document_batch_size
            )
            encoded_cpu = encode_text_batch(
                tokenizer, texts[batch_start:batch_stop], max_length
            )
            encoded = encoded_cpu.to(input_device)
            output = model(**encoded, use_cache=False)
            for local_index, index in enumerate(
                range(batch_start, batch_stop)
            ):
                length = int(
                    encoded_cpu.attention_mask[local_index].sum()
                )
                logits = output.logits[
                    local_index, : length - 1, :common_vocab_size
                ]
                targets = encoded_cpu.input_ids[
                    local_index, 1:length
                ].to(logits.device)
                left = int(reference.offsets[index])
                right = int(reference.offsets[index + 1])
                expected = reference.observed_ids[left:right]
                actual = targets.cpu().numpy()
                if not np.array_equal(actual, expected):
                    raise ValueError(
                        "GPT-J and GPT-Neo token IDs differ for document "
                        f"{index}; check tokenizer files"
                    )
                candidates = reference.candidate_ids[left:right]
                if candidates.size and candidates.max() >= logits.shape[-1]:
                    raise ValueError(
                        "Reference candidate ID exceeds scoring vocabulary"
                    )
                doc_observed = []
                doc_sampled = []
                for start in range(0, len(targets), chunk_tokens):
                    stop = min(len(targets), start + chunk_tokens)
                    log_probs = torch.log_softmax(
                        logits[start:stop].float(), dim=-1
                    )
                    target_chunk = targets[start:stop, None]
                    candidate_chunk = torch.as_tensor(
                        candidates[start:stop],
                        dtype=torch.long,
                        device=logits.device,
                    )
                    doc_observed.append(
                        log_probs.gather(-1, target_chunk)[
                            :, 0
                        ].cpu().numpy()
                    )
                    doc_sampled.append(
                        log_probs.gather(
                            -1, candidate_chunk
                        ).cpu().numpy()
                    )
                observed_parts.append(np.concatenate(doc_observed))
                sampled_parts.append(np.concatenate(doc_sampled))
            del output, encoded, encoded_cpu
            if batch_stop % 20 == 0 or batch_stop == len(texts):
                print(f"    GPT-Neo scoring: {batch_stop}/{len(texts)}")
    return TokenCache(
        offsets=reference.offsets.copy(),
        z_observed=np.concatenate(observed_parts),
        entropy=reference.entropy.copy(),
        z_sampled=np.concatenate(sampled_parts),
        num_samples=reference.num_samples,
    )


def cache_metadata(
    prefix: Path,
    data: dict[str, list[str]],
    *,
    num_samples: int,
    max_length: int,
    seed: int,
    common_vocab_size: int,
) -> dict:
    path = raw_data_path(prefix).resolve()
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "schema_version": 3,
        "raw_data_path": str(path),
        "raw_data_size": stat.st_size,
        "raw_data_sha256": digest.hexdigest(),
        "n_human": len(data["original"]),
        "n_machine": len(data["sampled"]),
        "num_samples": num_samples,
        "max_length": max_length,
        "seed": seed,
        "common_vocab_size": common_vocab_size,
        "reference_statistic": "GPT-J entropy and sampled token IDs",
        "scoring_statistic": "GPT-Neo observed/candidate log probabilities",
    }


def metadata_matches(path: Path, expected: dict) -> bool:
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8") as stream:
        actual = json.load(stream)
    return actual == expected


def ensure_token_caches(
    entries: dict[str, Path],
    *,
    source_model: str,
    cache_root: Path,
    sampling_model_dir: Path,
    scoring_model_dir: Path,
    sampling_dtype: str,
    scoring_dtype: str,
    num_samples: int,
    max_length: int,
    seed: int,
    chunk_tokens: int,
    overwrite: bool,
    sampling_device: str = "auto",
    scoring_device: str = "auto",
    document_batch_size: int = 2,
    cache_roles: dict[str, str] | None = None,
) -> dict[str, tuple[TokenCache, TokenCache]]:
    from transformers import AutoConfig

    sampling_config = AutoConfig.from_pretrained(
        str(sampling_model_dir), local_files_only=True, trust_remote_code=True
    )
    scoring_config = AutoConfig.from_pretrained(
        str(scoring_model_dir), local_files_only=True, trust_remote_code=True
    )
    common_vocab_size = min(
        int(sampling_config.vocab_size), int(scoring_config.vocab_size)
    )
    prepared: dict[str, tuple[dict[str, list[str]], Path, dict]] = {}
    pending: list[str] = []
    for domain, prefix in entries.items():
        data = load_raw_data(prefix)
        role = "shared" if cache_roles is None else cache_roles[domain]
        cell_dir = cache_root / source_model / role / domain
        expected = cache_metadata(
            prefix,
            data,
            num_samples=num_samples,
            max_length=max_length,
            seed=seed,
            common_vocab_size=common_vocab_size,
        )
        metadata_path = cell_dir / "metadata.json"
        final_paths = [
            cell_dir / "human_tokens.npz",
            cell_dir / "machine_tokens.npz",
        ]
        complete = all(path.is_file() for path in final_paths)
        if complete and metadata_matches(metadata_path, expected) and not overwrite:
            prepared[domain] = (data, cell_dir, expected)
            continue
        if complete and not overwrite:
            raise ValueError(
                f"Stale/incompatible Tensor cache in {cell_dir}; "
                "rerun with --overwrite-cache"
            )
        if any(path.is_file() for path in final_paths) and not overwrite:
            raise ValueError(
                f"Incomplete Tensor cache in {cell_dir}; "
                "rerun with --overwrite-cache"
            )
        prepared[domain] = (data, cell_dir, expected)
        pending.append(domain)

    if pending:
        print(f"Tensor cache: building reference statistics for {pending}")
        sampling_model, sampling_tokenizer = load_local_model(
            sampling_model_dir, sampling_dtype, sampling_device
        )
        try:
            for domain in pending:
                data, cell_dir, _ = prepared[domain]
                cell_dir.mkdir(parents=True, exist_ok=True)
                for split, key, offset in (
                    ("human", "original", 0),
                    ("machine", "sampled", 1000003),
                ):
                    reference_path = cell_dir / f"{split}_reference.npz"
                    if reference_path.is_file() and not overwrite:
                        continue
                    reference = reference_statistics(
                        sampling_model,
                        sampling_tokenizer,
                        data[key],
                        num_samples=num_samples,
                        max_length=max_length,
                        seed=seed + offset,
                        chunk_tokens=chunk_tokens,
                        common_vocab_size=common_vocab_size,
                        document_batch_size=document_batch_size,
                    )
                    reference.save(reference_path)
        finally:
            del sampling_model, sampling_tokenizer
            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"Tensor cache: building scoring statistics for {pending}")
        scoring_model, scoring_tokenizer = load_local_model(
            scoring_model_dir, scoring_dtype, scoring_device
        )
        try:
            for domain in pending:
                data, cell_dir, expected = prepared[domain]
                for split, key in (
                    ("human", "original"),
                    ("machine", "sampled"),
                ):
                    final_path = cell_dir / f"{split}_tokens.npz"
                    if final_path.is_file() and not overwrite:
                        continue
                    reference_path = cell_dir / f"{split}_reference.npz"
                    reference = ReferenceCache.load(reference_path)
                    if reference.num_samples != num_samples:
                        raise ValueError(
                            f"{reference_path} uses "
                            f"{reference.num_samples} candidates, expected "
                            f"{num_samples}; rerun with --overwrite-cache"
                        )
                    if len(reference.offsets) != len(data[key]) + 1:
                        raise ValueError(
                            f"{reference_path} document count does not match "
                            "the current raw data; rerun with --overwrite-cache"
                        )
                    final = scoring_statistics(
                        scoring_model,
                        scoring_tokenizer,
                        data[key],
                        reference,
                        max_length=max_length,
                        chunk_tokens=chunk_tokens,
                        common_vocab_size=common_vocab_size,
                        document_batch_size=document_batch_size,
                    )
                    final.save(final_path)
                atomic_json(cell_dir / "metadata.json", expected)
        finally:
            del scoring_model, scoring_tokenizer
            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for domain in pending:
            _, cell_dir, _ = prepared[domain]
            for split in ("human", "machine"):
                reference_path = cell_dir / f"{split}_reference.npz"
                if reference_path.is_file():
                    reference_path.unlink()

    result = {}
    for domain, (_, cell_dir, _) in prepared.items():
        result[domain] = (
            TokenCache.load(cell_dir / "human_tokens.npz"),
            TokenCache.load(cell_dir / "machine_tokens.npz"),
        )
    return result


@dataclass(frozen=True)
class BasisSpec:
    j_bases: int
    k_bases: int
    degree_z: int
    degree_h: int
    z_base_knots: tuple[float, ...]
    h_base_knots: tuple[float, ...]
    entropy_clip_low: float
    entropy_clip_high: float

    @property
    def dimension(self) -> int:
        return self.j_bases * self.k_bases


def strictly_increasing(values, gap: float = 1e-6) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    for index in range(1, len(result)):
        if result[index] <= result[index - 1]:
            result[index] = result[index - 1] + gap
    return result


def build_basis_spec(
    training_human: TokenCache,
    j_bases: int,
    k_bases: int,
    *,
    degree_z: int = 2,
    degree_h: int = 2,
    z_lower_quantile: float = 0.001,
    entropy_clip_low: float = 0.005,
    entropy_clip_high: float = 0.995,
) -> BasisSpec:
    z = training_human.z_observed
    levels = np.linspace(
        z_lower_quantile,
        1.0 - z_lower_quantile,
        j_bases - degree_z + 1,
    )
    z_knots = np.quantile(z, levels)
    z_knots[-1] = 0.0
    z_knots = strictly_increasing(z_knots)
    h_knots = np.linspace(0.0, 1.0, k_bases - degree_h + 1)
    return BasisSpec(
        j_bases=j_bases,
        k_bases=k_bases,
        degree_z=degree_z,
        degree_h=degree_h,
        z_base_knots=tuple(float(value) for value in z_knots),
        h_base_knots=tuple(float(value) for value in h_knots),
        entropy_clip_low=entropy_clip_low,
        entropy_clip_high=entropy_clip_high,
    )


def make_spline(knots, degree: int, expected_bases: int):
    knots = strictly_increasing(knots)
    transformer = SplineTransformer(
        degree=degree,
        knots=knots[:, None],
        include_bias=True,
        extrapolation="constant",
        sparse_output=False,
    )
    transformer.fit(np.asarray([[knots[0]], [knots[-1]]]))
    if transformer.n_features_out_ != expected_bases:
        raise ValueError("Unexpected spline output dimension")
    return transformer


def ecdf_transform(
    values,
    sorted_reference,
    low: float,
    high: float,
) -> np.ndarray:
    ranks = np.searchsorted(sorted_reference, values, side="right")
    transformed = (ranks + 0.5) / (len(sorted_reference) + 1.0)
    return np.clip(transformed, low, high)


def transform_token_cache(
    cache: TokenCache,
    spec: BasisSpec,
    sorted_training_entropy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    z_spline = make_spline(
        np.asarray(spec.z_base_knots), spec.degree_z, spec.j_bases
    )
    h_spline = make_spline(
        np.asarray(spec.h_base_knots), spec.degree_h, spec.k_bases
    )
    u = np.empty((cache.num_documents, spec.dimension), dtype=np.float64)
    v = np.empty(
        (cache.num_documents, spec.dimension, spec.dimension),
        dtype=np.float64,
    )
    for index in range(cache.num_documents):
        z_observed, entropy, z_sampled = cache.document(index)
        token_count, sample_count = z_sampled.shape
        entropy_u = ecdf_transform(
            entropy,
            sorted_training_entropy,
            spec.entropy_clip_low,
            spec.entropy_clip_high,
        )
        b_observed = z_spline.transform(z_observed[:, None])
        b_sampled = z_spline.transform(
            z_sampled.reshape(-1, 1)
        ).reshape(token_count, sample_count, spec.j_bases)
        c_entropy = h_spline.transform(entropy_u[:, None])
        phi_observed = (
            b_observed[:, :, None] * c_entropy[:, None, :]
        ).reshape(token_count, spec.dimension)
        phi_sampled = (
            b_sampled[:, :, :, None] * c_entropy[:, None, None, :]
        ).reshape(token_count, sample_count, spec.dimension)
        mean_phi = phi_sampled.mean(axis=1)
        u[index] = (phi_observed - mean_phi).sum(axis=0)
        centered = phi_sampled - mean_phi[:, None, :]
        v[index] = np.einsum(
            "tsd,tse->de", centered, centered, optimize=True
        ) / float(sample_count - 1)
    return u, v


def prepare_exact_requests(
    requests: dict[str, tuple[BasisSpec, np.ndarray]],
):
    prepared = {}
    results = {}
    for key, (spec, sorted_entropy) in requests.items():
        prepared[key] = (
            spec,
            np.asarray(sorted_entropy, dtype=np.float64),
            make_spline(
                np.asarray(spec.z_base_knots), spec.degree_z, spec.j_bases
            ),
            make_spline(
                np.asarray(spec.h_base_knots), spec.degree_h, spec.k_bases
            ),
        )
        results[key] = (
            np.zeros(spec.dimension, dtype=np.float64),
            np.zeros(
                (spec.dimension, spec.dimension), dtype=np.float64
            ),
        )
    return prepared, results


def accumulate_exact_moments(
    probs_np: np.ndarray,
    z_np: np.ndarray,
    z_observed: np.ndarray,
    entropy: np.ndarray,
    prepared,
    results,
) -> None:
    """Accumulate exact vocabulary moments from normalized q_r and log q_s."""
    token_count, vocab_size = probs_np.shape
    if z_np.shape != (token_count, vocab_size):
        raise ValueError("Probability and scoring arrays differ")
    if z_observed.shape != (token_count,):
        raise ValueError("Observed score count differs")
    for key, (
        spec,
        sorted_entropy,
        z_spline,
        h_spline,
    ) in prepared.items():
        entropy_u = ecdf_transform(
            entropy,
            sorted_entropy,
            spec.entropy_clip_low,
            spec.entropy_clip_high,
        )
        c_entropy = h_spline.transform(entropy_u[:, None])
        b_all = z_spline.transform(z_np.reshape(-1, 1)).reshape(
            token_count, vocab_size, spec.j_bases
        )
        b_observed = z_spline.transform(z_observed[:, None])
        mean_b = np.einsum(
            "tv,tvj->tj", probs_np, b_all, optimize=True
        )
        second_b = np.einsum(
            "tv,tvj,tvk->tjk",
            probs_np,
            b_all,
            b_all,
            optimize=True,
        )
        covariance_b = second_b - np.einsum(
            "tj,tk->tjk", mean_b, mean_b, optimize=True
        )
        phi_observed = (
            b_observed[:, :, None] * c_entropy[:, None, :]
        ).reshape(token_count, spec.dimension)
        mean_phi = (
            mean_b[:, :, None] * c_entropy[:, None, :]
        ).reshape(token_count, spec.dimension)
        u, v = results[key]
        u += (phi_observed - mean_phi).sum(axis=0)
        v += np.einsum(
            "tab,tc,td->acbd",
            covariance_b,
            c_entropy,
            c_entropy,
            optimize=True,
        ).reshape(spec.dimension, spec.dimension)


def exact_features_from_logits(
    logits_ref,
    logits_score,
    labels,
    requests: dict[str, tuple[BasisSpec, np.ndarray]],
    *,
    chunk_tokens: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Compute exact q_r-weighted Tensor moments without materializing T×V×D."""
    import torch

    # The reference implementation is used both by the production Torch path
    # and by CPU numerical-equivalence tests that pass NumPy arrays.
    logits_ref = torch.as_tensor(logits_ref)
    logits_score = torch.as_tensor(logits_score)
    if logits_ref.ndim != 2 or logits_score.ndim != 2:
        raise ValueError("Expected [tokens, vocabulary] logits")
    if logits_ref.shape[0] != logits_score.shape[0]:
        raise ValueError("Reference/scoring token counts differ")
    labels = torch.as_tensor(labels, dtype=torch.long).reshape(-1)
    if labels.numel() != logits_ref.shape[0]:
        raise ValueError("Label/token count differs")
    vocab_size = min(logits_ref.shape[-1], logits_score.shape[-1])
    if labels.numel() and int(labels.max()) >= vocab_size:
        raise ValueError("Observed token lies outside the shared vocabulary")

    prepared, results = prepare_exact_requests(requests)

    for start in range(0, labels.numel(), chunk_tokens):
        stop = min(labels.numel(), start + chunk_tokens)
        ref_chunk = logits_ref[start:stop, :vocab_size].float()
        score_chunk = logits_score[start:stop, :vocab_size].float()
        ref_log_probs = torch.log_softmax(ref_chunk, dim=-1)
        probs = torch.exp(ref_log_probs)
        score_log_probs = torch.log_softmax(score_chunk, dim=-1)
        entropy = (
            -(probs * ref_log_probs).sum(dim=-1).detach().cpu().numpy()
        )
        probs_np = probs.detach().cpu().numpy().astype(np.float64, copy=False)
        z_np = (
            score_log_probs.detach().cpu().numpy().astype(
                np.float64, copy=False
            )
        )
        label_np = labels[start:stop].cpu().numpy()
        row = np.arange(stop - start)
        z_observed = z_np[row, label_np]
        accumulate_exact_moments(
            probs_np,
            z_np,
            z_observed,
            entropy,
            prepared,
            results,
        )

    for key, (u, v) in results.items():
        results[key] = (u, 0.5 * (v + v.T))
    return results


@dataclass
class BatchedExactGroup:
    """Requests sharing one JxK shape, prepared once for repeated documents."""

    request_ids: list[str]
    specs: list[BasisSpec]
    sorted_entropies: list[np.ndarray]
    h_splines: list[SplineTransformer]
    z_knots: object
    z_low: object
    z_high: object


def prepare_batched_exact_plan(
    requests: dict[str, tuple[BasisSpec, np.ndarray]],
    device,
) -> list[BatchedExactGroup]:
    """Move spline knot vectors to the feature device once."""
    import torch

    grouped: dict[tuple[int, int, int, int], list] = {}
    for request_id, (spec, sorted_entropy) in requests.items():
        grouped.setdefault(
            (
                spec.j_bases,
                spec.k_bases,
                spec.degree_z,
                spec.degree_h,
            ),
            [],
        ).append((request_id, spec, sorted_entropy))

    result = []
    for entries in grouped.values():
        request_ids = []
        specs = []
        sorted_entropies = []
        h_splines = []
        z_knots = []
        for request_id, spec, sorted_entropy in entries:
            z_spline = make_spline(
                np.asarray(spec.z_base_knots),
                spec.degree_z,
                spec.j_bases,
            )
            request_ids.append(request_id)
            specs.append(spec)
            sorted_entropies.append(
                np.asarray(sorted_entropy, dtype=np.float64)
            )
            h_splines.append(
                make_spline(
                    np.asarray(spec.h_base_knots),
                    spec.degree_h,
                    spec.k_bases,
                )
            )
            z_knots.append(
                np.asarray(z_spline.bsplines_[0].t, dtype=np.float32)
            )
        result.append(
            BatchedExactGroup(
                request_ids=request_ids,
                specs=specs,
                sorted_entropies=sorted_entropies,
                h_splines=h_splines,
                z_knots=torch.as_tensor(
                    np.stack(z_knots), dtype=torch.float32, device=device
                ),
                z_low=torch.as_tensor(
                    [spec.z_base_knots[0] for spec in specs],
                    dtype=torch.float32,
                    device=device,
                ),
                z_high=torch.as_tensor(
                    [spec.z_base_knots[-1] for spec in specs],
                    dtype=torch.float32,
                    device=device,
                ),
            )
        )
    return result


def torch_bspline_basis(
    values,
    knots,
    degree: int,
    low,
    high,
):
    """Evaluate a batch of constant-extrapolated B-spline bases in Torch."""
    import torch

    values = values.float()
    shape = (len(low),) + (1,) * values.ndim
    x = values.unsqueeze(0)
    x = torch.maximum(x, low.reshape(shape))
    x = torch.minimum(x, high.reshape(shape))
    x = x.unsqueeze(-1)
    knot_shape = (knots.shape[0],) + (1,) * values.ndim + (knots.shape[1],)
    t = knots.reshape(knot_shape)
    basis = ((x >= t[..., :-1]) & (x < t[..., 1:])).to(values.dtype)
    for order in range(1, degree + 1):
        count = basis.shape[-1] - 1
        left_denom = t[..., order : order + count] - t[..., :count]
        right_denom = (
            t[..., order + 1 : order + 1 + count]
            - t[..., 1 : 1 + count]
        )
        left = torch.where(
            left_denom != 0,
            (x - t[..., :count]) / left_denom,
            torch.zeros_like(left_denom),
        )
        right = torch.where(
            right_denom != 0,
            (t[..., order + 1 : order + 1 + count] - x)
            / right_denom,
            torch.zeros_like(right_denom),
        )
        basis = left * basis[..., :count] + right * basis[..., 1 : count + 1]
    return basis


def exact_features_from_logits_batched(
    logits_ref,
    logits_score,
    labels,
    requests: dict[str, tuple[BasisSpec, np.ndarray]],
    *,
    chunk_tokens: int,
    request_batch_size: int,
    prepared_plan: list[BatchedExactGroup] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """GPU-batch all basis requests while keeping only a token chunk resident."""
    import torch

    logits_ref = torch.as_tensor(logits_ref)
    logits_score = torch.as_tensor(logits_score)
    if logits_ref.ndim != 2 or logits_score.ndim != 2:
        raise ValueError("Expected [tokens, vocabulary] logits")
    if logits_ref.shape[0] != logits_score.shape[0]:
        raise ValueError("Reference/scoring token counts differ")
    labels = torch.as_tensor(labels, dtype=torch.long).reshape(-1)
    if labels.numel() != logits_ref.shape[0]:
        raise ValueError("Label/token count differs")
    vocab_size = min(logits_ref.shape[-1], logits_score.shape[-1])
    if labels.numel() and int(labels.max()) >= vocab_size:
        raise ValueError("Observed token lies outside the shared vocabulary")
    if request_batch_size < 1:
        raise ValueError("request_batch_size must be positive")

    feature_device = logits_score.device
    plan = prepared_plan or prepare_batched_exact_plan(
        requests, feature_device
    )
    results = {
        request_id: (
            np.zeros(spec.dimension, dtype=np.float64),
            np.zeros((spec.dimension, spec.dimension), dtype=np.float64),
        )
        for request_id, (spec, _) in requests.items()
    }

    labels = labels.to(feature_device)
    for start in range(0, labels.numel(), chunk_tokens):
        stop = min(labels.numel(), start + chunk_tokens)
        ref_chunk = logits_ref[start:stop, :vocab_size].float()
        ref_log_probs = torch.log_softmax(ref_chunk, dim=-1)
        probs_ref = torch.exp(ref_log_probs)
        entropy = (
            -(probs_ref * ref_log_probs).sum(dim=-1)
            .detach()
            .cpu()
            .numpy()
        )
        probs = probs_ref.to(feature_device, non_blocking=True)
        score_log_probs = torch.log_softmax(
            logits_score[start:stop, :vocab_size].float(), dim=-1
        )
        row = torch.arange(stop - start, device=feature_device)
        observed_z = score_log_probs[row, labels[start:stop]]

        for group in plan:
            total = len(group.request_ids)
            for left in range(0, total, request_batch_size):
                right = min(total, left + request_batch_size)
                knots = group.z_knots[left:right]
                low = group.z_low[left:right]
                high = group.z_high[left:right]
                degree = group.specs[left].degree_z
                b_all = torch_bspline_basis(
                    score_log_probs, knots, degree, low, high
                )
                weighted_b = b_all * probs[None, :, :, None]
                mean_b = weighted_b.sum(dim=2)
                j_bases = b_all.shape[-1]
                second_b = torch.zeros(
                    (
                        right - left,
                        stop - start,
                        j_bases,
                        j_bases,
                    ),
                    dtype=torch.float32,
                    device=feature_device,
                )
                diagonal = torch.arange(
                    j_bases, device=feature_device
                )
                for offset in range(degree + 1):
                    values = (
                        weighted_b[..., : j_bases - offset]
                        * b_all[..., offset:]
                    ).sum(dim=2)
                    rows = diagonal[: j_bases - offset]
                    columns = diagonal[offset:]
                    second_b[..., rows, columns] = values
                    if offset:
                        second_b[..., columns, rows] = values
                b_observed = torch_bspline_basis(
                    observed_z, knots, degree, low, high
                )

                c_values = []
                for index in range(left, right):
                    spec = group.specs[index]
                    entropy_u = ecdf_transform(
                        entropy,
                        group.sorted_entropies[index],
                        spec.entropy_clip_low,
                        spec.entropy_clip_high,
                    )
                    c_values.append(
                        group.h_splines[index].transform(
                            entropy_u[:, None]
                        )
                    )
                c_entropy = torch.as_tensor(
                    np.stack(c_values),
                    dtype=torch.float32,
                    device=feature_device,
                )
                delta_b = b_observed - mean_b
                u_chunk = torch.einsum(
                    "rtj,rtk->rjk", delta_b, c_entropy
                )
                covariance_b = second_b - torch.einsum(
                    "rtj,rtk->rtjk", mean_b, mean_b
                )
                v_chunk = torch.einsum(
                    "rtab,rtc,rtd->racbd",
                    covariance_b,
                    c_entropy,
                    c_entropy,
                )
                for local, request_id in enumerate(
                    group.request_ids[left:right]
                ):
                    spec = group.specs[left + local]
                    u, v = results[request_id]
                    u += (
                        u_chunk[local]
                        .reshape(spec.dimension)
                        .double()
                        .cpu()
                        .numpy()
                    )
                    v += (
                        v_chunk[local]
                        .reshape(spec.dimension, spec.dimension)
                        .double()
                        .cpu()
                        .numpy()
                    )
                del (
                    b_all,
                    weighted_b,
                    mean_b,
                    second_b,
                    b_observed,
                    c_entropy,
                    delta_b,
                    covariance_b,
                    u_chunk,
                    v_chunk,
                )
        del ref_chunk, ref_log_probs, probs_ref, probs, score_log_probs

    for key, (u, v) in results.items():
        results[key] = (u, 0.5 * (v + v.T))
    return results


def text_sequence_digest(texts: Sequence[str]) -> str:
    """Hash an ordered text set without depending on its source path."""
    digest = hashlib.sha256()
    digest.update(b"water-text-sequence-v1\0")
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def exact_request_digest(
    spec: BasisSpec,
    sorted_entropy: np.ndarray,
) -> str:
    """Identify the full-vocabulary feature definition, not the task name."""
    digest = hashlib.sha256()
    digest.update(b"water-exact-request-v1\0")
    digest.update(
        json.dumps(
            asdict(spec),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    entropy = np.asarray(sorted_entropy, dtype="<f8")
    digest.update(entropy.shape[0].to_bytes(8, "big"))
    digest.update(entropy.tobytes(order="C"))
    return digest.hexdigest()


def model_directory_signature(path: Path) -> dict:
    """Cheaply invalidate feature caches when a local checkpoint changes."""
    path = path.resolve()
    patterns = (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "model*.safetensors",
        "pytorch_model*.bin",
        "*.index.json",
    )
    files = {}
    for pattern in patterns:
        for item in path.glob(pattern):
            if item.is_file():
                stat = item.stat()
                files[item.name] = [stat.st_size, stat.st_mtime_ns]
    return {"path": str(path), "files": dict(sorted(files.items()))}


def exact_model_context(
    sampling_model_dir: Path,
    scoring_model_dir: Path,
    *,
    sampling_dtype: str,
    scoring_dtype: str,
    max_length: int,
    common_vocab_size: int,
) -> tuple[str, dict]:
    metadata = {
        "schema_version": 1,
        "sampling_model": model_directory_signature(sampling_model_dir),
        "scoring_model": model_directory_signature(scoring_model_dir),
        "sampling_dtype": sampling_dtype,
        "scoring_dtype": scoring_dtype,
        "max_length": max_length,
        "common_vocab_size": common_vocab_size,
        "shared_vocabulary_rule": "crop_then_separately_normalize",
    }
    encoded = json.dumps(
        metadata, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), metadata


class ExactFeatureStore:
    """Content-addressed, cross-task store for per-document exact u/V."""

    def __init__(
        self,
        root: Path,
        *,
        model_context_id: str,
        model_context: dict,
    ) -> None:
        self.root = root / "full_vocab_features_v1" / model_context_id
        self.model_context_id = model_context_id
        self.model_context = model_context

    def request_dir(
        self,
        texts: Sequence[str],
        request_id: str,
    ) -> Path:
        text_id = text_sequence_digest(texts)
        return self.root / text_id[:2] / text_id / request_id

    def expected_metadata(
        self,
        texts: Sequence[str],
        request_id: str,
        spec: BasisSpec,
        sorted_entropy: np.ndarray,
    ) -> dict:
        return {
            "schema_version": 1,
            "model_context_id": self.model_context_id,
            "model_context": self.model_context,
            "text_sequence_sha256": text_sequence_digest(texts),
            "num_documents": len(texts),
            "request_sha256": request_id,
            "basis_spec": json.loads(json.dumps(asdict(spec))),
            "sorted_entropy_sha256": hashlib.sha256(
                np.asarray(sorted_entropy, dtype="<f8").tobytes(order="C")
            ).hexdigest(),
            "storage_dtype": "float32",
        }

    def is_valid(
        self,
        texts: Sequence[str],
        request_id: str,
        spec: BasisSpec,
        sorted_entropy: np.ndarray,
    ) -> bool:
        directory = self.request_dir(texts, request_id)
        metadata_path = directory / "metadata.json"
        if not (
            metadata_path.is_file()
            and (directory / "u.npy").is_file()
            and (directory / "v.npy").is_file()
        ):
            return False
        try:
            with metadata_path.open("r", encoding="utf-8") as stream:
                actual = json.load(stream)
        except (OSError, json.JSONDecodeError):
            return False
        return actual == self.expected_metadata(
            texts, request_id, spec, sorted_entropy
        )

    def load(
        self,
        texts: Sequence[str],
        request_id: str,
        spec: BasisSpec,
        sorted_entropy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_valid(texts, request_id, spec, sorted_entropy):
            raise FileNotFoundError(
                "Missing or incompatible exact feature cache for "
                f"text={text_sequence_digest(texts)[:12]} "
                f"request={request_id[:12]}"
            )
        directory = self.request_dir(texts, request_id)
        return (
            np.load(directory / "u.npy", mmap_mode="r"),
            np.load(directory / "v.npy", mmap_mode="r"),
        )


def build_global_exact_request_plan(
    caches: dict[str, tuple[TokenCache, TokenCache]],
    structures: Sequence[tuple[int, int]],
) -> tuple[
    dict[str, tuple[BasisSpec, np.ndarray]],
    dict[tuple[tuple[str, ...], int, int], str],
]:
    """Plan all outer-test and inner-validation bases for these domains."""
    domains = tuple(sorted(caches))
    if len(domains) < 3:
        raise ValueError("Exact request planning needs at least three domains")
    requests: dict[str, tuple[BasisSpec, np.ndarray]] = {}
    lookup: dict[tuple[tuple[str, ...], int, int], str] = {}
    for subset_size in sorted({len(domains) - 1, len(domains) - 2}):
        for subset in itertools.combinations(domains, subset_size):
            pooled_human = concatenate_token_caches(
                caches[name][0] for name in subset
            )
            sorted_entropy = np.sort(pooled_human.entropy)
            for j_bases, k_bases in structures:
                spec = build_basis_spec(
                    pooled_human, j_bases, k_bases
                )
                request_id = exact_request_digest(spec, sorted_entropy)
                requests.setdefault(
                    request_id, (spec, sorted_entropy)
                )
                lookup[(subset, j_bases, k_bases)] = request_id
    return requests, lookup


def write_exact_feature_set(
    store: ExactFeatureStore,
    texts: Sequence[str],
    requests: dict[str, tuple[BasisSpec, np.ndarray]],
    *,
    sampling_model,
    sampling_tokenizer,
    scoring_model,
    scoring_tokenizer,
    max_length: int,
    chunk_tokens: int,
    document_batch_size: int,
    request_batch_size: int,
    label: str,
    overwrite: bool,
) -> int:
    """Batch documents and basis requests, checkpointing after every batch."""
    pending = {
        request_id: value
        for request_id, value in requests.items()
        if overwrite
        or not store.is_valid(texts, request_id, value[0], value[1])
    }
    if not pending:
        print(
            f"    feature cache hit {label}: "
            f"{len(requests)}/{len(requests)} requests",
            flush=True,
        )
        return 0

    if document_batch_size < 1:
        raise ValueError("document_batch_size must be positive")
    if request_batch_size < 1:
        raise ValueError("request_batch_size must be positive")

    text_id = text_sequence_digest(texts)
    text_root = store.root / text_id[:2] / text_id
    text_root.mkdir(parents=True, exist_ok=True)
    progress_path = text_root / "build_progress_v2.json"
    request_ids = sorted(pending)
    completed = 0
    resume = False
    if progress_path.is_file() and not overwrite:
        try:
            with progress_path.open("r", encoding="utf-8") as stream:
                progress = json.load(stream)
            completed = int(progress["completed_documents"])
            resume = (
                progress.get("schema_version") == 2
                and progress.get("text_sequence_sha256") == text_id
                and progress.get("request_ids") == request_ids
                and progress.get("num_documents") == len(texts)
                and 0 <= completed <= len(texts)
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            completed = 0
            resume = False

    maps = {}
    for request_id, (spec, _) in pending.items():
        directory = store.request_dir(texts, request_id)
        directory.mkdir(parents=True, exist_ok=True)
        u_temp = directory / "u.partial.npy"
        v_temp = directory / "v.partial.npy"
        expected_u_shape = (len(texts), spec.dimension)
        expected_v_shape = (
            len(texts),
            spec.dimension,
            spec.dimension,
        )
        if resume:
            try:
                u_map = np.load(u_temp, mmap_mode="r+")
                v_map = np.load(v_temp, mmap_mode="r+")
                if (
                    u_map.shape != expected_u_shape
                    or v_map.shape != expected_v_shape
                    or u_map.dtype != np.float32
                    or v_map.dtype != np.float32
                ):
                    raise ValueError("partial cache shape/dtype mismatch")
            except (OSError, ValueError):
                resume = False
                completed = 0
        if not resume:
            u_map = np.lib.format.open_memmap(
                u_temp,
                mode="w+",
                dtype=np.float32,
                shape=expected_u_shape,
            )
            v_map = np.lib.format.open_memmap(
                v_temp,
                mode="w+",
                dtype=np.float32,
                shape=expected_v_shape,
            )
        maps[request_id] = (
            u_map,
            v_map,
            u_temp,
            v_temp,
        )

    import torch

    sampling_device = model_input_device(sampling_model)
    scoring_device = model_input_device(scoring_model)
    plan = prepare_batched_exact_plan(pending, scoring_device)
    if completed:
        print(
            f"    resuming full-vocab cache {label}: "
            f"{completed}/{len(texts)} documents already durable",
            flush=True,
        )
    started = time.monotonic()
    with torch.inference_mode():
        for batch_start in range(
            completed, len(texts), document_batch_size
        ):
            batch_stop = min(
                len(texts), batch_start + document_batch_size
            )
            batch_texts = texts[batch_start:batch_stop]
            ref_encoded_cpu = encode_text_batch(
                sampling_tokenizer, batch_texts, max_length
            )
            score_encoded_cpu = encode_text_batch(
                scoring_tokenizer, batch_texts, max_length
            )
            if not torch.equal(
                ref_encoded_cpu.input_ids,
                score_encoded_cpu.input_ids,
            ) or not torch.equal(
                ref_encoded_cpu.attention_mask,
                score_encoded_cpu.attention_mask,
            ):
                raise ValueError(
                    "GPT-J and GPT-Neo token IDs differ in document batch "
                    f"starting at {batch_start}; check tokenizer files"
                )
            ref_encoded = ref_encoded_cpu.to(sampling_device)
            score_encoded = score_encoded_cpu.to(scoring_device)
            ref_output = sampling_model(**ref_encoded, use_cache=False)
            score_output = scoring_model(**score_encoded, use_cache=False)
            for local_index, index in enumerate(
                range(batch_start, batch_stop)
            ):
                length = int(
                    score_encoded_cpu.attention_mask[
                        local_index
                    ].sum()
                )
                labels = score_encoded_cpu.input_ids[
                    local_index, 1:length
                ]
                features = exact_features_from_logits_batched(
                    ref_output.logits[local_index, : length - 1],
                    score_output.logits[local_index, : length - 1],
                    labels,
                    pending,
                    chunk_tokens=chunk_tokens,
                    request_batch_size=request_batch_size,
                    prepared_plan=plan,
                )
                for request_id, (u, v) in features.items():
                    maps[request_id][0][index] = u
                    maps[request_id][1][index] = v
            del (
                ref_output,
                score_output,
                ref_encoded,
                score_encoded,
                ref_encoded_cpu,
                score_encoded_cpu,
            )
            for u_map, v_map, _, _ in maps.values():
                u_map.flush()
                v_map.flush()
            atomic_json(
                progress_path,
                {
                    "schema_version": 2,
                    "text_sequence_sha256": text_id,
                    "request_ids": request_ids,
                    "num_documents": len(texts),
                    "completed_documents": batch_stop,
                },
            )
            elapsed = max(time.monotonic() - started, 1e-9)
            processed = batch_stop - completed
            rate = processed / elapsed
            remaining = (len(texts) - batch_stop) / max(rate, 1e-9)
            print(
                f"    full-vocab cache {label}: "
                f"{batch_stop}/{len(texts)} documents, "
                f"{len(pending)} requests, {rate:.3f} doc/s, "
                f"ETA {remaining / 60.0:.1f} min",
                flush=True,
            )

    for request_id, (spec, sorted_entropy) in pending.items():
        u_map, v_map, u_temp, v_temp = maps[request_id]
        u_map.flush()
        v_map.flush()
        del u_map, v_map
        directory = store.request_dir(texts, request_id)
        os.replace(u_temp, directory / "u.npy")
        os.replace(v_temp, directory / "v.npy")
        atomic_json(
            directory / "metadata.json",
            store.expected_metadata(
                texts, request_id, spec, sorted_entropy
            ),
        )
    if progress_path.is_file():
        progress_path.unlink()
    return len(pending)


def ensure_global_exact_feature_cache(
    raw_data: dict[str, dict[str, list[str]]],
    requests: dict[str, tuple[BasisSpec, np.ndarray]],
    *,
    store: ExactFeatureStore,
    sampling_model_dir: Path,
    scoring_model_dir: Path,
    sampling_dtype: str,
    scoring_dtype: str,
    max_length: int,
    chunk_tokens: int,
    document_batch_size: int = 2,
    request_batch_size: int = 15,
    sampling_device: str = "auto",
    scoring_device: str = "auto",
    overwrite: bool = False,
) -> int:
    """Warm all source-domain tasks once; skip model loading on a full hit."""
    pending_sets = []
    for domain, data in raw_data.items():
        for split, key in (("human", "original"), ("machine", "sampled")):
            texts = data[key]
            missing = {
                request_id: value
                for request_id, value in requests.items()
                if overwrite
                or not store.is_valid(
                    texts, request_id, value[0], value[1]
                )
            }
            if missing:
                pending_sets.append((domain, split, texts, missing))
    if not pending_sets:
        print(
            "Tensor exact feature cache: complete hit; "
            "proxy models will not be loaded.",
            flush=True,
        )
        return 0

    print(
        "Tensor exact feature cache: "
        f"{len(pending_sets)} text sets have missing requests; "
        "loading proxy models once.",
        flush=True,
    )
    sampling_model, sampling_tokenizer = load_local_model(
        sampling_model_dir, sampling_dtype, sampling_device
    )
    scoring_model, scoring_tokenizer = load_local_model(
        scoring_model_dir, scoring_dtype, scoring_device
    )
    built = 0
    try:
        for domain, split, texts, missing in pending_sets:
            built += write_exact_feature_set(
                store,
                texts,
                missing,
                sampling_model=sampling_model,
                sampling_tokenizer=sampling_tokenizer,
                scoring_model=scoring_model,
                scoring_tokenizer=scoring_tokenizer,
                max_length=max_length,
                chunk_tokens=chunk_tokens,
                document_batch_size=document_batch_size,
                request_batch_size=request_batch_size,
                label=f"{domain}/{split}",
                overwrite=overwrite,
            )
    finally:
        del (
            sampling_model,
            sampling_tokenizer,
            scoring_model,
            scoring_tokenizer,
        )
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return built


@dataclass(frozen=True)
class Config:
    j_bases: int
    k_bases: int
    smooth_lambda: float
    ridge_lambda: float
    covariance_shrinkage: float

    @property
    def dimension(self) -> int:
        return self.j_bases * self.k_bases

    @property
    def key(self) -> tuple:
        return (
            self.j_bases,
            self.k_bases,
            self.smooth_lambda,
            self.ridge_lambda,
            self.covariance_shrinkage,
        )


def second_difference_matrix(size: int) -> np.ndarray:
    if size < 3:
        return np.zeros((0, size), dtype=np.float64)
    matrix = np.zeros((size - 2, size), dtype=np.float64)
    rows = np.arange(size - 2)
    matrix[rows, rows] = 1.0
    matrix[rows, rows + 1] = -2.0
    matrix[rows, rows + 2] = 1.0
    return matrix


def smoothness_matrix(spec: BasisSpec) -> np.ndarray:
    j, k = spec.j_bases, spec.k_bases
    dz = second_difference_matrix(j)
    dh = second_difference_matrix(k)
    penalty = np.zeros((j * k, j * k), dtype=np.float64)
    penalty += np.kron(dz.T @ dz, np.eye(k)) / ((j - 2) * k)
    penalty += np.kron(np.eye(j), dh.T @ dh) / (j * (k - 2))
    return 0.5 * (penalty + penalty.T)


def within_class_covariance(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    return centered.T @ centered / (len(values) - 1)


def score_documents(
    u: np.ndarray,
    v: np.ndarray,
    theta: np.ndarray,
    epsilon: float = 1e-8,
) -> np.ndarray:
    numerator = u @ theta
    denominator2 = np.einsum(
        "d,nde,e->n", theta, v, theta, optimize=True
    )
    return numerator / np.sqrt(np.clip(denominator2, epsilon, None))


def auc(human_scores: np.ndarray, machine_scores: np.ndarray) -> float:
    labels = np.concatenate(
        [np.zeros(len(human_scores)), np.ones(len(machine_scores))]
    )
    scores = np.concatenate([human_scores, machine_scores])
    return float(roc_auc_score(labels, scores))


def fit_rayleigh(
    u_human: np.ndarray,
    v_human: np.ndarray,
    u_machine: np.ndarray,
    v_machine: np.ndarray,
    spec: BasisSpec,
    config: Config,
    epsilon: float = 1e-8,
) -> np.ndarray:
    delta = u_machine.mean(axis=0) - u_human.mean(axis=0)
    covariance = (
        within_class_covariance(u_human)
        + within_class_covariance(u_machine)
    )
    rho = config.covariance_shrinkage
    covariance = (
        (1.0 - rho) * covariance
        + rho * np.diag(np.diag(covariance))
    )
    covariance = 0.5 * (covariance + covariance.T)
    scale = float(np.trace(covariance) / covariance.shape[0])
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    penalty = smoothness_matrix(spec)
    positive = np.linalg.eigvalsh(penalty)
    positive = positive[positive > 1e-12]
    penalty /= float(positive.mean()) if len(positive) else 1.0
    system = (
        covariance
        + scale * config.smooth_lambda * penalty
        + scale * config.ridge_lambda * np.eye(covariance.shape[0])
    )
    eigenvalues, eigenvectors = np.linalg.eigh(
        0.5 * (system + system.T)
    )
    floor = max(
        epsilon * scale,
        np.finfo(np.float64).eps * max(float(eigenvalues[-1]), 1.0),
    )
    theta = eigenvectors @ (
        (eigenvectors.T @ delta) / np.maximum(eigenvalues, floor)
    )
    mean_v = np.concatenate([v_human, v_machine]).mean(axis=0)
    norm2 = float(theta @ mean_v @ theta)
    if np.isfinite(norm2) and norm2 > epsilon:
        theta /= math.sqrt(norm2)
    else:
        theta /= max(float(np.linalg.norm(theta)), epsilon)
    train_h = score_documents(u_human, v_human, theta, epsilon)
    train_m = score_documents(u_machine, v_machine, theta, epsilon)
    if auc(train_h, train_m) < 0.5:
        theta = -theta
    return theta


def concatenate_features(
    parts: Iterable[tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray]:
    parts = list(parts)
    return (
        np.concatenate([part[0] for part in parts], axis=0),
        np.concatenate([part[1] for part in parts], axis=0),
    )


def transform_domain(
    pair: tuple[TokenCache, TokenCache],
    spec: BasisSpec,
    sorted_entropy: np.ndarray,
):
    return (
        transform_token_cache(pair[0], spec, sorted_entropy),
        transform_token_cache(pair[1], spec, sorted_entropy),
    )


def select_config(
    training: dict[str, tuple[TokenCache, TokenCache]],
    configs: Sequence[Config],
) -> tuple[Config, list[dict]]:
    domains = list(training)
    if len(domains) < 2:
        raise ValueError(
            "Tensor-Rayleigh needs at least two training domains "
            "(use at least three total datasets)"
        )
    records = {config.key: [] for config in configs}
    structures = sorted({(c.j_bases, c.k_bases) for c in configs})
    for validation_domain in domains:
        fit_domains = [name for name in domains if name != validation_domain]
        fit_human = concatenate_token_caches(
            training[name][0] for name in fit_domains
        )
        sorted_entropy = np.sort(fit_human.entropy)
        for j_bases, k_bases in structures:
            spec = build_basis_spec(fit_human, j_bases, k_bases)
            transformed_fit = [
                transform_domain(training[name], spec, sorted_entropy)
                for name in fit_domains
            ]
            human_fit = concatenate_features(
                part[0] for part in transformed_fit
            )
            machine_fit = concatenate_features(
                part[1] for part in transformed_fit
            )
            validation = transform_domain(
                training[validation_domain], spec, sorted_entropy
            )
            for config in configs:
                if (
                    config.j_bases,
                    config.k_bases,
                ) != (j_bases, k_bases):
                    continue
                theta = fit_rayleigh(
                    human_fit[0],
                    human_fit[1],
                    machine_fit[0],
                    machine_fit[1],
                    spec,
                    config,
                )
                human_scores = score_documents(
                    validation[0][0], validation[0][1], theta
                )
                machine_scores = score_documents(
                    validation[1][0], validation[1][1], theta
                )
                records[config.key].append(
                    {
                        "validation_domain": validation_domain,
                        "auc": auc(human_scores, machine_scores),
                    }
                )
    summary = []
    by_key = {config.key: config for config in configs}
    for key, folds in records.items():
        values = [fold["auc"] for fold in folds]
        summary.append(
            {
                **asdict(by_key[key]),
                "mean_validation_auc": float(np.mean(values)),
                "std_validation_auc": float(np.std(values)),
                "folds": folds,
            }
        )
    summary.sort(
        key=lambda item: (
            item["mean_validation_auc"],
            -item["j_bases"] * item["k_bases"],
            -item["smooth_lambda"],
            -item["ridge_lambda"],
            -item["covariance_shrinkage"],
        ),
        reverse=True,
    )
    best_item = summary[0]
    best = Config(
        best_item["j_bases"],
        best_item["k_bases"],
        best_item["smooth_lambda"],
        best_item["ridge_lambda"],
        best_item["covariance_shrinkage"],
    )
    return best, summary


def evaluate(
    test_pair: tuple[TokenCache, TokenCache],
    training: dict[str, tuple[TokenCache, TokenCache]],
    configs: Sequence[Config],
) -> tuple[dict, np.ndarray, BasisSpec]:
    selected, validation = select_config(training, configs)
    pooled_human_cache = concatenate_token_caches(
        pair[0] for pair in training.values()
    )
    sorted_entropy = np.sort(pooled_human_cache.entropy)
    spec = build_basis_spec(
        pooled_human_cache,
        selected.j_bases,
        selected.k_bases,
    )
    transformed_train = [
        transform_domain(pair, spec, sorted_entropy)
        for pair in training.values()
    ]
    human_train = concatenate_features(
        part[0] for part in transformed_train
    )
    machine_train = concatenate_features(
        part[1] for part in transformed_train
    )
    theta = fit_rayleigh(
        human_train[0],
        human_train[1],
        machine_train[0],
        machine_train[1],
        spec,
        selected,
    )
    transformed_test = transform_domain(test_pair, spec, sorted_entropy)
    human_scores = score_documents(
        transformed_test[0][0], transformed_test[0][1], theta
    )
    machine_scores = score_documents(
        transformed_test[1][0], transformed_test[1][1], theta
    )
    return (
        {
            "selected": selected,
            "validation": validation,
            "human_scores": human_scores,
            "machine_scores": machine_scores,
            "sorted_entropy": sorted_entropy,
        },
        theta,
        spec,
    )


def load_exact_domain_pair(
    store: ExactFeatureStore,
    data: dict[str, list[str]],
    request_id: str,
    request: tuple[BasisSpec, np.ndarray],
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    spec, sorted_entropy = request
    return (
        store.load(
            data["original"], request_id, spec, sorted_entropy
        ),
        store.load(
            data["sampled"], request_id, spec, sorted_entropy
        ),
    )


def select_config_exact_cached(
    training_domains: Sequence[str],
    raw_data: dict[str, dict[str, list[str]]],
    configs: Sequence[Config],
    *,
    requests: dict[str, tuple[BasisSpec, np.ndarray]],
    request_lookup: dict[tuple[tuple[str, ...], int, int], str],
    store: ExactFeatureStore,
) -> tuple[Config, list[dict]]:
    domains = list(training_domains)
    if len(domains) < 2:
        raise ValueError(
            "Tensor-Rayleigh needs at least two training domains "
            "(use at least three total datasets)"
        )
    records = {config.key: [] for config in configs}
    structures = sorted({(c.j_bases, c.k_bases) for c in configs})
    for validation_domain in domains:
        fit_domains = sorted(
            name for name in domains if name != validation_domain
        )
        fit_subset = tuple(fit_domains)
        for j_bases, k_bases in structures:
            request_id = request_lookup[
                (fit_subset, j_bases, k_bases)
            ]
            request = requests[request_id]
            spec = request[0]
            transformed = {
                name: load_exact_domain_pair(
                    store,
                    raw_data[name],
                    request_id,
                    request,
                )
                for name in domains
            }
            human_fit = concatenate_features(
                transformed[name][0] for name in fit_domains
            )
            machine_fit = concatenate_features(
                transformed[name][1] for name in fit_domains
            )
            validation = transformed[validation_domain]
            for config in configs:
                if (
                    config.j_bases,
                    config.k_bases,
                ) != (j_bases, k_bases):
                    continue
                theta = fit_rayleigh(
                    human_fit[0],
                    human_fit[1],
                    machine_fit[0],
                    machine_fit[1],
                    spec,
                    config,
                )
                human_scores = score_documents(
                    validation[0][0], validation[0][1], theta
                )
                machine_scores = score_documents(
                    validation[1][0], validation[1][1], theta
                )
                records[config.key].append(
                    {
                        "validation_domain": validation_domain,
                        "auc": auc(human_scores, machine_scores),
                    }
                )

    summary = []
    by_key = {config.key: config for config in configs}
    for key, folds in records.items():
        values = [fold["auc"] for fold in folds]
        summary.append(
            {
                **asdict(by_key[key]),
                "mean_validation_auc": float(np.mean(values)),
                "std_validation_auc": float(np.std(values)),
                "folds": folds,
            }
        )
    summary.sort(
        key=lambda item: (
            item["mean_validation_auc"],
            -item["j_bases"] * item["k_bases"],
            -item["smooth_lambda"],
            -item["ridge_lambda"],
            -item["covariance_shrinkage"],
        ),
        reverse=True,
    )
    best_item = summary[0]
    best = Config(
        best_item["j_bases"],
        best_item["k_bases"],
        best_item["smooth_lambda"],
        best_item["ridge_lambda"],
        best_item["covariance_shrinkage"],
    )
    return best, summary


def evaluate_exact_cached(
    *,
    test_domain: str,
    raw_data: dict[str, dict[str, list[str]]],
    training_domains: Sequence[str],
    configs: Sequence[Config],
    requests: dict[str, tuple[BasisSpec, np.ndarray]],
    request_lookup: dict[tuple[tuple[str, ...], int, int], str],
    store: ExactFeatureStore,
) -> tuple[dict, np.ndarray, BasisSpec]:
    selected, validation = select_config_exact_cached(
        training_domains,
        raw_data,
        configs,
        requests=requests,
        request_lookup=request_lookup,
        store=store,
    )
    training_subset = tuple(sorted(training_domains))
    request_id = request_lookup[
        (training_subset, selected.j_bases, selected.k_bases)
    ]
    request = requests[request_id]
    spec, sorted_entropy = request
    transformed_train = {
        name: load_exact_domain_pair(
            store, raw_data[name], request_id, request
        )
        for name in training_domains
    }
    transformed_test = load_exact_domain_pair(
        store, raw_data[test_domain], request_id, request
    )
    human_train = concatenate_features(
        pair[0] for pair in transformed_train.values()
    )
    machine_train = concatenate_features(
        pair[1] for pair in transformed_train.values()
    )
    theta = fit_rayleigh(
        human_train[0],
        human_train[1],
        machine_train[0],
        machine_train[1],
        spec,
        selected,
    )
    human_scores = score_documents(
        transformed_test[0][0], transformed_test[0][1], theta
    )
    machine_scores = score_documents(
        transformed_test[1][0], transformed_test[1][1], theta
    )
    return (
        {
            "selected": selected,
            "validation": validation,
            "human_scores": human_scores,
            "machine_scores": machine_scores,
            "sorted_entropy": sorted_entropy,
        },
        theta,
        spec,
    )


def parse_train_domains(values: Sequence[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--train-domain must use NAME=DATA_PREFIX")
        name, prefix = value.split("=", 1)
        name = name.strip()
        if not name or not prefix.strip():
            raise ValueError("--train-domain contains an empty name/path")
        if name in result:
            raise ValueError(f"Duplicate train domain: {name}")
        result[name] = Path(prefix)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--dataset-file", type=Path, required=True)
    parser.add_argument(
        "--train-domain",
        action="append",
        default=[],
        help="Repeat NAME=DATA_PREFIX for each non-test domain",
    )
    parser.add_argument("--sampling-model-dir", type=Path, required=True)
    parser.add_argument("--scoring-model-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument(
        "--expectation-mode",
        choices=("full_vocab", "sampled"),
        default="full_vocab",
        help="Exact shared-vocabulary moments or Monte Carlo candidates",
    )
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--sampling-dtype", default="float16")
    parser.add_argument("--scoring-dtype", default="float16")
    parser.add_argument("--chunk-tokens", type=int, default=32)
    parser.add_argument("--document-batch-size", type=int, default=2)
    parser.add_argument("--request-batch-size", type=int, default=15)
    parser.add_argument("--sampling-device", default="auto")
    parser.add_argument("--scoring-device", default="auto")
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--structures", default="4x4,6x6,8x6")
    parser.add_argument(
        "--smooth-lambdas", default="0,1e-3,1e-2,1e-1,1"
    )
    parser.add_argument(
        "--ridge-lambdas", default="1e-5,1e-4,1e-3,1e-2"
    )
    parser.add_argument(
        "--covariance-shrinkages", default="0,0.05,0.2"
    )
    parser.add_argument("--overwrite-cache", action="store_true")
    args = parser.parse_args(argv)

    train_domains = parse_train_domains(args.train_domain)
    if len(train_domains) < 2:
        parser.error("Provide at least two --train-domain values")
    if args.dataset in train_domains:
        parser.error("The test domain must not occur in --train-domain")
    if args.num_samples < 2:
        parser.error("--num-samples must be at least 2")
    if args.chunk_tokens < 1:
        parser.error("--chunk-tokens must be positive")
    if args.document_batch_size < 1:
        parser.error("--document-batch-size must be positive")
    if args.request_batch_size < 1:
        parser.error("--request-batch-size must be positive")

    sampling_device, scoring_device = resolve_proxy_devices(
        args.sampling_device,
        args.scoring_device,
    )
    print(
        "Tensor proxy placement: "
        f"GPT-J={sampling_device}, GPT-Neo={scoring_device}; "
        f"document batch={args.document_batch_size}, "
        f"basis batch={args.request_batch_size}.",
        flush=True,
    )

    entries = {args.dataset: args.dataset_file, **train_domains}
    cache_roles = {name: "train" for name in train_domains}
    cache_roles[args.dataset] = "test"
    raw_data = {
        name: load_raw_data(prefix)
        for name, prefix in entries.items()
    }
    cache_samples = (
        args.num_samples if args.expectation_mode == "sampled" else 2
    )
    effective_cache_root = (
        args.cache_root
        / (
            f"sampled_{args.num_samples}"
            if args.expectation_mode == "sampled"
            else "full_vocab_observed"
        )
    )
    caches = ensure_token_caches(
        entries,
        source_model=args.source_model,
        cache_root=effective_cache_root,
        sampling_model_dir=args.sampling_model_dir,
        scoring_model_dir=args.scoring_model_dir,
        sampling_dtype=args.sampling_dtype,
        scoring_dtype=args.scoring_dtype,
        num_samples=cache_samples,
        max_length=args.max_length,
        seed=args.seed,
        chunk_tokens=args.chunk_tokens,
        overwrite=args.overwrite_cache,
        sampling_device=sampling_device,
        scoring_device=scoring_device,
        document_batch_size=args.document_batch_size,
        cache_roles=cache_roles,
    )
    structures = parse_structures(args.structures)
    configs = [
        Config(j, k, smooth, ridge, shrinkage)
        for (j, k), smooth, ridge, shrinkage in itertools.product(
            structures,
            parse_float_csv(args.smooth_lambdas),
            parse_float_csv(args.ridge_lambdas),
            parse_float_csv(args.covariance_shrinkages),
        )
    ]
    if args.expectation_mode == "sampled":
        evaluation, theta, spec = evaluate(
            caches[args.dataset],
            {name: caches[name] for name in train_domains},
            configs,
        )
    else:
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
            int(sampling_config.vocab_size),
            int(scoring_config.vocab_size),
        )
        model_context_id, model_context = exact_model_context(
            args.sampling_model_dir,
            args.scoring_model_dir,
            sampling_dtype=args.sampling_dtype,
            scoring_dtype=args.scoring_dtype,
            max_length=args.max_length,
            common_vocab_size=common_vocab_size,
        )
        store = ExactFeatureStore(
            args.cache_root,
            model_context_id=model_context_id,
            model_context=model_context,
        )
        requests, request_lookup = build_global_exact_request_plan(
            caches, structures
        )
        print(
            "Tensor full-vocabulary mode: preparing one content-addressed "
            f"feature bank for all {len(entries)} held-out-domain tasks "
            f"({len(requests)} unique basis requests).",
            flush=True,
        )
        built_requests = ensure_global_exact_feature_cache(
            raw_data,
            requests,
            store=store,
            sampling_model_dir=args.sampling_model_dir,
            scoring_model_dir=args.scoring_model_dir,
            sampling_dtype=args.sampling_dtype,
            scoring_dtype=args.scoring_dtype,
            max_length=args.max_length,
            chunk_tokens=args.chunk_tokens,
            document_batch_size=args.document_batch_size,
            request_batch_size=args.request_batch_size,
            sampling_device=sampling_device,
            scoring_device=scoring_device,
            overwrite=args.overwrite_cache,
        )
        print(
            "Tensor exact feature cache ready: "
            f"{built_requests} request/text-set entries built; "
            "selection and evaluation are cache-only.",
            flush=True,
        )
        evaluation, theta, spec = evaluate_exact_cached(
            test_domain=args.dataset,
            raw_data=raw_data,
            training_domains=list(train_domains),
            configs=configs,
            requests=requests,
            request_lookup=request_lookup,
            store=store,
        )
    human_scores = evaluation["human_scores"]
    machine_scores = evaluation["machine_scores"]
    labels = np.concatenate(
        [np.zeros(len(human_scores)), np.ones(len(machine_scores))]
    )
    scores = np.concatenate([human_scores, machine_scores])
    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    roc_auc = float(roc_auc_score(labels, scores))
    pr_auc = float(sklearn_auc(recall, precision))
    result = {
        "name": "Tensor-Rayleigh",
        "method": (
            "Water raw-z x GPT-J entropy-ECDF Tensor-Spline with "
            "regularized Rayleigh estimator"
        ),
        "info": {
            "dataset": args.dataset,
            "source_model": args.source_model,
            "training_domains": list(train_domains),
            "n_samples": len(human_scores),
            "expectation_mode": args.expectation_mode,
            "candidate_samples": (
                args.num_samples
                if args.expectation_mode == "sampled"
                else None
            ),
            "max_length": args.max_length,
            "document_batch_size": args.document_batch_size,
            "request_batch_size": args.request_batch_size,
            "sampling_device": sampling_device,
            "scoring_device": scoring_device,
            "hyperparameter_candidates": len(configs),
            "exact_feature_cache": (
                "content-addressed across held-out-domain and source-model tasks"
                if args.expectation_mode == "full_vocab"
                else None
            ),
        },
        "selected_hyperparameters": asdict(evaluation["selected"]),
        "basis_spec": asdict(spec),
        "domain_validation": evaluation["validation"],
        "predictions": {
            "real": human_scores.tolist(),
            "samples": machine_scores.tolist(),
        },
        "metrics": {
            "roc_auc": roc_auc,
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        },
        "pr_metrics": {
            "pr_auc": pr_auc,
            "precision": precision.tolist(),
            "recall": recall.tolist(),
        },
        "model": {
            "theta": theta.tolist(),
        },
        "notes": {
            "test_domain_is_untouched_during_selection": True,
            "entropy_source": "GPT-J-6B reference distribution",
            "log_probability_source": "GPT-Neo-2.7B scoring model",
            "conditional_moments": (
                "exact q_r-weighted moments over the shared vocabulary"
                if args.expectation_mode == "full_vocab"
                else f"{args.num_samples} samples per token from GPT-J"
            ),
            "features_computed_once": (
                args.expectation_mode == "full_vocab"
            ),
            "full_vocab_backend": (
                "GPU-batched Torch B-splines with banded second moments"
                if args.expectation_mode == "full_vocab"
                else None
            ),
            "partial_document_cache_resumable": (
                args.expectation_mode == "full_vocab"
            ),
            "shared_vocabulary_rule": (
                "crop both proxy logits to min(vocab_ref, vocab_score), "
                "then normalize, matching official AdaDetectGPT"
            ),
        },
    }
    result_path = Path(f"{args.output_file}.TensorRayleigh.json")
    atomic_json(result_path, result)
    print(
        f"Tensor-Rayleigh {args.source_model}/{args.dataset}: "
        f"ROC AUC={roc_auc:.6f}, PR AUC={pr_auc:.6f}"
    )
    print(f"Selected: {asdict(evaluation['selected'])}")
    print(f"Results written to {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
