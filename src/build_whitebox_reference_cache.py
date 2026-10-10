#!/usr/bin/env python3
"""Build exact/10k white-box features for the Water experiment.

The scoring model and sampling model are the same source model.  For every
scored token this script computes:

* Fast-DetectGPT with 10,000 categorical samples;
* Fast-DetectGPT with analytic full-vocabulary moments; and
* AdaDetectGPT's exact full-vocabulary B-spline moments; and
* Water-Full's exact entropy-conditioned tensor-spline moments.

Only document-level sufficient statistics and observed token log
probabilities are saved.  The 10,000 sampled token values are never written to
disk.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_MODELS = ["llama-2-7b", "opt-2.7b", "opt-6.7b", "falcon-7b"]
DEFAULT_DATASETS = ["c4", "squad", "xsum", "writing", "pubmed"]
DEFAULT_MODEL_PATHS = {
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "opt-2.7b": "facebook/opt-2.7b",
    "opt-6.7b": "facebook/opt-6.7b",
    "falcon-7b": "tiiuae/falcon-7b",
}
DEFAULT_DTYPES = {
    "llama-2-7b": "bfloat16",
    "opt-2.7b": "float16",
    "opt-6.7b": "float16",
    "falcon-7b": "bfloat16",
}

PACKAGE_VERSION = "2026.08.09-v5-water-full"
CACHE_SCHEMA_VERSION = 2


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_structures(value: str) -> list[tuple[int, int]]:
    structures: list[tuple[int, int]] = []
    for item in parse_csv(value):
        parts = item.lower().split("x")
        if len(parts) != 2:
            raise ValueError(
                f"invalid Water structure {item!r}; use forms such as 4x4,6x6"
            )
        structure = (int(parts[0]), int(parts[1]))
        if structure not in structures:
            structures.append(structure)
    return structures


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_mapping(path: Path | None, default: dict[str, str]) -> dict[str, str]:
    if path is None:
        return dict(default)
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return {str(key): str(item) for key, item in value.items()}


@dataclass(frozen=True)
class AdaSplineSpec:
    start: float = -32.0
    end: float = 0.0
    n_bases: int = 7
    degree: int = 2
    intercept: bool = True

    @property
    def dimension(self) -> int:
        return self.n_bases + int(self.intercept)


@dataclass(frozen=True)
class WaterSplineSpec:
    """The existing Water tensor basis, serialized without sklearn objects."""

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

    @property
    def name(self) -> str:
        return f"J{self.j_bases}_K{self.k_bases}"


@dataclass(frozen=True)
class WaterBasisReference:
    specs: tuple[WaterSplineSpec, ...]
    sorted_human_entropy: np.ndarray
    provenance: dict[str, Any]


@dataclass(frozen=True)
class TorchWaterSpec:
    spec: WaterSplineSpec
    z_knots: Any
    h_knots: Any


@dataclass(frozen=True)
class TorchWaterReference:
    specs: tuple[TorchWaterSpec, ...]
    sorted_human_entropy: Any


def ada_knots(spec: AdaSplineSpec) -> np.ndarray:
    """Reproduce AdaDetectGPT/scripts/helper.py's knot construction."""
    if spec.end <= spec.start:
        raise ValueError("spline end must exceed spline start")
    if spec.degree < 0:
        raise ValueError("spline degree must be non-negative")
    m = spec.degree - 1
    nk = spec.n_bases - m
    if nk < 2:
        raise ValueError("n_bases is too small for the requested degree")
    width = spec.end - spec.start
    expanded_start = spec.start - width * 0.001
    expanded_end = spec.end + width * 0.001
    delta = (expanded_end - expanded_start) / (nk - 1)
    knots = np.linspace(
        expanded_start - delta * (m + 1),
        expanded_end + delta * (m + 1),
        nk + 2 * m + 2,
        dtype=np.float64,
    )
    expected = spec.n_bases + spec.degree + 1
    if len(knots) != expected:
        raise AssertionError(f"expected {expected} knots, found {len(knots)}")
    return knots


def strictly_increasing(values: np.ndarray, gap: float = 1e-6) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    for index in range(1, len(result)):
        if result[index] <= result[index - 1]:
            result[index] = result[index - 1] + gap
    return result


def spline_full_knots(base_knots: Iterable[float], degree: int) -> np.ndarray:
    """Reproduce sklearn SplineTransformer's expanded knot vector.

    Water uses explicitly supplied base knots with ``include_bias=True`` and
    ``extrapolation='constant'``. sklearn extends the two boundary spacings by
    ``degree`` knots on each side. Values are clamped to the base interval
    before evaluation to implement constant extrapolation.
    """
    base = strictly_increasing(np.asarray(tuple(base_knots), dtype=np.float64))
    if len(base) < 2:
        raise ValueError("a spline needs at least two base knots")
    if degree < 0:
        raise ValueError("spline degree must be non-negative")
    left_gap = base[1] - base[0]
    right_gap = base[-1] - base[-2]
    left = base[0] - left_gap * np.arange(degree, 0, -1, dtype=np.float64)
    right = base[-1] + right_gap * np.arange(1, degree + 1, dtype=np.float64)
    return np.concatenate([left, base, right])


def build_water_specs(
    z_observed: np.ndarray,
    structures: Iterable[tuple[int, int]],
    *,
    degree_z: int,
    degree_h: int,
    z_lower_quantile: float,
    z_upper_bound: float,
    entropy_clip_low: float,
    entropy_clip_high: float,
) -> tuple[WaterSplineSpec, ...]:
    z = np.asarray(z_observed, dtype=np.float64)
    if z.ndim != 1 or len(z) == 0 or not np.all(np.isfinite(z)):
        raise ValueError("invalid development-human observed log probabilities")
    result: list[WaterSplineSpec] = []
    for j_bases, k_bases in structures:
        if j_bases <= degree_z or k_bases <= degree_h:
            raise ValueError("Water basis count must exceed spline degree")
        n_z_knots = j_bases - degree_z + 1
        levels = np.linspace(
            z_lower_quantile, 1.0 - z_lower_quantile, n_z_knots
        )
        z_knots = np.quantile(z, levels)
        z_knots[0] = np.quantile(z, z_lower_quantile)
        z_knots[-1] = z_upper_bound
        z_knots = strictly_increasing(z_knots)
        n_h_knots = k_bases - degree_h + 1
        h_knots = np.linspace(0.0, 1.0, n_h_knots, dtype=np.float64)
        result.append(
            WaterSplineSpec(
                j_bases=j_bases,
                k_bases=k_bases,
                degree_z=degree_z,
                degree_h=degree_h,
                z_base_knots=tuple(float(value) for value in z_knots),
                h_base_knots=tuple(float(value) for value in h_knots),
                entropy_clip_low=entropy_clip_low,
                entropy_clip_high=entropy_clip_high,
            )
        )
    return tuple(result)


def save_water_reference(path: Path, reference: WaterBasisReference) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        sorted_human_entropy=np.asarray(
            reference.sorted_human_entropy, dtype=np.float32
        ),
        specs_json=np.asarray(
            [json.dumps([asdict(spec) for spec in reference.specs], sort_keys=True)]
        ),
        provenance_json=np.asarray(
            [json.dumps(reference.provenance, sort_keys=True)]
        ),
    )
    os.replace(temporary, path)


def load_water_reference(path: Path) -> WaterBasisReference:
    with np.load(path, allow_pickle=False) as data:
        specs = tuple(
            WaterSplineSpec(
                **{
                    **value,
                    "z_base_knots": tuple(value["z_base_knots"]),
                    "h_base_knots": tuple(value["h_base_knots"]),
                }
            )
            for value in json.loads(str(np.asarray(data["specs_json"])[0]))
        )
        entropy = np.asarray(data["sorted_human_entropy"], dtype=np.float32)
        provenance = json.loads(str(np.asarray(data["provenance_json"])[0]))
    if entropy.ndim != 1 or len(entropy) == 0:
        raise ValueError(f"{path}: invalid human entropy reference")
    if not np.all(np.isfinite(entropy)) or np.any(entropy[1:] < entropy[:-1]):
        raise ValueError(f"{path}: human entropy reference is not finite/sorted")
    return WaterBasisReference(specs, entropy, provenance)


def prepare_torch_water_reference(
    reference: WaterBasisReference, device
) -> TorchWaterReference:
    import torch

    return TorchWaterReference(
        specs=tuple(
            TorchWaterSpec(
                spec=spec,
                z_knots=torch.as_tensor(
                    spline_full_knots(spec.z_base_knots, spec.degree_z),
                    dtype=torch.float32,
                    device=device,
                ),
                h_knots=torch.as_tensor(
                    spline_full_knots(spec.h_base_knots, spec.degree_h),
                    dtype=torch.float32,
                    device=device,
                ),
            )
            for spec in reference.specs
        ),
        sorted_human_entropy=torch.as_tensor(
            reference.sorted_human_entropy, dtype=torch.float32, device=device
        ),
    )


def torch_bspline_basis(values, knots, degree: int):
    """Evaluate all B-spline columns using the Cox--de Boor recursion.

    ``values`` may have any shape.  A final basis dimension is appended.
    This implementation is differentiability-agnostic and used under
    inference_mode only.
    """
    import torch

    x = values.unsqueeze(-1)
    knot = torch.as_tensor(knots, dtype=values.dtype, device=values.device)
    basis = ((x >= knot[:-1]) & (x < knot[1:])).to(values.dtype)
    for order in range(1, degree + 1):
        count = basis.shape[-1] - 1
        left_denominator = knot[order : order + count] - knot[:count]
        right_denominator = (
            knot[order + 1 : order + 1 + count] - knot[1 : 1 + count]
        )
        left_weight = torch.where(
            left_denominator != 0,
            (x - knot[:count]) / left_denominator,
            torch.zeros((), dtype=values.dtype, device=values.device),
        )
        right_weight = torch.where(
            right_denominator != 0,
            (knot[order + 1 : order + 1 + count] - x)
            / right_denominator,
            torch.zeros((), dtype=values.dtype, device=values.device),
        )
        basis = left_weight * basis[..., :count] + right_weight * basis[..., 1:]
    return basis


def add_intercept(basis, intercept: bool):
    if not intercept:
        return basis
    import torch

    ones = torch.ones((*basis.shape[:-1], 1), dtype=basis.dtype, device=basis.device)
    return torch.cat([ones, basis], dim=-1)


def torch_constant_spline_basis(values, full_knots, degree: int, lower: float, upper: float):
    """Evaluate the Water/sklearn basis with constant boundary extrapolation."""
    clamped = values.clamp(min=float(lower), max=float(upper))
    return torch_bspline_basis(clamped, full_knots, degree)


@dataclass
class DocumentStatistics:
    z_observed: np.ndarray
    fast_mc_numerator: float
    fast_mc_variance: float
    fast_exact_numerator: float
    fast_exact_variance: float
    ada_u: np.ndarray
    ada_v: np.ndarray
    water_u: dict[str, np.ndarray]
    water_v: dict[str, np.ndarray]


@dataclass
class ReferenceSplitCache:
    offsets: np.ndarray
    z_observed: np.ndarray
    fast_mc_numerator: np.ndarray
    fast_mc_variance: np.ndarray
    fast_exact_numerator: np.ndarray
    fast_exact_variance: np.ndarray
    ada_u: np.ndarray
    ada_v: np.ndarray
    water_u: dict[str, np.ndarray]
    water_v: dict[str, np.ndarray]

    @property
    def num_documents(self) -> int:
        return int(len(self.offsets) - 1)

    def observed_document(self, index: int) -> np.ndarray:
        start = int(self.offsets[index])
        stop = int(self.offsets[index + 1])
        return self.z_observed[start:stop]

    @classmethod
    def from_documents(cls, documents: Iterable[DocumentStatistics]):
        docs = list(documents)
        if not docs:
            raise ValueError("cannot pack an empty split")
        lengths = np.asarray([len(doc.z_observed) for doc in docs], dtype=np.int64)
        offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(lengths)])
        water_names = tuple(sorted(docs[0].water_u))
        if not water_names:
            raise ValueError("documents contain no Water-Full statistics")
        for doc in docs:
            if tuple(sorted(doc.water_u)) != water_names:
                raise ValueError("inconsistent Water u structures across documents")
            if tuple(sorted(doc.water_v)) != water_names:
                raise ValueError("inconsistent Water v structures across documents")
        return cls(
            offsets=offsets,
            z_observed=np.concatenate([doc.z_observed for doc in docs]).astype(np.float32),
            fast_mc_numerator=np.asarray(
                [doc.fast_mc_numerator for doc in docs], dtype=np.float64
            ),
            fast_mc_variance=np.asarray(
                [doc.fast_mc_variance for doc in docs], dtype=np.float64
            ),
            fast_exact_numerator=np.asarray(
                [doc.fast_exact_numerator for doc in docs], dtype=np.float64
            ),
            fast_exact_variance=np.asarray(
                [doc.fast_exact_variance for doc in docs], dtype=np.float64
            ),
            ada_u=np.stack([doc.ada_u for doc in docs]).astype(np.float64),
            ada_v=np.stack([doc.ada_v for doc in docs]).astype(np.float64),
            water_u={
                name: np.stack([doc.water_u[name] for doc in docs]).astype(np.float64)
                for name in water_names
            },
            water_v={
                name: np.stack([doc.water_v[name] for doc in docs]).astype(np.float64)
                for name in water_names
            },
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp.npz")
        water_values = {
            **{f"water_u__{name}": value for name, value in self.water_u.items()},
            **{f"water_v__{name}": value for name, value in self.water_v.items()},
        }
        np.savez_compressed(
            temporary,
            offsets=self.offsets,
            z_observed=self.z_observed,
            fast_mc_numerator=self.fast_mc_numerator,
            fast_mc_variance=self.fast_mc_variance,
            fast_exact_numerator=self.fast_exact_numerator,
            fast_exact_variance=self.fast_exact_variance,
            ada_u=self.ada_u,
            ada_v=self.ada_v,
            **water_values,
        )
        os.replace(temporary, path)

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as data:
            core_names = (
                "offsets",
                "z_observed",
                "fast_mc_numerator",
                "fast_mc_variance",
                "fast_exact_numerator",
                "fast_exact_variance",
                "ada_u",
                "ada_v",
            )
            water_u = {
                name.removeprefix("water_u__"): np.asarray(data[name])
                for name in data.files
                if name.startswith("water_u__")
            }
            water_v = {
                name.removeprefix("water_v__"): np.asarray(data[name])
                for name in data.files
                if name.startswith("water_v__")
            }
            return cls(
                **{name: np.asarray(data[name]) for name in core_names},
                water_u=water_u,
                water_v=water_v,
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
        raise ValueError(f"unsupported dtype {name!r}") from exc


def load_model(model_name: str, model_path: str, dtype_name: str):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=resolve_dtype(dtype_name),
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    module = type(model).__module__
    if model_name == "falcon-7b" and not module.startswith("transformers.models.falcon."):
        raise RuntimeError(
            "Falcon must use Transformers' maintained implementation; "
            f"loaded module={module!r}"
        )
    raw_device_map = getattr(model, "hf_device_map", None)
    serializable_device_map = (
        None
        if raw_device_map is None
        else {str(key): str(value) for key, value in raw_device_map.items()}
    )
    cpu_offloaded_modules = (
        []
        if serializable_device_map is None
        else [
            key
            for key, value in serializable_device_map.items()
            if value.lower() == "cpu"
        ]
    )
    disk_offloaded_modules = (
        []
        if serializable_device_map is None
        else [
            key
            for key, value in serializable_device_map.items()
            if value.lower() == "disk"
        ]
    )
    runtime = {
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "model_path": str(Path(model_path)),
        "model_class": type(model).__name__,
        "model_module": module,
        "tokenizer_class": type(tokenizer).__name__,
        "requested_dtype": dtype_name,
        "loaded_dtype": str(next(model.parameters()).dtype),
        "device_map": "auto",
        "resolved_device_map": serializable_device_map,
        "cpu_offloaded_modules": cpu_offloaded_modules,
        "disk_offloaded_modules": disk_offloaded_modules,
        "trust_remote_code": False,
    }
    if cpu_offloaded_modules or disk_offloaded_modules:
        print(
            "WARNING: model has CPU/disk-offloaded modules: "
            f"cpu={len(cpu_offloaded_modules)}, disk={len(disk_offloaded_modules)}. "
            "This is valid but slower and may not reproduce a legacy GPU-only "
            "BF16 forward bit-for-bit. Check nvidia-smi before a full run."
        )
    return model, tokenizer, runtime


def model_input_device(model):
    return model.get_input_embeddings().weight.device


def stable_document_seed(base_seed: int, split_index: int, document_index: int) -> int:
    # Keep seeds stable when a run is resumed or token chunk size changes.
    return int(base_seed + 1_000_003 * split_index + 10_007 * document_index)


def conditional_statistics(
    logits,
    targets,
    *,
    observed_override=None,
    spec: AdaSplineSpec,
    knots: np.ndarray,
    water_reference: TorchWaterReference,
    fast_samples: int,
    token_chunk_size: int,
    sample_seed: int,
) -> DocumentStatistics:
    import torch

    if fast_samples < 2:
        raise ValueError("Fast Monte Carlo requires at least two samples")
    generator = torch.Generator(device=logits.device)
    generator.manual_seed(sample_seed)
    observed_parts = []
    fast_mc_numerator = 0.0
    fast_mc_variance = 0.0
    fast_exact_numerator = 0.0
    fast_exact_variance = 0.0
    ada_u = torch.zeros(spec.dimension, dtype=torch.float64, device="cpu")
    ada_v = torch.zeros(
        (spec.dimension, spec.dimension), dtype=torch.float64, device="cpu"
    )
    water_u = {
        item.spec.name: torch.zeros(
            item.spec.dimension, dtype=torch.float64, device="cpu"
        )
        for item in water_reference.specs
    }
    water_v = {
        item.spec.name: torch.zeros(
            (item.spec.dimension, item.spec.dimension),
            dtype=torch.float64,
            device="cpu",
        )
        for item in water_reference.specs
    }

    for start in range(0, logits.shape[0], token_chunk_size):
        stop = min(logits.shape[0], start + token_chunk_size)
        chunk_logits = logits[start:stop].float()
        chunk_targets = targets[start:stop].to(chunk_logits.device)
        log_probs = torch.log_softmax(chunk_logits, dim=-1)
        probs = torch.exp(log_probs)
        if observed_override is None:
            observed = log_probs.gather(-1, chunk_targets[:, None])[:, 0]
        else:
            observed = observed_override[start:stop].to(chunk_logits.device)

        exact_mean = torch.sum(probs * log_probs, dim=-1)
        exact_var = torch.sum(
            probs * torch.square(log_probs - exact_mean[:, None]), dim=-1
        )
        sampled_ids = torch.multinomial(
            probs,
            num_samples=fast_samples,
            replacement=True,
            generator=generator,
        )
        sampled_log_probs = log_probs.gather(-1, sampled_ids)
        mc_mean = sampled_log_probs.mean(dim=-1)
        mc_var = sampled_log_probs.var(dim=-1, unbiased=True)

        clamped_vocab = torch.clamp_min(log_probs, spec.start)
        phi_vocab = add_intercept(
            torch_bspline_basis(clamped_vocab, knots, spec.degree),
            spec.intercept,
        )
        phi_observed = add_intercept(
            torch_bspline_basis(
                torch.clamp_min(observed, spec.start), knots, spec.degree
            ),
            spec.intercept,
        )
        phi_mean = torch.einsum("tv,tvd->td", probs, phi_vocab)
        phi_second = torch.einsum("tv,tvd,tve->tde", probs, phi_vocab, phi_vocab)
        phi_covariance = phi_second - torch.einsum(
            "td,te->tde", phi_mean, phi_mean
        )

        # Water-Full uses the exact entropy from this same forward but maps it
        # through the frozen development-human ECDF. For each Water basis, the
        # vocabulary is marginalized exactly; only document U and V are kept.
        entropy = -exact_mean
        entropy_ranks = torch.searchsorted(
            water_reference.sorted_human_entropy,
            entropy.contiguous(),
            right=True,
        )
        entropy_ecdf = (
            entropy_ranks.to(dtype=log_probs.dtype) + 0.5
        ) / float(len(water_reference.sorted_human_entropy) + 1)
        for water_item in water_reference.specs:
            water_spec = water_item.spec
            entropy_unit = entropy_ecdf.clamp(
                min=water_spec.entropy_clip_low,
                max=water_spec.entropy_clip_high,
            )
            c_entropy = torch_constant_spline_basis(
                entropy_unit,
                water_item.h_knots,
                water_spec.degree_h,
                water_spec.h_base_knots[0],
                water_spec.h_base_knots[-1],
            )
            b_vocab = torch_constant_spline_basis(
                log_probs,
                water_item.z_knots,
                water_spec.degree_z,
                water_spec.z_base_knots[0],
                water_spec.z_base_knots[-1],
            )
            b_observed = torch_constant_spline_basis(
                observed,
                water_item.z_knots,
                water_spec.degree_z,
                water_spec.z_base_knots[0],
                water_spec.z_base_knots[-1],
            )
            b_mean = torch.einsum("tv,tvj->tj", probs, b_vocab)
            b_second = torch.einsum(
                "tv,tvi,tvj->tij", probs, b_vocab, b_vocab
            )
            b_covariance = b_second - torch.einsum(
                "ti,tj->tij", b_mean, b_mean
            )
            observed_tensor = (
                b_observed[:, :, None] * c_entropy[:, None, :]
            ).reshape(-1, water_spec.dimension)
            mean_tensor = (
                b_mean[:, :, None] * c_entropy[:, None, :]
            ).reshape(-1, water_spec.dimension)
            covariance_tensor = torch.einsum(
                "tij,tk,tl->tikjl", b_covariance, c_entropy, c_entropy
            ).reshape(-1, water_spec.dimension, water_spec.dimension)
            water_u[water_spec.name] += torch.sum(
                observed_tensor - mean_tensor, dim=0
            ).double().cpu()
            water_v[water_spec.name] += torch.sum(
                covariance_tensor, dim=0
            ).double().cpu()
            del (
                entropy_unit,
                c_entropy,
                b_vocab,
                b_observed,
                b_mean,
                b_second,
                b_covariance,
                observed_tensor,
                mean_tensor,
                covariance_tensor,
            )

        observed_parts.append(observed.detach().cpu().numpy().astype(np.float32))
        fast_mc_numerator += float(torch.sum(observed - mc_mean).item())
        fast_mc_variance += float(torch.sum(mc_var).item())
        fast_exact_numerator += float(torch.sum(observed - exact_mean).item())
        fast_exact_variance += float(torch.sum(exact_var).item())
        ada_u += torch.sum(phi_observed - phi_mean, dim=0).double().cpu()
        ada_v += torch.sum(phi_covariance, dim=0).double().cpu()

        del (
            chunk_logits,
            chunk_targets,
            log_probs,
            probs,
            observed,
            exact_mean,
            exact_var,
            sampled_ids,
            sampled_log_probs,
            mc_mean,
            mc_var,
            clamped_vocab,
            phi_vocab,
            phi_observed,
            phi_mean,
            phi_second,
            phi_covariance,
            entropy,
            entropy_ranks,
            entropy_ecdf,
        )

    ada_v = 0.5 * (ada_v + ada_v.T)
    for name in water_v:
        water_v[name] = 0.5 * (water_v[name] + water_v[name].T)
    return DocumentStatistics(
        z_observed=np.concatenate(observed_parts),
        fast_mc_numerator=fast_mc_numerator,
        fast_mc_variance=max(fast_mc_variance, 0.0),
        fast_exact_numerator=fast_exact_numerator,
        fast_exact_variance=max(fast_exact_variance, 0.0),
        ada_u=ada_u.numpy(),
        ada_v=ada_v.numpy(),
        water_u={name: value.numpy() for name, value in water_u.items()},
        water_v={name: value.numpy() for name, value in water_v.items()},
    )


def score_documents(
    model,
    tokenizer,
    texts: list[str],
    *,
    spec: AdaSplineSpec,
    knots: np.ndarray,
    water_reference: TorchWaterReference,
    max_length: int,
    skip_prefix_tokens: int,
    fast_samples: int,
    token_chunk_size: int,
    sample_seeds: list[int],
) -> list[DocumentStatistics]:
    import torch

    if len(texts) != len(sample_seeds):
        raise ValueError("texts and sample_seeds must have the same length")
    if not texts:
        return []
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(model_input_device(model))
    attention_mask = encoded["attention_mask"].to(model_input_device(model))
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        if water_reference.sorted_human_entropy.device != outputs.logits.device:
            water_reference = TorchWaterReference(
                specs=tuple(
                    TorchWaterSpec(
                        item.spec,
                        item.z_knots.to(outputs.logits.device),
                        item.h_knots.to(outputs.logits.device),
                    )
                    for item in water_reference.specs
                ),
                sorted_human_entropy=water_reference.sorted_human_entropy.to(
                    outputs.logits.device
                ),
            )
        results: list[DocumentStatistics] = []
        for index in range(len(texts)):
            length = int(attention_mask[index].sum().item())
            if length < 2:
                raise ValueError("text tokenized to fewer than two tokens")
            # Match the legacy 128-cache forward exactly: slice one document
            # out of the padded batch, cast the complete document logits to
            # float32, then compute its observed log probabilities.  With
            # BF16 models, changing the forward batch size can otherwise
            # change the selected GEMM/attention kernel enough to make a
            # strict observed-log-probability comparison fail even when the
            # token IDs are unchanged.
            logits = outputs.logits[index, : length - 1, :].float()
            targets = input_ids[index, 1:length].to(logits.device)
            start = max(0, skip_prefix_tokens - 1)
            if start >= logits.shape[0]:
                raise ValueError(
                    f"skip_prefix_tokens={skip_prefix_tokens} removes all "
                    f"{length} tokens"
                )
            logits = logits[start:]
            targets = targets[start:]
            full_log_probs = torch.log_softmax(logits, dim=-1)
            observed = full_log_probs.gather(-1, targets[:, None])[:, 0]
            del full_log_probs
            results.append(
                conditional_statistics(
                    logits,
                    targets,
                    observed_override=observed,
                    spec=spec,
                    knots=knots,
                    water_reference=water_reference,
                    fast_samples=fast_samples,
                    token_chunk_size=token_chunk_size,
                    sample_seed=sample_seeds[index],
                )
            )
            del logits, targets, observed
    del outputs, input_ids, attention_mask, encoded
    return results


def validate_split(cache: ReferenceSplitCache, split: str, expected: int) -> None:
    if cache.num_documents != expected:
        raise AssertionError(f"{split}: expected {expected} docs, got {cache.num_documents}")
    arrays = [
        cache.z_observed,
        cache.fast_mc_numerator,
        cache.fast_mc_variance,
        cache.fast_exact_numerator,
        cache.fast_exact_variance,
        cache.ada_u,
        cache.ada_v,
        *cache.water_u.values(),
        *cache.water_v.values(),
    ]
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise FloatingPointError(f"{split}: non-finite value in reference cache")
    if np.any(cache.fast_mc_variance < 0) or np.any(cache.fast_exact_variance < 0):
        raise FloatingPointError(f"{split}: negative Fast variance")
    if np.max(np.abs(cache.ada_v - np.swapaxes(cache.ada_v, 1, 2))) > 1e-8:
        raise FloatingPointError(f"{split}: asymmetric Ada covariance")
    if set(cache.water_u) != set(cache.water_v) or not cache.water_u:
        raise ValueError(f"{split}: incomplete Water-Full structures")
    for name, covariance in cache.water_v.items():
        if np.max(np.abs(covariance - np.swapaxes(covariance, 1, 2))) > 1e-8:
            raise FloatingPointError(f"{split}: asymmetric Water covariance {name}")


def water_reference_from_legacy(
    legacy_cell_dir: Path,
    args: argparse.Namespace,
) -> WaterBasisReference:
    path = legacy_cell_dir / "human_tokens.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Water-Full development basis requires the frozen human token cache: {path}"
        )
    with np.load(path, allow_pickle=False) as data:
        z_observed = np.asarray(data["z_observed"], dtype=np.float64)
        entropy = np.asarray(data["entropy"], dtype=np.float64)
        offsets = np.asarray(data["offsets"], dtype=np.int64)
    if len(offsets) < args.expected_pairs + 1:
        raise ValueError(
            f"{path}: only {len(offsets) - 1} human documents; "
            f"Water basis requires all {args.expected_pairs} development documents"
        )
    token_stop = int(offsets[args.expected_pairs])
    z_observed = z_observed[:token_stop]
    entropy = entropy[:token_stop]
    if not np.all(np.isfinite(entropy)):
        raise ValueError(f"{path}: non-finite entropy values")
    specs = build_water_specs(
        z_observed,
        parse_structures(args.water_structures),
        degree_z=args.water_degree_z,
        degree_h=args.water_degree_h,
        z_lower_quantile=args.water_z_lower_quantile,
        z_upper_bound=args.water_z_upper_bound,
        entropy_clip_low=args.water_entropy_clip_low,
        entropy_clip_high=args.water_entropy_clip_high,
    )
    return WaterBasisReference(
        specs=specs,
        sorted_human_entropy=np.sort(entropy).astype(np.float32),
        provenance={
            "kind": "frozen_development_human_token_cache",
            "path": str(path),
            "documents": args.expected_pairs,
            "tokens": token_stop,
            "candidate_samples_used_for_reference": 0,
            "basis_configuration": {
                "degree_z": args.water_degree_z,
                "degree_h": args.water_degree_h,
                "z_lower_quantile": args.water_z_lower_quantile,
                "z_upper_bound": args.water_z_upper_bound,
                "entropy_clip_low": args.water_entropy_clip_low,
                "entropy_clip_high": args.water_entropy_clip_high,
            },
            "note": (
                "Only observed z and exact source entropy define the Water basis; "
                "the legacy 128 candidate samples are not used by Water-Full."
            ),
        },
    )


def resolve_water_reference(
    output_dir: Path,
    legacy_cell_dir: Path | None,
    args: argparse.Namespace,
) -> WaterBasisReference:
    requested = parse_structures(args.water_structures)
    if args.water_reference_root is None:
        if legacy_cell_dir is None:
            raise ValueError(
                "Water-Full development build requires --legacy-cache-root"
            )
        reference = water_reference_from_legacy(legacy_cell_dir, args)
    else:
        source = args.water_reference_root / output_dir.parent.name / output_dir.name
        reference = load_water_reference(source / "water_reference.npz")
        actual = [(spec.j_bases, spec.k_bases) for spec in reference.specs]
        if actual != requested:
            raise ValueError(
                f"{source}: frozen Water structures {actual} != requested {requested}"
            )
        requested_configuration = {
            "degree_z": args.water_degree_z,
            "degree_h": args.water_degree_h,
            "z_lower_quantile": args.water_z_lower_quantile,
            "z_upper_bound": args.water_z_upper_bound,
            "entropy_clip_low": args.water_entropy_clip_low,
            "entropy_clip_high": args.water_entropy_clip_high,
        }
        frozen_configuration = reference.provenance.get("basis_configuration")
        if frozen_configuration != requested_configuration:
            raise ValueError(
                f"{source}: frozen Water basis configuration "
                f"{frozen_configuration} != requested {requested_configuration}"
            )
        reference = WaterBasisReference(
            reference.specs,
            reference.sorted_human_entropy,
            {
                **reference.provenance,
                "loaded_from_development_reference": str(
                    source / "water_reference.npz"
                ),
            },
        )
    save_water_reference(output_dir / "water_reference.npz", reference)
    return reference


def validate_against_legacy_cache(
    cache: ReferenceSplitCache,
    legacy_path: Path,
    *,
    split: str,
    observed_atol: float,
    observed_policy: str,
) -> dict[str, Any]:
    """Validate structural compatibility and diagnose numerical drift.

    Token offsets are a structural invariant and must match.  Observed log
    probabilities are model-forward outputs, not token-stream identifiers.
    Across PyTorch/Transformers/CUDA versions, attention backends, or device
    maps, BF16 outputs need not agree at a sub-millithreshold tolerance even
    for identical inputs.  Their comparison is therefore warning-only unless
    ``observed_policy`` is explicitly ``strict``.
    """
    if not legacy_path.exists():
        raise FileNotFoundError(legacy_path)
    with np.load(legacy_path, allow_pickle=False) as legacy:
        legacy_offsets = np.asarray(legacy["offsets"], dtype=np.int64)
        legacy_observed = np.asarray(legacy["z_observed"], dtype=np.float64)
    needed_offsets = cache.num_documents + 1
    if len(legacy_offsets) < needed_offsets:
        raise ValueError(
            f"{legacy_path}: only {len(legacy_offsets) - 1} documents, "
            f"need {cache.num_documents}"
        )
    expected_offsets = legacy_offsets[:needed_offsets]
    if not np.array_equal(cache.offsets, expected_offsets):
        raise ValueError(
            f"{split}: token offsets differ from legacy cache {legacy_path}; "
            "check tokenizer, max_length, and skip_prefix_tokens"
        )
    token_count = int(cache.offsets[-1])
    legacy_prefix = legacy_observed[:token_count]
    differences = np.abs(
        np.asarray(cache.z_observed, dtype=np.float64) - legacy_prefix
    )
    maximum = float(np.max(differences))
    median = float(np.median(differences))
    q99 = float(np.quantile(differences, 0.99))
    exceeding = int(np.count_nonzero(differences > observed_atol))
    observed_close = bool(np.allclose(
        np.asarray(cache.z_observed, dtype=np.float64),
        legacy_prefix,
        rtol=0.0,
        atol=observed_atol,
    ))
    diagnostics = {
        "token_count": token_count,
        "offsets_match": True,
        "observed_allclose": observed_close,
        "observed_atol": observed_atol,
        "max_abs_difference": maximum,
        "q99_abs_difference": q99,
        "median_abs_difference": median,
        "values_over_atol": exceeding,
        "observed_policy": observed_policy,
    }
    if not observed_close:
        message = (
            f"{split}: observed log probabilities differ from {legacy_path}; "
            f"max_abs_difference={maximum:.6g}, q99_abs_difference={q99:.6g}, "
            f"median_abs_difference={median:.6g}, "
            f"values_over_atol={exceeding}/{token_count}, "
            f"atol={observed_atol:.6g}. Token offsets match. This comparison "
            "measures model-forward numerical reproducibility, not token "
            "identity; BF16 values may drift across runtime, attention backend, "
            "or device placement."
        )
        if observed_policy == "strict":
            raise ValueError(message)
        print(f"  WARNING {message}")
        print(
            f"  VERIFIED {split}: {token_count} token offsets match legacy; "
            "observed-log-probability drift recorded in metadata"
        )
    else:
        print(
            f"  VERIFIED {split}: offsets and {token_count} observed "
            f"log-probabilities match {legacy_path}"
        )
    return diagnostics


def compare_legacy_metadata(
    legacy_cell_dir: Path,
    *,
    generated_path: Path,
    processed_pairs: int,
    max_length: int,
    skip_prefix_tokens: int,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    """Check available protocol metadata and report runtime changes.

    Older cache builders did not always write every metadata field. A missing
    value (or the historical ``-1`` sentinel) means "unknown", not a protocol
    mismatch. Structural compatibility is still enforced later by comparing
    the requested prefix of every split's token offsets.
    """
    metadata_path = legacy_cell_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"  WARNING legacy metadata is missing: {metadata_path}")
        return {"available": False, "path": str(metadata_path)}
    with metadata_path.open("r", encoding="utf-8") as stream:
        legacy = json.load(stream)

    def optional_nonnegative_int(key: str) -> int | None:
        raw = legacy.get(key)
        if raw is None:
            return None
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{metadata_path}: invalid integer metadata {key}={raw!r}"
            ) from exc
        return None if value < 0 else value

    mismatches: list[str] = []
    unknown_fields: list[str] = []
    legacy_max_length = optional_nonnegative_int("max_length")
    if legacy_max_length is None:
        unknown_fields.append("max_length")
    elif legacy_max_length != max_length:
        mismatches.append(
            f"max_length legacy={legacy_max_length} current={max_length}"
        )
    legacy_skip_prefix = optional_nonnegative_int("skip_prefix_tokens")
    if legacy_skip_prefix is None:
        unknown_fields.append("skip_prefix_tokens")
    elif legacy_skip_prefix != skip_prefix_tokens:
        mismatches.append(
            "skip_prefix_tokens "
            f"legacy={legacy_skip_prefix} "
            f"current={skip_prefix_tokens}"
        )
    legacy_pairs = optional_nonnegative_int("processed_pairs")
    if legacy_pairs is None:
        unknown_fields.append("processed_pairs")
    elif legacy_pairs < processed_pairs:
        mismatches.append(
            f"processed_pairs legacy={legacy_pairs} current={processed_pairs}"
        )
    if mismatches:
        raise ValueError(
            f"{metadata_path}: protocol metadata is incompatible: "
            + "; ".join(mismatches)
        )
    if unknown_fields:
        print(
            "  NOTE legacy metadata does not record "
            + ", ".join(unknown_fields)
            + "; compatibility will be established by strict token-offset "
            "prefix checks for each split"
        )

    legacy_runtime = legacy.get("runtime", {})
    aliases = {
        "requested_dtype": ("requested_dtype", "requested_torch_dtype"),
        "loaded_dtype": ("loaded_dtype", "loaded_torch_dtype"),
    }

    def lookup(value: dict[str, Any], key: str):
        for candidate in aliases.get(key, (key,)):
            if candidate in value:
                return value[candidate]
        return None

    runtime_keys = [
        "torch_version",
        "transformers_version",
        "model_path",
        "model_class",
        "model_module",
        "tokenizer_class",
        "requested_dtype",
        "loaded_dtype",
        "device_map",
    ]
    runtime_differences = {
        key: {
            "legacy": lookup(legacy_runtime, key),
            "current": lookup(runtime, key),
        }
        for key in runtime_keys
        if lookup(legacy_runtime, key) != lookup(runtime, key)
    }
    if runtime_differences:
        print(
            "  NOTE legacy/current runtime metadata differs: "
            + ", ".join(runtime_differences)
        )
    return {
        "available": True,
        "path": str(metadata_path),
        "legacy_generated_path": legacy.get("generated_path"),
        "current_generated_path": str(generated_path),
        "legacy_processed_pairs": legacy_pairs,
        "current_processed_pairs": processed_pairs,
        "metadata_fields_unknown": unknown_fields,
        "legacy_contains_requested_prefix": (
            None if legacy_pairs is None else legacy_pairs >= processed_pairs
        ),
        "protocol_fields_compatible": True,
        "runtime_differences": runtime_differences,
    }


def build_cell(
    generated_path: Path,
    output_dir: Path,
    model,
    tokenizer,
    runtime: dict[str, Any],
    legacy_cell_dir: Path | None,
    args: argparse.Namespace,
) -> None:
    output_paths = {
        "human": output_dir / "human_reference.npz",
        "machine": output_dir / "machine_reference.npz",
        "calibration": output_dir / "calibration_reference.npz",
    }
    metadata_path = output_dir / "metadata.json"
    water_reference_path = output_dir / "water_reference.npz"
    existing_metadata: dict[str, Any] = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as stream:
            existing_metadata = json.load(stream)
    requested_structures = [
        f"J{j_bases}_K{k_bases}"
        for j_bases, k_bases in parse_structures(args.water_structures)
    ]
    requested_basis_configuration = {
        "degree_z": args.water_degree_z,
        "degree_h": args.water_degree_h,
        "z_lower_quantile": args.water_z_lower_quantile,
        "z_upper_bound": args.water_z_upper_bound,
        "entropy_clip_low": args.water_entropy_clip_low,
        "entropy_clip_high": args.water_entropy_clip_high,
    }
    complete_v5 = (
        all(path.exists() for path in output_paths.values())
        and metadata_path.exists()
        and water_reference_path.exists()
        and int(existing_metadata.get("cache_schema_version", 0))
        >= CACHE_SCHEMA_VERSION
        and bool(existing_metadata.get("water_full", {}).get("enabled", False))
        and existing_metadata.get("builder_version") == PACKAGE_VERSION
        and int(existing_metadata.get("processed_pairs", -1))
        == (args.limit_pairs or args.expected_pairs)
        and int(existing_metadata.get("fast_mc_samples", -1)) == args.fast_samples
        and int(existing_metadata.get("max_length", -1)) == args.max_length
        and int(existing_metadata.get("skip_prefix_tokens", -1))
        == args.skip_prefix_tokens
        and int(existing_metadata.get("forward_batch_size", -1)) == args.batch_size
        and existing_metadata.get("water_full", {}).get("structures")
        == requested_structures
        and existing_metadata.get("water_full", {}).get("basis_configuration")
        == requested_basis_configuration
    )
    if complete_v5 and not args.overwrite:
        print(f"SKIP existing {output_dir}")
        return
    if existing_metadata and not args.overwrite:
        print(
            f"REBUILD {output_dir}: existing cache predates Water-Full schema "
            f"{CACHE_SCHEMA_VERSION}"
        )

    all_rows = read_jsonl(generated_path)
    if len(all_rows) != args.expected_pairs:
        raise ValueError(
            f"{generated_path}: expected {args.expected_pairs} rows, found {len(all_rows)}"
        )
    rows = all_rows if args.limit_pairs is None else all_rows[: args.limit_pairs]
    fields = {
        "human": "human_text",
        "machine": "machine_text",
        "calibration": "calibration_text",
    }
    for row_index, row in enumerate(rows):
        for field in fields.values():
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"{generated_path}: row {row_index} has invalid {field}")

    spec = AdaSplineSpec(
        start=args.spline_start,
        end=args.spline_end,
        n_bases=args.n_bases,
        degree=args.spline_degree,
        intercept=not args.no_intercept,
    )
    knots = ada_knots(spec)
    output_dir.mkdir(parents=True, exist_ok=True)
    water_basis = resolve_water_reference(output_dir, legacy_cell_dir, args)
    torch_water_basis = prepare_torch_water_reference(
        water_basis, model_input_device(model)
    )
    print(
        "  WATER-FULL "
        + ",".join(spec.name for spec in water_basis.specs)
        + f"; entropy reference tokens={len(water_basis.sorted_human_entropy)}"
    )
    token_counts: dict[str, int] = {}
    legacy_metadata = None
    if legacy_cell_dir is not None:
        legacy_metadata = compare_legacy_metadata(
            legacy_cell_dir,
            generated_path=generated_path,
            processed_pairs=len(rows),
            max_length=args.max_length,
            skip_prefix_tokens=args.skip_prefix_tokens,
            runtime=runtime,
        )
    legacy_split_diagnostics: dict[str, Any] = {}
    for split_index, (split, field) in enumerate(fields.items()):
        documents: list[DocumentStatistics] = []
        for batch_start in range(0, len(rows), args.batch_size):
            batch_stop = min(len(rows), batch_start + args.batch_size)
            batch_rows = rows[batch_start:batch_stop]
            documents.extend(
                score_documents(
                    model,
                    tokenizer,
                    [row[field] for row in batch_rows],
                    spec=spec,
                    knots=knots,
                    water_reference=torch_water_basis,
                    max_length=args.max_length,
                    skip_prefix_tokens=args.skip_prefix_tokens,
                    fast_samples=args.fast_samples,
                    token_chunk_size=args.token_chunk_size,
                    sample_seeds=[
                        stable_document_seed(args.seed, split_index, document_index)
                        for document_index in range(batch_start, batch_stop)
                    ],
                )
            )
            completed = batch_stop
            if completed == len(rows) or completed % args.log_every == 0:
                print(
                    f"  {generated_path.parent.name}/{generated_path.stem} "
                    f"{split}: {completed}/{len(rows)}"
                )
            if args.empty_cache_every and completed % args.empty_cache_every == 0:
                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        cache = ReferenceSplitCache.from_documents(documents)
        validate_split(cache, split, len(rows))
        if legacy_cell_dir is not None:
            legacy_filename = {
                "human": "human_tokens.npz",
                "machine": "machine_tokens.npz",
                "calibration": "calibration_tokens.npz",
            }[split]
            legacy_split_diagnostics[split] = validate_against_legacy_cache(
                cache,
                legacy_cell_dir / legacy_filename,
                split=split,
                observed_atol=args.observed_atol,
                observed_policy=args.legacy_observed_policy,
            )
        cache.save(output_paths[split])
        token_counts[split] = int(len(cache.z_observed))
        del cache, documents
        gc.collect()

    atomic_write_json(
        metadata_path,
        {
            "schema_version": 2,
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "builder_version": PACKAGE_VERSION,
            "generated_path": str(generated_path),
            "processed_pairs": len(rows),
            "source_expected_pairs": args.expected_pairs,
            "limit_pairs": args.limit_pairs,
            "source_model_equals_sampling_model": True,
            "source_model_equals_scoring_model": True,
            "fast_mc_samples": args.fast_samples,
            "fast_mc_variance": "unbiased sample variance (ddof=1)",
            "fast_exact_moments": "full vocabulary under source distribution",
            "ada_conditional_moments": "full vocabulary under source distribution",
            "ada_spline": asdict(spec),
            "ada_knots": knots.tolist(),
            "water_full": {
                "enabled": True,
                "conditional_moments": "exact full vocabulary under source distribution",
                "feature": "B(z) tensor C(F_H(h))",
                "flatten_order": "C order with entropy-basis index fastest",
                "specs": [asdict(item) for item in water_basis.specs],
                "structures": [item.name for item in water_basis.specs],
                "basis_configuration": requested_basis_configuration,
                "reference_path": str(water_reference_path),
                "reference_provenance": water_basis.provenance,
                "candidate_samples_used": 0,
            },
            "max_length": args.max_length,
            "skip_prefix_tokens": args.skip_prefix_tokens,
            "token_chunk_size": args.token_chunk_size,
            "forward_batch_size": args.batch_size,
            "seed": args.seed,
            "token_counts": token_counts,
            "legacy_offsets_verified": legacy_cell_dir is not None,
            "legacy_observed_verified": (
                None
                if legacy_cell_dir is None
                else all(
                    item["observed_allclose"]
                    for item in legacy_split_diagnostics.values()
                )
            ),
            "legacy_cache_verified": (
                False
                if legacy_cell_dir is None
                else all(
                    item["observed_allclose"]
                    for item in legacy_split_diagnostics.values()
                )
            ),
            "legacy_cache_path": (
                None if legacy_cell_dir is None else str(legacy_cell_dir)
            ),
            "observed_log_probability_atol": args.observed_atol,
            "legacy_observed_policy": args.legacy_observed_policy,
            "legacy_metadata_comparison": legacy_metadata,
            "legacy_split_diagnostics": legacy_split_diagnostics,
            "runtime": runtime,
        },
    )
    print(f"BUILT {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="version", version=PACKAGE_VERSION)
    parser.add_argument(
        "--generated-root", type=Path, default=Path("data/train/text")
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("cache/whitebox/train"),
    )
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--model-paths-json", type=Path)
    parser.add_argument("--model-dtypes-json", type=Path)
    parser.add_argument(
        "--legacy-cache-root",
        type=Path,
        help=(
            "Optional old 128-cache root. Before saving, require identical token "
            "offsets and matching observed log probabilities."
        ),
    )
    parser.add_argument(
        "--water-reference-root",
        type=Path,
        help=(
            "Development Water-Full cache root whose frozen water_reference.npz "
            "is reused for independent test. Omit for development."
        ),
    )
    parser.add_argument("--observed-atol", type=float, default=5e-4)
    parser.add_argument(
        "--legacy-observed-policy",
        choices=("warn", "strict"),
        default="warn",
        help=(
            "How to handle legacy z_observed drift after token offsets match. "
            "The default records a warning because BF16 model outputs are not "
            "bitwise stable across runtimes/device maps. Use strict only when "
            "reproducing the exact legacy software and hardware path."
        ),
    )
    parser.add_argument("--expected-pairs", type=int, default=500)
    parser.add_argument("--limit-pairs", type=int)
    parser.add_argument("--fast-samples", type=int, default=10_000)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--skip-prefix-tokens", type=int, default=0)
    parser.add_argument("--token-chunk-size", type=int, default=8)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help=(
            "Model-forward batch size. Keep at 2 to match the legacy 128-cache "
            "builder's BF16 numerical path."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--spline-start", type=float, default=-32.0)
    parser.add_argument("--spline-end", type=float, default=0.0)
    parser.add_argument("--n-bases", type=int, default=7)
    parser.add_argument("--spline-degree", type=int, default=2)
    parser.add_argument("--no-intercept", action="store_true")
    parser.add_argument("--water-structures", default="4x4,6x6,8x6")
    parser.add_argument("--water-degree-z", type=int, default=2)
    parser.add_argument("--water-degree-h", type=int, default=2)
    parser.add_argument("--water-z-lower-quantile", type=float, default=0.001)
    parser.add_argument("--water-z-upper-bound", type=float, default=0.0)
    parser.add_argument("--water-entropy-clip-low", type=float, default=0.005)
    parser.add_argument("--water-entropy-clip-high", type=float, default=0.995)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--empty-cache-every", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    print(f"whitebox-reference-builder {PACKAGE_VERSION}")
    if args.expected_pairs <= 0:
        parser.error("--expected-pairs must be positive")
    if args.limit_pairs is not None and not 0 < args.limit_pairs <= args.expected_pairs:
        parser.error("--limit-pairs must be in [1, expected-pairs]")
    if args.fast_samples < 2:
        parser.error("--fast-samples must be at least 2")
    if args.token_chunk_size <= 0:
        parser.error("--token-chunk-size must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    try:
        structures = parse_structures(args.water_structures)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    if not structures:
        parser.error("--water-structures cannot be empty")
    if not 0.0 <= args.water_z_lower_quantile < 0.5:
        parser.error("--water-z-lower-quantile must be in [0, 0.5)")
    if not (
        0.0
        <= args.water_entropy_clip_low
        < args.water_entropy_clip_high
        <= 1.0
    ):
        parser.error("Water entropy clipping must satisfy 0 <= low < high <= 1")
    if args.log_every <= 0:
        parser.error("--log-every must be positive")
    if args.observed_atol < 0:
        parser.error("--observed-atol must be non-negative")

    model_paths = load_mapping(args.model_paths_json, DEFAULT_MODEL_PATHS)
    model_dtypes = load_mapping(args.model_dtypes_json, DEFAULT_DTYPES)
    models = parse_csv(args.models)
    datasets = parse_csv(args.datasets)
    for model_name in models:
        if model_name not in model_paths or model_name not in model_dtypes:
            raise KeyError(f"missing path/dtype mapping for {model_name}")
        print(f"LOAD {model_name}: {model_paths[model_name]} ({model_dtypes[model_name]})")
        model, tokenizer, runtime = load_model(
            model_name, model_paths[model_name], model_dtypes[model_name]
        )
        for dataset in datasets:
            generated_path = args.generated_root / model_name / f"{dataset}.jsonl"
            if not generated_path.exists():
                raise FileNotFoundError(generated_path)
            build_cell(
                generated_path,
                args.cache_root / model_name / dataset,
                model,
                tokenizer,
                runtime,
                (
                    None
                    if args.legacy_cache_root is None
                    else args.legacy_cache_root / model_name / dataset
                ),
                args,
            )
        del model, tokenizer
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
