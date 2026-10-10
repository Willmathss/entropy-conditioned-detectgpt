#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import SplineTransformer


def parse_csv(value: str):
    return [x.strip() for x in value.split(',') if x.strip()]


def parse_float_csv(value: str):
    return [float(x.strip()) for x in value.split(',') if x.strip()]


def parse_structures(value: str):
    result = []
    for item in parse_csv(value):
        parts = item.lower().split('x')
        if len(parts) != 2:
            raise ValueError(f'Invalid structure {item!r}; use forms such as 4x4,6x6')
        j, k = int(parts[0]), int(parts[1])
        if j < 3 or k < 3:
            raise ValueError('Each spline direction needs at least 3 bases')
        result.append((j, k))
    return result


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_write_json(path: Path, value: Mapping[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def read_jsonl(path: Path):
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{line_no}: invalid JSON: {exc}') from exc
            if not isinstance(row, dict):
                raise ValueError(f'{path}:{line_no}: expected JSON object')
            rows.append(row)
    return rows


@dataclass
class RaggedTokenCache:
    offsets: np.ndarray
    z_observed: np.ndarray
    entropy: np.ndarray
    z_sampled: np.ndarray
    num_samples: int

    @property
    def num_documents(self):
        return int(len(self.offsets) - 1)

    @property
    def total_tokens(self):
        return int(self.offsets[-1])

    def document(self, index: int):
        a = int(self.offsets[index])
        b = int(self.offsets[index + 1])
        return self.z_observed[a:b], self.entropy[a:b], self.z_sampled[a:b]

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + '.tmp.npz')
        np.savez(
            tmp,
            offsets=self.offsets.astype(np.int64),
            z_observed=self.z_observed.astype(np.float32),
            entropy=self.entropy.astype(np.float32),
            z_sampled=self.z_sampled.astype(np.float16),
            num_samples=np.asarray([self.num_samples], dtype=np.int64),
        )
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as d:
            return cls(
                offsets=np.asarray(d['offsets'], dtype=np.int64),
                z_observed=np.asarray(d['z_observed'], dtype=np.float32),
                entropy=np.asarray(d['entropy'], dtype=np.float32),
                z_sampled=np.asarray(d['z_sampled'], dtype=np.float32),
                num_samples=int(np.asarray(d['num_samples'])[0]),
            )


def pack_documents(documents, num_samples: int):
    offsets = [0]
    z_parts, h_parts, s_parts = [], [], []
    for z, h, s in documents:
        z = np.asarray(z, dtype=np.float32)
        h = np.asarray(h, dtype=np.float32)
        s = np.asarray(s, dtype=np.float32)
        if z.ndim != 1 or h.ndim != 1 or len(z) != len(h):
            raise ValueError('Invalid observed z/entropy arrays')
        if s.shape != (len(z), num_samples):
            raise ValueError(f'Expected sampled-z shape {(len(z), num_samples)}, found {s.shape}')
        if len(z) == 0:
            raise ValueError('Document has zero scored tokens')
        if not (np.isfinite(z).all() and np.isfinite(h).all() and np.isfinite(s).all()):
            raise ValueError('Non-finite token statistic')
        z_parts.append(z); h_parts.append(h); s_parts.append(s)
        offsets.append(offsets[-1] + len(z))
    return RaggedTokenCache(
        offsets=np.asarray(offsets, dtype=np.int64),
        z_observed=np.concatenate(z_parts),
        entropy=np.concatenate(h_parts),
        z_sampled=np.concatenate(s_parts),
        num_samples=num_samples,
    )


def ecdf_transform(values, sorted_reference, low=0.005, high=0.995):
    values = np.asarray(values, dtype=np.float64)
    ref = np.asarray(sorted_reference, dtype=np.float64)
    ranks = np.searchsorted(ref, values, side='right')
    u = (ranks + 0.5) / (len(ref) + 1.0)
    return np.clip(u, low, high)


def strictly_increasing(values, gap=1e-6):
    values = np.asarray(values, dtype=np.float64).copy()
    for i in range(1, len(values)):
        if values[i] <= values[i - 1]:
            values[i] = values[i - 1] + gap
    return values


@dataclass(frozen=True)
class BasisSpec:
    j_bases: int
    k_bases: int
    degree_z: int
    degree_h: int
    z_base_knots: Tuple[float, ...]
    h_base_knots: Tuple[float, ...]
    entropy_clip_low: float
    entropy_clip_high: float

    @property
    def dimension(self):
        return self.j_bases * self.k_bases

    @property
    def name(self):
        return f'J{self.j_bases}_K{self.k_bases}'


def make_spline(base_knots, degree: int, expected_bases: int):
    knots = strictly_increasing(base_knots)
    transformer = SplineTransformer(
        degree=degree,
        knots=knots[:, None],
        include_bias=True,
        extrapolation='constant',
        sparse_output=False,
    )
    transformer.fit(np.asarray([[knots[0]], [knots[-1]]], dtype=np.float64))
    if int(transformer.n_features_out_) != expected_bases:
        raise ValueError(
            f'SplineTransformer returned {transformer.n_features_out_} bases, '
            f'expected {expected_bases}'
        )
    return transformer


def build_basis_spec(human_cache: RaggedTokenCache, j: int, k: int, degree_z=2, degree_h=2,
                     z_lower_quantile=0.001, z_upper_bound=0.0,
                     entropy_clip_low=0.005, entropy_clip_high=0.995):
    if j <= degree_z or k <= degree_h:
        raise ValueError('basis count must exceed spline degree')
    z = np.asarray(human_cache.z_observed, dtype=np.float64)
    n_z_knots = j - degree_z + 1
    levels = np.linspace(z_lower_quantile, 1.0 - z_lower_quantile, n_z_knots)
    z_knots = np.quantile(z, levels)
    z_knots[0] = np.quantile(z, z_lower_quantile)
    z_knots[-1] = z_upper_bound
    z_knots = strictly_increasing(z_knots)
    n_h_knots = k - degree_h + 1
    h_knots = np.linspace(0.0, 1.0, n_h_knots)
    return BasisSpec(
        j_bases=j,
        k_bases=k,
        degree_z=degree_z,
        degree_h=degree_h,
        z_base_knots=tuple(float(x) for x in z_knots),
        h_base_knots=tuple(float(x) for x in h_knots),
        entropy_clip_low=entropy_clip_low,
        entropy_clip_high=entropy_clip_high,
    )


def instantiate_basis(spec: BasisSpec):
    return (
        make_spline(np.asarray(spec.z_base_knots), spec.degree_z, spec.j_bases),
        make_spline(np.asarray(spec.h_base_knots), spec.degree_h, spec.k_bases),
    )


def linear_z_initialization(spec: BasisSpec, z_transformer, grid_size=4096):
    z = np.linspace(spec.z_base_knots[0], spec.z_base_knots[-1], grid_size)
    design = z_transformer.transform(z[:, None])
    beta, *_ = np.linalg.lstsq(design, z, rcond=None)
    max_error = float(np.max(np.abs(design @ beta - z)))
    if max_error > 1e-4:
        print(f'WARNING: linear-z spline initialization max error={max_error:.3e}')
    return np.repeat(beta[:, None], spec.k_bases, axis=1).astype(np.float64)


@dataclass
class UVFeatures:
    u_human: np.ndarray
    v_human: np.ndarray
    u_machine: np.ndarray
    v_machine: np.ndarray
    u_calibration: np.ndarray
    v_calibration: np.ndarray

    def save(self, path: Path, spec: BasisSpec):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + '.tmp.npz')
        np.savez(
            tmp,
            u_human=self.u_human.astype(np.float32),
            v_human=self.v_human.astype(np.float32),
            u_machine=self.u_machine.astype(np.float32),
            v_machine=self.v_machine.astype(np.float32),
            u_calibration=self.u_calibration.astype(np.float32),
            v_calibration=self.v_calibration.astype(np.float32),
            spec_json=np.asarray([json.dumps(asdict(spec), sort_keys=True)]),
        )
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as d:
            features = cls(
                u_human=np.asarray(d['u_human'], dtype=np.float64),
                v_human=np.asarray(d['v_human'], dtype=np.float64),
                u_machine=np.asarray(d['u_machine'], dtype=np.float64),
                v_machine=np.asarray(d['v_machine'], dtype=np.float64),
                u_calibration=np.asarray(d['u_calibration'], dtype=np.float64),
                v_calibration=np.asarray(d['v_calibration'], dtype=np.float64),
            )
            spec_dict = json.loads(str(np.asarray(d['spec_json'])[0]))
        spec_dict['z_base_knots'] = tuple(spec_dict['z_base_knots'])
        spec_dict['h_base_knots'] = tuple(spec_dict['h_base_knots'])
        return features, BasisSpec(**spec_dict)

def document_uv(z_observed, entropy, z_sampled, sorted_human_entropy,
                spec: BasisSpec, z_transformer, h_transformer):
    z_observed = np.asarray(z_observed, dtype=np.float64)
    entropy = np.asarray(entropy, dtype=np.float64)
    z_sampled = np.asarray(z_sampled, dtype=np.float64)
    token_count, sample_count = z_sampled.shape
    if sample_count < 2:
        raise ValueError('At least two sampled candidate tokens are required')

    u_h = ecdf_transform(
        entropy,
        sorted_human_entropy,
        low=spec.entropy_clip_low,
        high=spec.entropy_clip_high,
    )
    b_obs = z_transformer.transform(z_observed[:, None])
    b_samp = z_transformer.transform(z_sampled.reshape(-1, 1)).reshape(
        token_count, sample_count, spec.j_bases
    )
    c_h = h_transformer.transform(u_h[:, None])

    obs_phi = (b_obs[:, :, None] * c_h[:, None, :]).reshape(
        token_count, spec.dimension
    )
    sampled_phi = (b_samp[:, :, :, None] * c_h[:, None, None, :]).reshape(
        token_count, sample_count, spec.dimension
    )
    mean_phi = sampled_phi.mean(axis=1)
    u_doc = (obs_phi - mean_phi).sum(axis=0)
    centered = sampled_phi - mean_phi[:, None, :]
    v_doc = np.einsum('tsd,tse->de', centered, centered, optimize=True)
    v_doc /= float(sample_count - 1)
    v_doc = 0.5 * (v_doc + v_doc.T)
    return u_doc.astype(np.float64), v_doc.astype(np.float64)


def transform_cache(cache: RaggedTokenCache, sorted_human_entropy, spec: BasisSpec,
                    z_transformer, h_transformer, split_name='split'):
    d = spec.dimension
    u = np.empty((cache.num_documents, d), dtype=np.float64)
    v = np.empty((cache.num_documents, d, d), dtype=np.float64)
    for i in range(cache.num_documents):
        z, h, s = cache.document(i)
        u[i], v[i] = document_uv(
            z, h, s, sorted_human_entropy, spec, z_transformer, h_transformer
        )
        if i + 1 == cache.num_documents or (i + 1) % 50 == 0:
            print(f'    UV {spec.name} {split_name}: {i + 1}/{cache.num_documents}')
    return u, v


def build_or_load_uv(cell_cache_dir: Path, feature_dir: Path, spec: BasisSpec,
                     sorted_human_entropy, overwrite=False):
    path = feature_dir / f'{spec.name}.npz'
    if path.exists() and not overwrite:
        features, stored_spec = UVFeatures.load(path)
        if stored_spec != spec:
            raise ValueError(f'Feature spec mismatch in {path}')
        print(f'  LOAD UV {path}')
        return features

    human = RaggedTokenCache.load(cell_cache_dir / 'human_tokens.npz')
    machine = RaggedTokenCache.load(cell_cache_dir / 'machine_tokens.npz')
    calibration = RaggedTokenCache.load(cell_cache_dir / 'calibration_tokens.npz')
    z_transformer, h_transformer = instantiate_basis(spec)
    print(f'  BUILD UV {spec.name}, dim={spec.dimension}')
    uh, vh = transform_cache(human, sorted_human_entropy, spec, z_transformer, h_transformer, 'human')
    um, vm = transform_cache(machine, sorted_human_entropy, spec, z_transformer, h_transformer, 'machine')
    uc, vc = transform_cache(calibration, sorted_human_entropy, spec, z_transformer, h_transformer, 'calibration')
    features = UVFeatures(uh, vh, um, vm, uc, vc)
    features.save(path, spec)
    return features


def second_difference_penalty(theta_matrix: torch.Tensor):
    result = torch.zeros((), dtype=theta_matrix.dtype, device=theta_matrix.device)
    if theta_matrix.shape[0] >= 3:
        dz = theta_matrix[2:, :] - 2 * theta_matrix[1:-1, :] + theta_matrix[:-2, :]
        result = result + dz.square().mean()
    if theta_matrix.shape[1] >= 3:
        dh = theta_matrix[:, 2:] - 2 * theta_matrix[:, 1:-1] + theta_matrix[:, :-2]
        result = result + dh.square().mean()
    return result


def scores_torch(u, v, theta, epsilon=1e-8):
    numerator = u @ theta
    denominator2 = torch.einsum('d,nde,e->n', theta, v, theta)
    return numerator / torch.sqrt(torch.clamp(denominator2, min=epsilon))


def soft_upper_quantile(scores, alpha=0.05, eta=0.1, iterations=12):
    q = torch.quantile(scores.detach(), 1.0 - alpha).to(scores.dtype)
    low = scores.min().detach() - 10 * eta
    high = scores.max().detach() + 10 * eta
    for _ in range(iterations):
        p = torch.sigmoid((scores - q) / eta)
        f = p.mean() - alpha
        derivative = -(p * (1 - p)).mean() / eta
        derivative = torch.where(
            derivative.abs() < 1e-8,
            torch.full_like(derivative, -1e-8),
            derivative,
        )
        q = torch.clamp(q - f / derivative, min=low, max=high)
    return q


@torch.no_grad()
def normalize_theta(theta_matrix, mean_v, epsilon=1e-8):
    theta = theta_matrix.reshape(-1)
    norm2 = torch.einsum('d,de,e->', theta, mean_v, theta)
    theta_matrix.div_(torch.sqrt(torch.clamp(norm2, min=epsilon)))


@dataclass
class FitResult:
    theta: np.ndarray
    objective: float
    soft_power: float
    threshold: float
    epochs_run: int
    best_epoch: int
    seed: int


def evaluate_training_state(theta_matrix, uh, vh, um, vm, spec, smooth_lambda,
                            train_alpha, tau, eta, epsilon):
    theta = theta_matrix.reshape(-1)
    sh = scores_torch(uh, vh, theta, epsilon)
    sm = scores_torch(um, vm, theta, epsilon)
    threshold = soft_upper_quantile(sh, alpha=train_alpha, eta=eta)
    soft_power = torch.sigmoid((sm - threshold) / tau).mean()
    objective = soft_power - smooth_lambda * second_difference_penalty(theta_matrix)
    return objective, soft_power, threshold


def fit_theta_once(features: UVFeatures, train_idx, theta_initial, spec: BasisSpec,
                   smooth_lambda: float, *, train_alpha=0.05, tau=0.1, eta=0.1,
                   learning_rate=0.03, epochs=200, patience=40, epsilon=1e-8,
                   seed=0, device='cpu', noise_scale=0.0):
    set_seed(seed)
    device = torch.device(device)
    dtype = torch.float64
    idx = np.asarray(train_idx, dtype=np.int64)
    uh = torch.as_tensor(features.u_human[idx], dtype=dtype, device=device)
    vh = torch.as_tensor(features.v_human[idx], dtype=dtype, device=device)
    um = torch.as_tensor(features.u_machine[idx], dtype=dtype, device=device)
    vm = torch.as_tensor(features.v_machine[idx], dtype=dtype, device=device)

    initial = np.asarray(theta_initial, dtype=np.float64).copy()
    if noise_scale > 0:
        rng = np.random.default_rng(seed)
        initial += noise_scale * rng.standard_normal(initial.shape)
    theta_matrix = torch.nn.Parameter(torch.as_tensor(initial, dtype=dtype, device=device))

    mean_v = torch.cat([vh, vm], dim=0).mean(dim=0)
    mean_v = mean_v + epsilon * torch.eye(mean_v.shape[0], dtype=dtype, device=device)
    normalize_theta(theta_matrix, mean_v, epsilon)
    optimizer = torch.optim.Adam([theta_matrix], lr=learning_rate)

    # Treat the normalized initialization as epoch 0. This both guarantees that
    # optimization cannot return a detector worse than its linear-z starting
    # point on the training objective and gives early stopping a valid baseline.
    with torch.no_grad():
        objective, soft_power, threshold = evaluate_training_state(
            theta_matrix, uh, vh, um, vm, spec, smooth_lambda,
            train_alpha, tau, eta, epsilon,
        )
    best_obj = float(objective.cpu())
    best_theta = theta_matrix.detach().cpu().numpy().copy()
    best_power = float(soft_power.cpu())
    best_threshold = float(threshold.cpu())
    best_epoch = 0
    stale = 0
    epochs_run = 0

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        objective, _, _ = evaluate_training_state(
            theta_matrix, uh, vh, um, vm, spec, smooth_lambda,
            train_alpha, tau, eta, epsilon,
        )
        loss = -objective
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite loss at epoch {epoch}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_([theta_matrix], 10.0)
        optimizer.step()
        normalize_theta(theta_matrix, mean_v, epsilon)
        epochs_run = epoch

        # IMPORTANT: evaluate the objective again after optimizer.step().
        # Previously `value` came from the pre-step theta while `best_theta`
        # stored the post-step theta, so the reported best objective and saved
        # checkpoint could refer to different parameter states.
        with torch.no_grad():
            objective, soft_power, threshold = evaluate_training_state(
                theta_matrix, uh, vh, um, vm, spec, smooth_lambda,
                train_alpha, tau, eta, epsilon,
            )
        value = float(objective.cpu())
        if value > best_obj + 1e-8:
            best_obj = value
            best_theta = theta_matrix.detach().cpu().numpy().copy()
            best_power = float(soft_power.cpu())
            best_threshold = float(threshold.cpu())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if patience > 0 and stale >= patience:
            break

    return FitResult(
        best_theta,
        best_obj,
        best_power,
        best_threshold,
        epochs_run,
        best_epoch,
        seed,
    )


def fit_theta(features: UVFeatures, train_idx, theta_initial, spec: BasisSpec,
              smooth_lambda: float, args, restarts=1):
    results = []
    for restart in range(restarts):
        results.append(fit_theta_once(
            features, train_idx, theta_initial, spec, smooth_lambda,
            train_alpha=args.train_alpha,
            tau=args.tau,
            eta=args.eta,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            patience=args.patience,
            epsilon=args.score_epsilon,
            seed=args.seed + 1009 * restart,
            device=args.optimization_device,
            noise_scale=0.0 if restart == 0 else args.restart_noise,
        ))
    return max(results, key=lambda x: x.objective)


def score_numpy(u, v, theta_matrix, epsilon=1e-8):
    theta = np.asarray(theta_matrix, dtype=np.float64).reshape(-1)
    numerator = np.asarray(u, dtype=np.float64) @ theta
    denominator2 = np.einsum('d,nde,e->n', theta, np.asarray(v, dtype=np.float64), theta, optimize=True)
    return numerator / np.sqrt(np.clip(denominator2, epsilon, None))


def conformal_p_values(calibration_scores, test_scores):
    calibration_scores = np.asarray(calibration_scores, dtype=np.float64)
    test_scores = np.asarray(test_scores, dtype=np.float64)
    sorted_cal = np.sort(calibration_scores)
    smaller = np.searchsorted(sorted_cal, test_scores, side='left')
    ge = len(sorted_cal) - smaller
    return (1.0 + ge.astype(np.float64)) / (len(sorted_cal) + 1.0)


def full_auc(human_scores, machine_scores):
    labels = np.concatenate([np.zeros(len(human_scores)), np.ones(len(machine_scores))])
    scores = np.concatenate([human_scores, machine_scores])
    return float(roc_auc_score(labels, scores))


def partial_auc(human_scores, machine_scores, max_fpr=0.05):
    labels = np.concatenate([np.zeros(len(human_scores)), np.ones(len(machine_scores))])
    scores = np.concatenate([human_scores, machine_scores])
    return float(roc_auc_score(labels, scores, max_fpr=max_fpr))


def empirical_power_type_i(human_scores, machine_scores, alpha=0.05):
    try:
        threshold = float(np.quantile(human_scores, 1.0 - alpha, method='higher'))
    except TypeError:
        threshold = float(np.quantile(human_scores, 1.0 - alpha, interpolation='higher'))
    return (
        float(np.mean(machine_scores >= threshold)),
        float(np.mean(human_scores >= threshold)),
        threshold,
    )


@dataclass(frozen=True)
class HyperparameterConfig:
    j_bases: int
    k_bases: int
    smooth_lambda: float

    @property
    def dimension(self):
        return self.j_bases * self.k_bases

    @property
    def name(self):
        return f'J{self.j_bases}_K{self.k_bases}_lambda{self.smooth_lambda:g}'


def config_sort_key(record):
    return (
        float(record['cv_pauc']),
        float(record['cv_power_at_selection_fpr']),
        -int(record['dimension']),
        float(record['smooth_lambda']),
    )
