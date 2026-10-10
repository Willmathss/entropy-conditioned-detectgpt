"""AdaDetectGPT spline fit used by the paper's exact white-box baseline.

This is the small, cache-independent part of the original Ada witness helper.
The paper's runner supplies exact full-vocabulary moments, not a 128-token
approximation; the historical helper module is therefore unnecessary here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.interpolate import BSpline

from build_whitebox_reference_cache import AdaSplineSpec, ada_knots


class OfficialAdaSpline:
    """NumPy/SciPy equivalent of the public AdaDetectGPT spline basis."""

    def __init__(self, spec: AdaSplineSpec):
        self.spec = spec
        self.knots = ada_knots(spec)
        self._spline = BSpline(
            self.knots,
            np.eye(spec.n_bases, dtype=np.float64),
            spec.degree,
            extrapolate=True,
        )

    def transform(self, values) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        original_shape = values.shape
        flat = np.maximum(values.reshape(-1), self.spec.start)
        design = np.asarray(self._spline(flat), dtype=np.float64)
        if design.shape != (len(flat), self.spec.n_bases):
            raise ValueError(f"Unexpected spline design shape {design.shape}")
        if self.spec.intercept:
            design = np.concatenate((np.ones((len(flat), 1)), design), axis=1)
        return design.reshape(*original_shape, self.spec.dimension)


@dataclass
class Ada128Features:
    """Passage statistics; historical name retained for runner compatibility."""

    train_mean_human: np.ndarray
    train_cov_human: np.ndarray
    train_mean_machine: np.ndarray
    train_cov_machine: np.ndarray
    u_human: np.ndarray
    v_human: np.ndarray
    u_machine: np.ndarray
    v_machine: np.ndarray
    u_calibration: np.ndarray
    v_calibration: np.ndarray


@dataclass
class AdaFit:
    beta: np.ndarray
    delta_norm: float
    sigma_min_eigenvalue: float
    sigma_max_eigenvalue: float
    solver: str


def fit_witness(
    features: Ada128Features,
    train_indices: Iterable[int],
    *,
    solver: str,
    eigenvalue_floor: float,
    ridge: float,
) -> AdaFit:
    indices = np.asarray(list(train_indices), dtype=np.int64)
    delta = (
        features.train_mean_machine[indices].sum(axis=0)
        - features.train_mean_human[indices].sum(axis=0)
    )
    sigma = (
        features.train_cov_human[indices].sum(axis=0)
        + features.train_cov_machine[indices].sum(axis=0)
    )
    sigma = 0.5 * (sigma + sigma.T)
    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    if solver == "official-inv-sqrt":
        clipped = np.maximum(eigenvalues, eigenvalue_floor)
        inverse_sqrt = (eigenvectors * clipped[None, :] ** -0.5) @ eigenvectors.T
        beta = inverse_sqrt @ delta
    elif solver == "rayleigh-solve":
        beta = np.linalg.solve(sigma + ridge * np.eye(sigma.shape[0]), delta)
    else:
        raise ValueError(f"Unknown Ada solver {solver!r}")
    norm = float(np.linalg.norm(beta))
    if not np.isfinite(norm) or norm <= 1e-14:
        raise FloatingPointError(f"Degenerate Ada witness: beta norm={norm}")
    return AdaFit(
        beta=beta / norm,
        delta_norm=float(np.linalg.norm(delta)),
        sigma_min_eigenvalue=float(eigenvalues.min()),
        sigma_max_eigenvalue=float(eigenvalues.max()),
        solver=solver,
    )


def score_from_features(
    u: np.ndarray, v: np.ndarray, beta: np.ndarray, epsilon: float
) -> np.ndarray:
    numerator = np.asarray(u, dtype=np.float64) @ beta
    denominator2 = np.einsum(
        "d,nde,e->n", beta, np.asarray(v, dtype=np.float64), beta, optimize=True
    )
    return numerator / np.sqrt(np.clip(denominator2, epsilon, None))
