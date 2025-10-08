"""Baseline change detectors for benchmarking Difference Subspace.

Band order (MultiSenGE S2):
[0:B02, 1:B03, 2:B04, 3:B08, 4:B05, 5:B06, 6:B07, 7:B8A, 8:B11, 9:B12]
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

__all__ = [
    "delta_ndvi",
    "delta_nbr",
    "delta_ndwi",
    "delta_mndwi",
    "delta_nir",
    "delta_swir",
    "cva_magnitude",
    "pca_difference_energy",
]

# Indices of commonly used bands
B03, B04, B08, B11, B12 = 1, 2, 3, 8, 9


def _safe_ratio(num: np.ndarray, den: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    den = den.copy()
    den[np.abs(den) < eps] = eps
    return num / den


def delta_ndvi(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    ndvi1 = _safe_ratio(X1[..., B08] - X1[..., B04], X1[..., B08] + X1[..., B04])
    ndvi2 = _safe_ratio(X2[..., B08] - X2[..., B04], X2[..., B08] + X2[..., B04])
    return (ndvi2 - ndvi1).astype(np.float32)


def delta_nbr(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    nbr1 = _safe_ratio(X1[..., B08] - X1[..., B12], X1[..., B08] + X1[..., B12])
    nbr2 = _safe_ratio(X2[..., B08] - X2[..., B12], X2[..., B08] + X2[..., B12])
    return (nbr2 - nbr1).astype(np.float32)


def delta_ndwi(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    ndwi1 = _safe_ratio(X1[..., B03] - X1[..., B08], X1[..., B03] + X1[..., B08])
    ndwi2 = _safe_ratio(X2[..., B03] - X2[..., B08], X2[..., B03] + X2[..., B08])
    return (ndwi2 - ndwi1).astype(np.float32)


def delta_mndwi(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    m1 = _safe_ratio(X1[..., B03] - X1[..., B11], X1[..., B03] + X1[..., B11])
    m2 = _safe_ratio(X2[..., B03] - X2[..., B11], X2[..., B03] + X2[..., B11])
    return (m2 - m1).astype(np.float32)


def delta_nir(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    return (X2[..., B08] - X1[..., B08]).astype(np.float32)


def delta_swir(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    return (X2[..., B12] - X1[..., B12]).astype(np.float32)


def cva_magnitude(X1: np.ndarray, X2: np.ndarray, standardize: bool = False) -> np.ndarray:
    dx = (X2 - X1).astype(np.float32)
    if standardize:
        # Standardize by pooled per-band std
        H, W, B = dx.shape
        flat1 = X1.reshape(-1, B)
        flat2 = X2.reshape(-1, B)
        pooled = np.vstack([flat1, flat2])
        std = pooled.std(axis=0, ddof=1)
        std[std == 0] = 1.0
        dx = dx / std.reshape(1, 1, B)
    mag = np.sqrt(np.sum(dx * dx, axis=-1))
    # Normalize to [0,1] for visualization
    mmax = float(mag.max())
    return (mag / mmax if mmax > 0 else mag).astype(np.float32)


def pca_difference_energy(X1: np.ndarray, X2: np.ndarray, k: Optional[int] = None) -> np.ndarray:
    """Energy of Δx projected on pooled PCA basis (baseline comparator).
    This is a naïve PCA-difference without canonical-angle normalization.
    """
    H, W, B = X1.shape
    flat1 = X1.reshape(-1, B).astype(np.float32)
    flat2 = X2.reshape(-1, B).astype(np.float32)
    pooled = np.vstack([flat1, flat2])
    # Compute covariance eigendecomposition
    C = pooled.T @ pooled / max(1, pooled.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    V = eigvecs[:, idx]
    if k is None:
        k = B
    V = V[:, :k].astype(np.float32)
    dx = (X2 - X1).reshape(-1, B).astype(np.float32)
    proj = dx @ V
    energy = np.sum(proj * proj, axis=1).reshape(H, W)
    # Normalize for visualization
    emax = float(energy.max())
    return (energy / emax if emax > 0 else energy).astype(np.float32)
