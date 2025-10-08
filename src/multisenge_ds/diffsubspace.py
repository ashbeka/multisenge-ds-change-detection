"""Difference subspace computations and change scoring.

This module implements first‑order difference subspace (DS) analysis for
multi‑spectral imagery.  Given two sets of pixel spectra representing the
same spatial patch at two different times, it constructs orthonormal bases
for each time via PCA, then computes the canonical angles between these
subspaces to derive a difference subspace capturing spectral directions that
change the most.  Per‑pixel change scores are computed either as the
projection of the spectral difference onto the difference subspace or as
cross‑residual reconstruction errors.

References
----------
Fukui and Maki, "Difference Subspace and its Generalization", 2015.
Kanai et al., "Time‑series Anomaly Detection based on Difference Subspace", 2023.
"Second‑order Difference Subspace", 2024.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Dict, Optional, Union

from .pca import pca_basis

__all__ = [
    "difference_subspace",
    "projection_change_score",
    "cross_residual_change_score",
    "canonical_angles",
]


def canonical_angles(phi: np.ndarray, psi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute canonical (principal) angles between two subspaces.

    Given two orthonormal basis matrices ``phi`` and ``psi`` of shape
    (B, k1) and (B, k2), this function computes the singular values of
    ``phi.T @ psi`` to derive the cosines of the principal angles.  It returns
    ``U, cos_theta, V`` where ``U`` and ``V`` are the left and right singular
    vectors of the product matrix and ``cos_theta`` is a 1D array of singular
    values in descending order.

    Parameters
    ----------
    phi : np.ndarray
        Orthonormal basis of shape (B, k1).
    psi : np.ndarray
        Orthonormal basis of shape (B, k2).

    Returns
    -------
    (U, cos_theta, V)
        ``U`` and ``V`` are orthonormal matrices; ``cos_theta`` is a 1D array
        whose entries are the cosines of the principal angles between the
        subspaces.  ``cos_theta`` is sorted in descending order.
    """
    # Compute the cross matrix M = phi^T psi.  Size (k1, k2)
    M = phi.T @ psi
    # Perform singular value decomposition
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    # Clip singular values into [0, 1] to handle numerical errors
    s = np.clip(s, -1.0, 1.0)
    return U, s, Vt.T


def difference_subspace(
    phi: np.ndarray,
    psi: np.ndarray,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the first‑order difference subspace and principal subspace.

    Given two orthonormal basis matrices ``phi`` and ``psi`` of shape
    (B, k1) and (B, k2), compute matrices ``D`` and ``M`` whose columns
    form orthonormal bases for the difference and principal subspaces,
    respectively.  Also return the principal angles between the original
    subspaces.

    Parameters
    ----------
    phi : np.ndarray
        Orthonormal basis of shape (B, k1).
    psi : np.ndarray
        Orthonormal basis of shape (B, k2).
    tol : float, optional
        Tolerance for discarding singular values extremely close to 1 or 0.

    Returns
    -------
    (D, M, cosines, U, V)
        ``D`` : (B, d_D) basis for the difference subspace.
        ``M`` : (B, d_M) basis for the principal (mean) subspace.
        ``cosines`` : 1D array of cosines of principal angles.
        ``U, V`` : left and right singular vectors of ``phi.T psi``.
    """
    U, s, V = canonical_angles(phi, psi)
    # Determine which singular values correspond to nearly identical or
    # orthogonal directions
    mask_diff = (1.0 - s) > tol  # not too close to 1.0 (common directions)
    mask_prin = (1.0 + s) > tol  # not too close to -1.0 (should always be True)
    # Build difference basis
    # diff = phi@U - psi@V
    diff_raw = phi @ U - psi @ V
    # Scale columns by 1/sqrt(2*(1 - s_i)) for each i
    scales_diff = np.zeros_like(s)
    # Avoid division by zero by setting scale to 0 for cosines ~1
    for i, c in enumerate(s):
        if 1.0 - c > tol:
            scales_diff[i] = 1.0 / np.sqrt(2.0 * (1.0 - c))
        else:
            scales_diff[i] = 0.0
    # Form full difference basis and then select non‑zero columns
    D_full = diff_raw * scales_diff[np.newaxis, :]
    D = D_full[:, mask_diff]
    # Orthogonalize in case of numerical issues (QR)
    if D.size > 0:
        D, _ = np.linalg.qr(D)

    # Build principal (mean) basis
    sum_raw = phi @ U + psi @ V
    scales_prin = np.zeros_like(s)
    for i, c in enumerate(s):
        if 1.0 + c > tol:
            scales_prin[i] = 1.0 / np.sqrt(2.0 * (1.0 + c))
        else:
            scales_prin[i] = 0.0
    M_full = sum_raw * scales_prin[np.newaxis, :]
    M = M_full[:, mask_prin]
    if M.size > 0:
        M, _ = np.linalg.qr(M)

    return D.astype(np.float32), M.astype(np.float32), s.astype(np.float32), U.astype(np.float32), V.astype(np.float32)


def projection_change_score(
    X1: np.ndarray,
    X2: np.ndarray,
    D: np.ndarray,
    scaled: bool = False,
) -> np.ndarray:
    """Compute per‑pixel change magnitude using the projection onto the difference subspace.

    Parameters
    ----------
    X1 : np.ndarray of shape (H, W, B)
        Pre‑event image cube.
    X2 : np.ndarray of shape (H, W, B)
        Post‑event image cube.
    D : np.ndarray of shape (B, d_D)
        Orthonormal basis for the difference subspace.
    scaled : bool, optional
        If True, scale the scores to the range [0, 1] by dividing by the
        maximum value across the image.  Defaults to False.

    Returns
    -------
    np.ndarray of shape (H, W)
        Change magnitude map.  Higher values indicate greater spectral
        differences along the difference subspace directions.
    """
    if D.size == 0:
        # No difference components, return zero map
        return np.zeros((X1.shape[0], X1.shape[1]), dtype=np.float32)
    # Compute difference image (H,W,B)
    delta = X2.astype(np.float32) - X1.astype(np.float32)
    H, W, B = delta.shape
    # Reshape to (N,B)
    delta_flat = delta.reshape(-1, B)
    # Project onto D: (N,B) @ (B,d) -> (N,d)
    proj = delta_flat @ D
    # Compute squared norm of projected vectors
    scores = np.sum(proj ** 2, axis=1)
    scores_map = scores.reshape(H, W)
    if scaled and scores_map.max() > 0:
        scores_map = scores_map / scores_map.max()
    return scores_map.astype(np.float32)


def cross_residual_change_score(
    X1: np.ndarray,
    X2: np.ndarray,
    phi: np.ndarray,
    psi: np.ndarray,
    scaled: bool = False,
) -> np.ndarray:
    """Compute per‑pixel change score using cross‑reconstruction residuals.

    Each pixel at time ``t1`` is projected onto the subspace for ``t2``
    (defined by ``psi``) and vice versa.  The residual (difference between
    original and reconstruction) is squared and summed over bands.  The
    average of the two residuals yields the final change score.  Large
    residuals indicate that the pixel spectrum does not lie well within the
    other time’s subspace and therefore likely underwent a spectral change.

    Parameters
    ----------
    X1 : np.ndarray of shape (H, W, B)
        Pre‑event image cube.
    X2 : np.ndarray of shape (H, W, B)
        Post‑event image cube.
    phi : np.ndarray of shape (B, k1)
        PCA basis for pre‑event image.
    psi : np.ndarray of shape (B, k2)
        PCA basis for post‑event image.
    scaled : bool, optional
        If True, scale the scores to [0, 1] by dividing by the maximum.

    Returns
    -------
    np.ndarray of shape (H, W)
        Change score map based on cross residuals.
    """
    H, W, B = X1.shape
    # Flatten to (N,B)
    X1_f = X1.reshape(-1, B).astype(np.float32)
    X2_f = X2.reshape(-1, B).astype(np.float32)
    # Project onto other subspaces and compute residuals
    # P_psi = psi psi^T
    # P_phi = phi phi^T
    # Residual of x1 onto psi
    # Instead of computing full BxB projector, compute projections in band space
    proj1_on_psi = (X1_f @ psi) @ psi.T  # (N,k2)@(k2,B) -> (N,B)
    r1 = X1_f - proj1_on_psi
    proj2_on_phi = (X2_f @ phi) @ phi.T
    r2 = X2_f - proj2_on_phi
    # Compute squared norm of residuals per pixel
    s1 = np.sum(r1 ** 2, axis=1)
    s2 = np.sum(r2 ** 2, axis=1)
    scores = 0.5 * (s1 + s2)
    scores_map = scores.reshape(H, W)
    if scaled and scores_map.max() > 0:
        scores_map = scores_map / scores_map.max()
    return scores_map.astype(np.float32)