"""Principal Component Analysis utilities for spectral subspace extraction.

This module implements simple PCA routines suitable for multispectral
remote‑sensing imagery.  The focus is on extracting an orthonormal basis
representing the dominant variance directions of a set of pixel spectra.

For a matrix ``Z`` of shape (n_samples, n_features) with columns
representing spectral bands (features) and rows representing pixel
observations, the PCA basis returned by :func:`pca_basis` consists of
eigenvectors of the covariance matrix sorted by descending eigenvalues.  The
covariance matrix is computed on the mean‑centered data.  If
``n_components`` is ``None`` all available components are returned (i.e. the
full band space).  Otherwise a reduced set of the top ``n_components`` is
returned.

The PCA computation is performed using either numpy or PyTorch depending on
what is available.  PyTorch may accelerate the singular value decomposition
when a GPU is available.  For reproducibility and ease of use the code falls
back to numpy's linear algebra routines if PyTorch is not installed.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

__all__ = ["pca_basis"]


def pca_basis(
    Z: np.ndarray,
    n_components: Optional[int] = None,
    center: bool = True,
    whiten: bool = False,
    use_torch: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Compute a PCA basis for mean‑centered data.

    Parameters
    ----------
    Z : np.ndarray of shape (n_samples, n_features)
        Each row corresponds to one pixel and each column to one spectral band.
    n_components : int or None, optional
        Number of principal components to retain.  If ``None`` (default) the
        full set of components is returned.
    center : bool, optional
        If True (default), subtract the mean of each column from the data
        before computing the SVD.
    whiten : bool, optional
        If True, scale the principal components to have unit variance
        (i.e. divide by the singular values).  This is rarely needed for
        subspace methods and defaults to False.
    use_torch : bool, optional
        If True and PyTorch is available, use ``torch.linalg.svd`` for the
        decomposition.  Otherwise use numpy.  Using PyTorch on CPU is often
        slower but may be desirable for GPU acceleration.

    Returns
    -------
    (components, singular_values, mean, total_variance, explained_variance_ratio)
        ``components`` is an array of shape (n_features, n_components)
        containing orthonormal basis vectors.  ``singular_values`` is the
        singular values (square roots of eigenvalues) in descending order.
        ``mean`` is the column means used for centering.  ``total_variance``
        is the sum of variances across all features.  ``explained_variance_ratio``
        contains the fraction of total variance explained by each retained
        component.
    """
    # Validate input
    if Z.ndim != 2:
        raise ValueError("Input Z must be a 2D matrix of shape (n_samples, n_features)")
    n_samples, n_features = Z.shape
    k = n_features if n_components is None else int(n_components)
    if k < 1 or k > n_features:
        raise ValueError(f"n_components must be in [1, {n_features}], got {n_components}")

    # Compute column means and center data
    if center:
        mu = Z.mean(axis=0, keepdims=True)
        Z_centered = Z - mu
    else:
        mu = np.zeros((1, n_features), dtype=Z.dtype)
        Z_centered = Z

    # Use PyTorch if requested and available
    if use_torch and _TORCH_AVAILABLE:
        # Convert to float32 tensor
        Zt = torch.from_numpy(Z_centered).to(torch.float32)
        # Perform low‑rank SVD
        # Note: torch.linalg.svd returns U, S, Vh such that Z = U @ diag(S) @ Vh
        U, S, Vh = torch.linalg.svd(Zt, full_matrices=False)
        # Convert results back to numpy
        singular_values = S.detach().cpu().numpy()
        V = Vh.detach().cpu().numpy().T  # (n_features, n_features)
    else:
        # Use numpy for SVD.  Compute on the covariance matrix for efficiency.
        # Compute covariance matrix C = (Z_centered^T @ Z_centered) / (n_samples - 1)
        C = Z_centered.T @ Z_centered / float(max(1, n_samples - 1))
        # eigh returns eigenvalues in ascending order; take largest
        eigvals, eigvecs = np.linalg.eigh(C)
        # Sort descending by eigenvalue
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        singular_values = np.sqrt(np.maximum(eigvals, 0))  # nonnegative
        V = eigvecs
    # Select top k components
    V_k = V[:, :k]
    S_k = singular_values[:k]
    # Optionally whiten: divide each component by its singular value
    if whiten:
        V_k = V_k / S_k[np.newaxis, :]
    # Compute explained variance ratio
    total_var = (singular_values ** 2).sum()
    explained_variance_ratio = (S_k ** 2) / total_var if total_var > 0 else np.zeros_like(S_k)
    return V_k.astype(np.float32), S_k.astype(np.float32), mu.squeeze().astype(np.float32), float(total_var), explained_variance_ratio.astype(np.float32)