"""Core package for MultiSenGE Difference Subspace utilities.

Provides PCA, difference subspace computation, data loading, and visualization helpers.
"""

from .pca import pca_basis
from .diffsubspace import (
    difference_subspace,
    projection_change_score,
    cross_residual_change_score,
    canonical_angles,
)
from .loader import load_patch, read_multiband_image
