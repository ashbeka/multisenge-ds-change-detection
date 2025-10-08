"""Visualization utilities for difference subspace change detection.

This module provides helper functions to create visual representations of
multispectral Sentinel‑2 data and change detection results.  The functions
use matplotlib and follow guidelines: each plot is in its own figure,
no specific colours are forced, and default colormaps are used unless
explicitly specified by the caller.  Heatmaps include a colour bar to
indicate the magnitude scale.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non‑interactive backend for saving files
import matplotlib.pyplot as plt
from typing import Tuple, Iterable

__all__ = [
    "rgb_composite",
    "save_heatmap",
    "save_histogram",
    "save_angle_bar",
]


def rgb_composite(
    X: np.ndarray,
    rgb_bands: Tuple[int, int, int] = (2, 1, 0),
    percentile: Tuple[float, float] = (2, 98),
) -> np.ndarray:
    """Generate an RGB composite from a multispectral image cube.

    Parameters
    ----------
    X : np.ndarray of shape (H, W, B)
        Image cube with spectral bands along the last axis.
    rgb_bands : tuple of int, optional
        Indices of the bands to use for red, green and blue channels.
        Defaults to (2, 1, 0), corresponding to (B4, B3, B2) if the first
        three bands of X are ordered as B2, B3, B4.
    percentile : tuple of float, optional
        Lower and upper percentiles for contrast stretching.  Values below
        the lower percentile are clipped to 0, and values above the upper
        percentile are clipped to 1.

    Returns
    -------
    np.ndarray of shape (H, W, 3)
        An RGB image scaled to [0, 1].
    """
    H, W, B = X.shape
    assert all(0 <= b < B for b in rgb_bands), "RGB band indices out of range"
    rgb = X[..., list(rgb_bands)].astype(np.float32)
    # Percentile stretch per channel
    stretched = np.zeros_like(rgb)
    for i in range(3):
        band = rgb[..., i]
        lo, hi = np.percentile(band, percentile)
        if hi - lo > 0:
            stretched[..., i] = np.clip((band - lo) / (hi - lo), 0, 1)
        else:
            stretched[..., i] = 0
    return stretched


def save_heatmap(
    matrix: np.ndarray,
    out_path: str,
    title: str = "Change Map",
    cmap: str = 'viridis',
) -> None:
    """Save a heatmap image of a 2D matrix.

    Parameters
    ----------
    matrix : np.ndarray of shape (H, W)
        The data to visualize.
    out_path : str
        File path to save the figure (e.g. 'ds_change.png').
    title : str, optional
        Figure title.
    cmap : str, optional
        Matplotlib colormap name for the heatmap.  Defaults to 'viridis'.
    """
    plt.figure()
    im = plt.imshow(matrix, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_histogram(
    scores: np.ndarray,
    out_path: str,
    title: str = "Change Score Histogram",
    bins: int = 50,
) -> None:
    """Save a histogram of change scores.

    Parameters
    ----------
    scores : np.ndarray
        Flattened array of change scores.
    out_path : str
        File path to save the histogram.
    title : str, optional
        Title of the histogram.
    bins : int, optional
        Number of bins for the histogram.
    """
    plt.figure()
    plt.hist(scores.ravel(), bins=bins)
    plt.title(title)
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_angle_bar(
    cosines: Iterable[float],
    out_path: str,
    title: str = "Principal Angles",
) -> None:
    """Save a bar chart of principal angles from cosines.

    Parameters
    ----------
    cosines : iterable of float
        Cosines of principal angles, sorted in descending order.
    out_path : str
        Path to save the bar chart.
    title : str, optional
        Chart title.
    """
    cos = np.clip(np.array(list(cosines)), -1.0, 1.0)
    angles = np.degrees(np.arccos(cos))
    plt.figure()
    plt.bar(range(len(angles)), angles)
    plt.title(title)
    plt.xlabel("Component index")
    plt.ylabel("Angle (degrees)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()