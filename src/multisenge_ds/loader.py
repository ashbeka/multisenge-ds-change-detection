"""Loader utilities for MultiSenGE Sentinel‑2 data.

This module provides a simple interface for loading multi‑temporal Sentinel‑2
image patches from the MultiSenGE dataset.  Each patch comprises multiple
acquisition dates of the same geographic region.  The current implementation
focuses on loading two selected timestamps for change detection experiments.

Given the limitations of this environment, we cannot access the actual
MultiSenGE data files directly.  Instead this loader outlines the expected
file structure and provides minimal functionality for synthetic examples.  If
you have the dataset locally, you can extend ``read_multiband_image`` to
open GeoTIFF or NumPy files from disk.

Typical usage::

    from pathlib import Path
    from ds.loader import load_patch

    # Load two dates from a patch directory on disk
    X_t1, X_t2, meta = load_patch(
        patch_dir=Path('/path/to/patch'),
        date_indices=(0, -1),
        reflectance_scale=10000.0,
    )

``X_t1`` and ``X_t2`` will be returned as ``(H, W, B)`` arrays of type
``float32``.  The optional ``meta`` dictionary includes the paths or
timestamps of the loaded files.

Note that MultiSenGE uses a resampled band set of 10 Sentinel‑2 bands (B2,
B3, B4, B8, B5, B6, B7, B8A, B11, B12) at 10 m resolution.  If your data
contains all 13 bands you can modify ``read_multiband_image`` to return
additional channels.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np

__all__ = ["load_patch", "read_multiband_image"]


def read_multiband_image(path: Path, reflectance_scale: Optional[float] = None) -> np.ndarray:
    """Read a multi‑band satellite image from disk.

    This helper function abstracts the underlying file format.  If the file is
    a NumPy ``.npy`` file it is loaded with ``np.load``.  For other formats
    (e.g. GeoTIFF) you would normally use ``rasterio`` to read the data; here
    we raise ``NotImplementedError`` to encourage users to implement their own
    loader if they have access to the actual data.

    Parameters
    ----------
    path : Path
        Path to the multi‑band image file.
    reflectance_scale : float, optional
        Sentinel‑2 Level‑2A products are typically stored as 16‑bit integers
        scaled by 10 000.  If provided, the data are divided by this factor
        to yield reflectance in the range [0, 1].

    Returns
    -------
    np.ndarray
        Array of shape (H, W, B) containing float32 reflectance values.
    """
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
        if reflectance_scale:
            arr = arr.astype(np.float32) / float(reflectance_scale)
        else:
            arr = arr.astype(np.float32)
        return arr
    elif suffix == ".tif" or suffix == ".tiff":
        try:
            import rasterio
            with rasterio.open(path) as src:
                # Read all bands and transpose to (H, W, B)
                arr = src.read().transpose(1, 2, 0).astype(np.float32)
                if reflectance_scale:
                    arr = arr / float(reflectance_scale)
                return arr
        except ImportError:
            raise ImportError("rasterio is required to read GeoTIFF files. Install with: pip install rasterio")
    raise NotImplementedError(
        f"Unable to read file type '{suffix}'. Please implement raster loading via rasterio or GDAL."
    )


def load_patch(
    patch_dir: Path,
    date_indices: Tuple[int, int] = (0, -1),
    reflectance_scale: Optional[float] = 10000.0,
    custom_loader: Optional[callable] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Load two Sentinel‑2 acquisitions from a MultiSenGE patch directory.

    Parameters
    ----------
    patch_dir : Path
        Directory containing multiple acquisition files for a single patch.
    date_indices : tuple of int, optional
        Index pair ``(i, j)`` indicating which two dates to load from the
        sorted list of files.  Negative indices are permitted (as in Python
        slicing).  By default the earliest and latest dates are used.
    reflectance_scale : float, optional
        If provided, divide integer pixel values by this scale to obtain
        reflectance.
    custom_loader : callable, optional
        Function to override ``read_multiband_image``.  It should accept a
        ``Path`` and ``reflectance_scale`` and return a numpy array of
        shape (H, W, B).

    Returns
    -------
    (np.ndarray, np.ndarray, Dict[str, object])
        ``X1, X2, meta`` where ``X1`` and ``X2`` are arrays of shape
        ``(H, W, B)`` for the selected dates.  The ``meta`` dictionary
        includes the sorted file paths and the indices used.

    Notes
    -----
    If there are fewer than two files in ``patch_dir`` this function will
    raise a ``ValueError``.  The data are assumed to be co‑registered across
    dates, as specified in the MultiSenGE documentation.
    """
    # Discover candidate files
    files = [p for p in sorted(patch_dir.iterdir()) if p.is_file() and not p.name.startswith('.')]
    if len(files) < 2:
        raise ValueError(f"Expected at least two files in patch directory {patch_dir}, found {len(files)}")

    # Sort by date encoded in filename (heuristic: assume ISO date after last underscore)
    def date_key(p: Path) -> str:
        parts = p.stem.split('_')
        for part in reversed(parts):
            if len(part) >= 8 and part[:4].isdigit():
                return part
        return p.stem

    files.sort(key=date_key)
    i, j = date_indices
    # Normalize indices
    if i < 0:
        i = len(files) + i
    if j < 0:
        j = len(files) + j
    if not (0 <= i < len(files) and 0 <= j < len(files)):
        raise IndexError(f"date_indices {date_indices} out of range for {len(files)} files")
    if i == j:
        raise ValueError("date_indices must refer to two distinct dates")

    load_fn = custom_loader or read_multiband_image
    f1, f2 = files[i], files[j]
    X1 = load_fn(f1, reflectance_scale=reflectance_scale)
    X2 = load_fn(f2, reflectance_scale=reflectance_scale)
    meta: Dict[str, object] = {
        "files": files,
        "selected": (f1, f2),
        "indices": (i, j),
    }
    return X1, X2, meta