#!/usr/bin/env python3
"""Run a Difference Subspace demo on a MultiSenGE patch or synthetic data.

This script demonstrates the first窶双rder Difference Subspace (DS) change
detection method on two multi窶奏emporal Sentinel窶・ acquisitions.  The user
provides a YAML configuration specifying the dataset root, a patch ID and
two timestamps, along with other parameters such as the number of PCA
components.  The script loads the images, computes PCA bases, derives
the difference subspace, computes change scores using both projection and
cross窶喪esidual methods, and saves visualizations.

If a dataset is not available, you can omit ``data_root`` and the script
will generate synthetic multispectral data to illustrate the workflow.  In
that case random Gaussian noise is added between the two timestamps to
simulate change.  This option is useful for testing the DS implementation
without requiring the MultiSenGE dataset.

Example configuration (YAML)::

    data_root: /path/to/MultiSenGE
    s2_dir: s2
    patch_id: PATCH_0000
    timestamps: ["2020-04-01", "2020-09-01"]
    rgb_bands: [2, 1, 0]  # indices for bands B4, B3, B2
    subspace_k: 6
    change_score: proj_energy
    use_cloud_mask: false
    out_dir: outputs/PATCH_0000

Run via::

    python scripts/run_ds_demo.py --config path/to/config.yaml

Note: this script depends only on numpy, pyyaml and matplotlib.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import yaml
import numpy as np

from multisenge_ds.loader import load_patch
from multisenge_ds.pca import pca_basis
from multisenge_ds.diffsubspace import difference_subspace, projection_change_score, cross_residual_change_score
from multisenge_ds.vis import rgb_composite, save_heatmap, save_histogram, save_angle_bar


def load_s2_files_by_date_pattern(
    s2_dir: Path,
    tile_prefix: str,
    spatial_coords: str | None = None,
    date_indices: Sequence[int] | None = (0, -1),
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Load S2 files by selecting different dates for the same spatial location.
    
    Parameters
    ----------
    s2_dir : Path
        Directory containing S2 .tif files
    tile_prefix : str
        Tile prefix pattern like "32ULV"
    spatial_coords : str, optional
        Specific spatial coordinates like "257_6425"
    date_indices : tuple
        Which two dates to select
        
    Returns
    -------
    X1, X2, meta : tuple
        Two arrays and metadata
    """
    from multisenge_ds.loader import read_multiband_image
    
    # Get all files matching the tile prefix
    all_files = [f for f in s2_dir.iterdir() if f.name.startswith(tile_prefix) and f.suffix == '.tif']
    
    # Prepare date indices pair
    if date_indices is None:
        index_pair = (0, -1)
    else:
        index_pair = tuple(int(idx) for idx in date_indices)
        if len(index_pair) != 2:
            raise ValueError('date_indices must contain exactly two entries')

    # Extract unique dates and spatial coordinates
    dates = set()
    spatial_coords_set = set()
    
    for f in all_files:
        parts = f.stem.split('_')
        if len(parts) >= 5:
            date = parts[1]  # e.g., "20201031"
            spatial = f"{parts[3]}_{parts[4]}"  # e.g., "257_6425"
            dates.add(date)
            spatial_coords_set.add(spatial)
    
    dates = sorted(list(dates))
    
    # If no specific spatial coordinates given, use the first available one
    if spatial_coords is None:
        spatial_coords = sorted(list(spatial_coords_set))[0]
    
    print(f"Found {len(dates)} dates: {dates}")
    print(f"Using spatial coordinates: {spatial_coords}")
    
    # Select dates
    i, j = index_pair
    if i < 0:
        i = len(dates) + i
    if j < 0:
        j = len(dates) + j
    
    selected_dates = [dates[i], dates[j]]
    print(f"Selected dates: {selected_dates}")
    
    # Find files for the selected dates and spatial coordinates
    selected_files = []
    for date in selected_dates:
        pattern = f"{tile_prefix}_{date}_S2_{spatial_coords}.tif"
        file_path = s2_dir / pattern
        if file_path.exists():
            selected_files.append(file_path)
        else:
            raise FileNotFoundError(f"Could not find file: {file_path}")
    
    # Load the two files
    X1 = read_multiband_image(selected_files[0], reflectance_scale=10000.0)
    X2 = read_multiband_image(selected_files[1], reflectance_scale=10000.0)
    
    meta = {
        "files": all_files,
        "selected": selected_files,
        "dates": selected_dates,
        "spatial_coords": spatial_coords
    }
    
    return X1, X2, meta


def generate_synthetic_patch(
    shape: tuple[int, int, int] = (256, 256, 10),
    change_std: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic pair of multispectral images.

    The first image is drawn from a uniform distribution over [0, 1], and the
    second is the same plus additive Gaussian noise.  Use this to test
    Difference Subspace without a real dataset.
    """
    rng = np.random.default_rng(seed)
    X1 = rng.random(shape, dtype=np.float32)
    noise = rng.normal(scale=change_std, size=shape).astype(np.float32)
    X2 = np.clip(X1 + noise, 0, 1)
    return X1, X2


def main(config_path: Path) -> None:  # noqa: C0116
    # Load YAML configuration
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    # Resolve output directory
    out_dir = Path(cfg.get('out_dir', 'outputs'))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine whether to use synthetic data
    if cfg.get('data_root'):
        # Use real dataset
        data_root = Path(cfg['data_root'])
        s2_dir = data_root / cfg.get('s2_dir', 's2')
        patch_id = cfg['patch_id']
        date_indices = cfg.get('date_indices')
        if date_indices is None:
            # Use first and last by default
            date_indices = (0, -1)
            
        # Check if patch_id looks like an S2 tile pattern
        if '_S2' in patch_id:
            # Extract tile prefix (e.g., "32ULV" from "32ULV_20201031_S2")
            tile_prefix = patch_id.split('_')[0]
            spatial_coords = cfg.get('spatial_coords', None)  # e.g., "257_6425"
            X1, X2, meta = load_s2_files_by_date_pattern(s2_dir, tile_prefix, spatial_coords, date_indices)
            dates_meta = cast(Sequence[str], meta.get('dates', []))
            print(f"Loaded S2 data for tile {tile_prefix} from dates {list(dates_meta)}")
        else:
            # Use original patch loading method
            patch_path = s2_dir / patch_id
            load_indices = (0, -1) if date_indices is None else date_indices
            X1, X2, meta = load_patch(
                patch_path,
                date_indices=load_indices,
                reflectance_scale=cfg.get('reflectance_scale', 10000.0),
            )
            selected_meta = cast(Sequence[Path], meta.get('selected', []))
            if len(selected_meta) >= 2:
                print(f"Loaded patch {patch_id} from {selected_meta[0].name} and {selected_meta[1].name}")
    else:
        # Generate synthetic data
        shape = cfg.get('synthetic_shape', [256, 256, 10])
        change_std = cfg.get('synthetic_change_std', 0.15)
        seed = cfg.get('seed', 0)
        X1, X2 = generate_synthetic_patch(tuple(shape), change_std=change_std, seed=seed)
        print(f"Generated synthetic data of shape {X1.shape}")

    H, W, B = X1.shape
    # Flatten to (N,B) for PCA
    Z1 = X1.reshape(-1, B)
    Z2 = X2.reshape(-1, B)
    # Compute PCA bases
    k = cfg.get('subspace_k', B)
    phi, _, _, _, _ = pca_basis(Z1, n_components=k, center=True)
    psi, _, _, _, _ = pca_basis(Z2, n_components=k, center=True)
    # Compute difference subspace
    D, _, cosines, _, _ = difference_subspace(phi, psi)
    # Compute projection change map
    proj_map = projection_change_score(X1, X2, D, scaled=True)
    # Compute cross residual change map
    res_map = cross_residual_change_score(X1, X2, phi, psi, scaled=True)
    # Build RGB composites
    rgb_sequence = tuple(int(b) for b in cfg.get('rgb_bands', [2, 1, 0]))
    if len(rgb_sequence) != 3:
        raise ValueError('rgb_bands must contain exactly three band indices')
    rgb_bands = cast(tuple[int, int, int], rgb_sequence)
    rgb1 = rgb_composite(X1, rgb_bands=rgb_bands)
    rgb2 = rgb_composite(X2, rgb_bands=rgb_bands)
    # Save figures
    # Save RGB images
    # Save RGB composites using matplotlib.  We avoid imageio to minimise
    # external dependencies.
    import matplotlib.pyplot as plt
    for img, name in ((rgb1, 't1_rgb.png'), (rgb2, 't2_rgb.png')):
        plt.figure()
        plt.imshow(np.clip(img, 0, 1))
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(out_dir / name)
        plt.close()
    # Save heatmaps
    save_heatmap(proj_map, out_path=str(out_dir / 'ds_proj_map.png'), title='DS Projection Change Map')
    save_heatmap(res_map, out_path=str(out_dir / 'ds_cross_residual_map.png'), title='Cross Residual Change Map')
    # Save histograms
    save_histogram(proj_map, out_path=str(out_dir / 'ds_proj_hist.png'), title='DS Projection Change Score Histogram')
    save_histogram(res_map, out_path=str(out_dir / 'ds_cross_residual_hist.png'), title='Cross Residual Change Score Histogram')
    # Save principal angles
    save_angle_bar(cosines, out_path=str(out_dir / 'ds_principal_angles.png'), title='Principal Angles')
    print(f"Saved outputs to {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run Difference Subspace demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--config',
        type=str,
        default=str(PROJECT_ROOT / 'demo_config.yaml'),
        help='Path to YAML configuration file',
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    if config_path == (PROJECT_ROOT / 'demo_config.yaml'):
        print(f"Using default configuration: {config_path}")

    main(config_path)









