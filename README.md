# Difference Subspace Change Detection Prototype

This repository contains a prototype implementation of **first-order Difference Subspace (DS) analysis** for multi-temporal Sentinel-2 imagery. It focuses on the core linear-algebra workflow (PCA + subspace analysis) and does *not* include deep learning components. The goal is to provide a reference implementation that can be extended or embedded in larger remote-sensing pipelines.

## Background

Difference Subspace analysis identifies directions in spectral space where two images of the same scene differ the most. Each date is represented by a PCA-derived subspace; the singular vectors of the cross-basis product yield principal angles and a difference subspace that captures dominant spectral changes. Projecting pixel-wise spectral differences onto this subspace highlights land-cover changes and other anomalies. See the reference papers in `docs/refs/papers/` for details (Fukui and Maki 2015; Fukui et al. 2024).

## Repository Layout

- `src/multisenge_ds/diffsubspace.py` – DS routines (canonical angles, difference subspace, projection and cross-residual scores).
- `src/multisenge_ds/pca.py` – PCA utilities (NumPy-based with optional PyTorch fallback) returning components, singular values, and explained variance.
- `src/multisenge_ds/loader.py` – Sentinel-2 loader supporting `.npy` out of the box and GeoTIFF via `rasterio` when installed.
- `src/multisenge_ds/vis/plots.py` – Visualization helpers for RGB composites, heatmaps, histograms, and principal-angle bar charts.
- `scripts/run_ds_demo.py` – Command-line demo that reads YAML configs, loads data (synthetic or real), runs DS, and saves outputs.
- `docs/` – Documentation index (`docs/README.md`), core specs (`docs/core/`), design notes (`docs/design/`), guides (`docs/guides/`), examples, and references.
- `.vscode/launch.json` – Ready-to-use VS Code launch configurations for running the demos without manual shell commands.

## Environment Setup

```
powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install numpy matplotlib pyyaml rasterio
```

The demos expect Python to locate modules under `src/`. The VS Code launch configurations set `PYTHONPATH=${workspaceFolder}/src`. From PowerShell you can run:

```
powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts\run_ds_demo.py --config demo_config.yaml
```

## Running the Synthetic Demo

1. Adjust `demo_config.yaml` if desired (subspace dimension, RGB bands, synthetic cube shape, etc.).
2. Run using the VS Code configuration **DS Demo (Synthetic)** or the command above.
3. Outputs (RGB composites, DS projection/cross-residual maps, histograms, angles) are written to `outputs/synthetic_demo/`. Example artifacts are stored in `docs/examples/synthetic/` for quick reference.

## Running on MultiSenGE Sentinel-2 Tiles

1. Place Sentinel-2 tiles on local storage (see `docs/guides/DATA_MANAGEMENT.md` for guidance). The repository expects filenames of the form `{tile}_{yyyymmdd}_S2_{x}_{y}.tif`.
2. Edit `real_s2_config.yaml` (or create a custom config) with:
   - `data_root`: directory containing the `s2/` folder.
   - `patch_id`: tile prefix (for example `32ULV_20201031_S2`).
   - Optional `spatial_coords`: specific `{x}_{y}`; if omitted, the first available coordinate is used.
   - `date_indices`: indices into the sorted date list (defaults to first/last).
3. Run via the **DS Demo (Sentinel-2)** VS Code configuration or:

```
powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts\run_ds_demo.py --config real_s2_config.yaml
```

The script loads the selected pair of dates, computes PCA bases, derives the difference subspace, and saves outputs to `outputs/real_s2_demo/` (configurable via `out_dir`).

## Notes and Limitations

- The loader assumes all tiles share the documented 10-band order (B02,B03,B04,B08,B05,B06,B07,B8A,B11,B12) and are co-registered at 256×256 pixels.
- No cloud or shadow masking is applied; clouds may appear as change.
- PCA uses full SVD; very large tiles may require incremental or randomized SVD for performance.
- Raster data are not packaged with the repository. Keep large datasets outside version control and follow `docs/guides/DATA_MANAGEMENT.md` for best practices.

## Roadmap and Future Work

Planning, design discussions, and upcoming features (including the sliding-window DS extension) are tracked in `docs/design/ROADMAP.md` and `docs/design/SLIDING_WINDOW_DESIGN.md`. Contributions should align with those documents.

## License

This project is provided for research and educational purposes. Cite the associated papers when using this code in academic work.

