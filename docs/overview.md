# Project Overview

This repository implements a prototype of the first-order Difference Subspace (DS) change-detection method for Sentinel-2 multispectral imagery. The focus is on linear algebra workflows—principal component analysis (PCA) and subspace comparisons—that highlight spectral differences between two acquisition dates without relying on deep learning models.

## Key Concepts

- **Principal Component Analysis (PCA):** Each date's spectral cube is compressed into an orthonormal basis using PCA to capture most variance with fewer components.
- **Difference Subspace:** The cross-basis product between the two PCA bases yields canonical angles whose singular vectors form the DS basis. Large canonical angles correspond to directions dominated by change across the two dates.
- **Projection Metrics:** Pixel-wise spectral differences are projected onto the DS basis and summarized as scores (projection energy and cross-residual) that emphasize land-cover changes and other anomalies.

## Code Organization

- `src/multisenge_ds/pca.py` provides PCA utilities with NumPy backends and optional PyTorch fallbacks.
- `src/multisenge_ds/diffsubspace.py` implements canonical angle computation, DS basis derivation, and projection metrics.
- `src/multisenge_ds/loader.py` handles loading Sentinel-2 tiles saved as `.npy` arrays or GeoTIFF rasters (via `rasterio` when installed).
- `src/multisenge_ds/vis/plots.py` offers visualization helpers for RGB composites, heatmaps, histograms, and principal-angle bar charts.
- `scripts/run_ds_demo.py` wires everything together: it loads two dates, computes PCA bases, derives the DS, and writes diagnostic plots defined by a YAML config.

## Usage Highlights

1. Create a Python environment and install dependencies (`numpy`, `matplotlib`, `pyyaml`, and optionally `rasterio`).
2. Set `PYTHONPATH=src` so the package modules resolve correctly.
3. Run `scripts/run_ds_demo.py --config <config.yaml>` to execute either the synthetic example (`demo_config.yaml`) or a Sentinel-2 pair (`real_s2_config.yaml`). Outputs (RGB composites, DS projections, histograms, angles) land in the configured `out_dir`.

## Cloud Workflow Walkthrough

The prototype runs equally well on a local workstation or a hosted environment such as GitHub Codespaces or a managed Jupyter hub. The following checklist describes the typical “first session” flow when you launch a cloud container:

1. **Initialize the workspace**
   - Clone or mount the repository into the remote environment.
   - Verify that the working directory contains `src/` and `scripts/`. If not, re-run the clone or adjust the mount path.
2. **Create an isolated Python environment**
   - Install `uv`/`pip`/`conda` (depending on what the platform provides by default).
   - Create and activate a fresh environment (e.g., `python -m venv .venv && source .venv/bin/activate`).
3. **Install dependencies**
   - From the repo root run `pip install -r configs/requirements-demo.txt` if present, or manually install `numpy`, `scipy`, `matplotlib`, `pyyaml`, and optional extras such as `rasterio` for GeoTIFF support.
   - On air-gapped environments download wheels locally and upload them before installation.
4. **Stage input data**
   - Upload `.npy` cubes or GeoTIFF scenes into a workspace folder and update the YAML config paths.
   - Ensure both dates are co-registered and share the same spatial dimensions before running the DS analysis.
5. **Configure the run**
   - Copy `demo_config.yaml` or `real_s2_config.yaml` into a personal config file.
   - Edit keys such as `input.a.path`, `input.b.path`, `out_dir`, and any plotting toggles to match your dataset and storage locations.
6. **Execute the pipeline**
   - Export `PYTHONPATH=src` in the active shell.
   - Launch `python scripts/run_ds_demo.py --config my_config.yaml` (replace with your filename). Monitor stdout for progress logs indicating PCA, DS, and plotting steps.
7. **Review outputs**
   - Download the figures from `out_dir` or open them directly within the cloud IDE’s file browser.
   - Inspect RGB composites, DS energy maps, and histograms to validate the run.
8. **Persist or share results**
   - Commit configs and lightweight artifacts to Git if desired.
   - For large rasters, archive them to cloud object storage before shutting down the session.
9. **Shut down cleanly**
   - Deactivate the virtual environment, stop the interpreter sessions, and terminate the cloud container to avoid unnecessary costs.

## Limitations and Next Steps

- The current implementation assumes 256×256, co-registered Sentinel-2 tiles ordered as B02,B03,B04,B08,B05,B06,B07,B8A,B11,B12.
- No cloud/shadow masking is built in; bright clouds may be interpreted as change.
- PCA uses full SVD and may need optimization for very large scenes.
- Design discussions and future enhancements (e.g., sliding-window DS) live under `docs/design/`.

For deeper theoretical background, consult `docs/refs/papers/` and the repository README for high-level guidance.
