# MultiSenGE Difference Subspace Technical Report


## Abstract

This report documents a first‑order Difference Subspace (DS) prototype for change detection using multi‑temporal Sentinel‑2 imagery from the MultiSenGE dataset. We model each date as a low‑dimensional spectral subspace via PCA, compute canonical angles between dates, derive a difference subspace, and score per‑pixel change using projection‑energy and cross‑residual criteria. The implementation is lightweight (NumPy; rasterio optional), configuration‑driven (YAML), and packaged for scripting or notebooks. We outline design choices, assumptions, limitations, and a roadmap including a sliding‑window multi‑date extension.

### DRAFT: Alternative Abstract (for review)

Remote sensing users need fast, transparent, and reproducible methods to highlight where the Earth’s surface has changed between two satellite acquisitions. We present a lightweight, interpretable Difference Subspace (DS) prototype for multi‑temporal Sentinel‑2 imagery that requires no deep learning, minimal compute, and produces actionable outputs (RGBs, change maps, histograms, principal angles). Each date is modeled as a low‑dimensional spectral subspace via PCA; canonical angles between dates yield a difference subspace that concentrates inter‑date spectral variation. Projecting per‑pixel spectral differences onto this subspace—and cross‑reconstructing between subspaces—provides two complementary change scores. The implementation is configuration‑driven (YAML), packaged for scripting and notebooks, and validated on synthetic and real tiles. A planned sliding‑window extension scales DS from bi‑temporal to multi‑temporal monitoring.





### DRAFT: Alternative Problem & Objective (for review)
- Problem. Given two co‑registered multispectral images of the same location at times t1 and t2, identify pixels whose spectra changed in a way that is (i) physically meaningful across bands, (ii) robust to nuisance variation (illumination and small radiometric shifts), and (iii) explainable without heavy training or labels.
- Objective. Deliver a transparent, reproducible DS baseline that:
  - Implements first‑order DS math cleanly and auditable in code (`src/multisenge_ds/diffsubspace.py`, `src/multisenge_ds/pca.py`).
  - Loads real Sentinel‑2 data reliably (`src/multisenge_ds/loader.py`) and runs from a single YAML config (`scripts/run_ds_demo.py`).
  - Produces interpretable outputs (RGB composites, DS projection and cross‑residual maps, histograms, principal angles) that practitioners can trust and extend.
  - Provides a forward path to multi‑temporal analysis (sliding windows), simple performance baselines, and modular integration (masking/thresholds/tiling).

### DRAFT: Why This Matters (for review)
- Operational: rapid triage for disasters, deforestation alerts, agriculture, urban change.
- Scientific: interpretable, geometry‑based descriptors (principal angles, subspace differences) that complement or seed ML models.
- Engineering: dependency‑light baseline that runs on Windows/PowerShell and in CI.

### DRAFT: Sharpening The Problem Statement (choose one anchor)
- Disaster mapping (floods/wildfires): within 24 hours, generate change‑intensity maps that correlate with damaged/charred areas at ≥0.70 AUROC.
- Deforestation alerts: flag canopy loss at 256×256 scale with ≥0.70 F1 against known logging; minimize phenology‑driven false positives.
- Urban expansion: detect impervious growth with ≥0.75 precision at 0.50 recall over quarterly intervals.

Suggested evaluation knobs: AUROC/AP and F1 at Otsu/percentile thresholds; baselines (ΔNDVI/ΔNBR, simple PCA‑difference); nuisance handling via cloud/shadow masks; seasonal spans; principal‑angle reporting for interpretability.
 Background: Difference Subspace

Let X₁, X₂ ∈ R^{N×B} be pixel spectra for two dates (N pixels, B bands).
After mean‑centering columns, PCA yields orthonormal bases Φ, Ψ ∈ R^{B×k}
(k ≤ B) for the dominant spectral directions at each date.

- Canonical angles: compute M = ΦᵀΨ; SVD M = U diag(s) Vᵀ, where s contains
  cosines of principal angles θ = arccos(s). Small θ implies shared
  directions; larger θ indicates change.
- Difference and common subspaces:
  - Raw difference columns: dᵢ ∝ Φuᵢ − Ψvᵢ, scaled by 1/√(2(1−sᵢ)).
  - Raw common (principal) columns: mᵢ ∝ Φuᵢ + Ψvᵢ, scaled by 1/√(2(1+sᵢ)).
  - Orthonormalize columns (QR) and drop numerically unstable cases when s≈1.

References: Fukui & Maki (2015), Fukui et al. (2024).

## 3. Scoring Change

- Projection energy (DS): For per‑pixel Δx = x₂ − x₁ (B‑vector), project onto
  D (difference basis) and score ∥Proj_D(Δx)∥². Implemented in
  `src/multisenge_ds/diffsubspace.py:projection_change_score`.
- Cross‑residual: Project x₁ onto Ψ and x₂ onto Φ and average residual norms:
  0.5(∥x₁−Proj_Ψ(x₁)∥² + ∥x₂−Proj_Φ(x₂)∥²). Implemented in
  `cross_residual_change_score`.
Both maps can be normalized to [0,1] for visualization.

## 4. Data & Assumptions

- Dataset: MultiSenGE Sentinel‑2: 256×256×10 patches; band order
  [B02,B03,B04,B08,B05,B06,B07,B8A,B11,B12].
- Scaling: DN/10 000 → reflectance (float32).
- Registration: per‑patch dates are co‑registered (same pixel grid); no cloud
  masking is applied by default.

See `docs/core/ASSUMPTIONS.md` for full details.

## 5. Implementation Overview (Code Map)

- Subspace math: `src/multisenge_ds/diffsubspace.py`
  - `canonical_angles(Φ,Ψ)` → U, s, V
  - `difference_subspace(Φ,Ψ)` → D, M, s, U, V
  - `projection_change_score`, `cross_residual_change_score`
- PCA: `src/multisenge_ds/pca.py`
  - `pca_basis(Z, n_components, center)` computes orthonormal components via
    covariance eigen‑decomposition (NumPy) or optional Torch.
- I/O: `src/multisenge_ds/loader.py`
  - `read_multiband_image(path, reflectance_scale)` supports `.npy` and
    GeoTIFF via rasterio; returns (H,W,B) float32.
  - `load_patch(patch_dir, date_indices, reflectance_scale)` returns two
    dates and metadata.
- Visualization: `src/multisenge_ds/vis/plots.py` (RGB composites, heatmaps,
  histograms, principal angles).
- CLI: `scripts/run_ds_demo.py`
  - Loads YAML (`demo_config.yaml` for synthetic, `real_s2_config.yaml` for
    real S2), executes the pipeline, saves outputs to `out_dir`.

## 6. Pipeline (CLI)

1. Load config (`cfg = yaml.safe_load(...)`).
2. If `data_root` not set → synthetic path:
   - `generate_synthetic_patch(shape, change_std, seed)` → X₁, X₂.
3. Else real path:
   - Build `s2_dir = data_root/s2`, parse `patch_id`, choose `spatial_coords`
     and `date_indices` (first/last by default).
   - Load two GeoTIFFs with `read_multiband_image` (DN→reflectance).
4. Flatten to (N,B), compute PCA bases: `pca_basis` → Φ, Ψ.
5. `difference_subspace(Φ,Ψ)` → D, cosines.
6. Change maps: `projection_change_score(X₁,X₂,D)` and
   `cross_residual_change_score(X₁,X₂,Φ,Ψ)`; normalize if requested.
7. Visuals: RGB composites (`rgb_bands`), heatmaps, histograms, angle bars.

## 7. Configuration (YAML)

- Synthetic (`demo_config.yaml`):
  - `synthetic_shape: [H,W,B]`, `synthetic_change_std`, `seed`.
  - `subspace_k`, `rgb_bands: [r,g,b]`, `out_dir`.
- Real (`real_s2_config.yaml`):
  - `data_root`, `s2_dir`, `patch_id`, `spatial_coords`, `date_indices`.
  - `reflectance_scale`, `subspace_k`, `rgb_bands`, `out_dir`.

## 8. Design Choices & Rationale

- Keep PCA linear and centered (no whitening by default) for stable angles and
  interpretability.
- Full SVD via NumPy on covariance; Torch optional for acceleration.
- Band order fixed to MultiSenGE convention; low `k` (e.g., 6) often removes
  noise while preserving salient variation.
- DS scoring: projection energy highlights aligned changes; cross‑residual is
  robust to basis mismatch; we export both.
- Minimal dependencies; rasterio optional, matplotlib for figures.

## 9. Validation & Outputs

- Synthetic verification confirms normalized maps in [0,1], no NaNs; D rank
  consistent with angle spectrum.
- Real verification (tile 32ULV, 20200601 vs 20201105, coords 257_6425) shows
  well‑formed outputs and reasonable principal angles.
- Artifacts: `t1_rgb.png`, `t2_rgb.png`, `ds_proj_map.png`,
  `ds_cross_residual_map.png`, `ds_*_hist.png`, `ds_principal_angles.png`.

## 10. Limitations

- No cloud/shadow mask; atmospheric effects may appear as change.
- Full SVD scales poorly on very large tiles; consider incremental/randomized
  SVD for bigger scenes.
- Bi‑temporal only in current CLI; multi‑date support planned.

## 11. Roadmap (Short‑Term)

- Sliding‑window DS (consecutive pairs) with metrics CSV/JSON, time‑series
  plots, and optional GIFs. See `docs/design/SLIDING_WINDOW_DESIGN.md`.
- Smoke/regression tests (deterministic synthetic & small S2 sample).
- Packaging (`requirements.txt` or `pyproject.toml`), CI hooks.

## 12. Reproducibility & Usage

- Environment:
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\activate
  python -m pip install --upgrade pip
  python -m pip install numpy matplotlib pyyaml rasterio
  ```
- Synthetic run:
  ```powershell
  $env:PYTHONPATH='src'
  python scripts\run_ds_demo.py --config demo_config.yaml
  ```
- Real run:
  ```powershell
  $env:PYTHONPATH='src'
  python scripts\run_ds_demo.py --config real_s2_config.yaml
  ```

## 13. Maintenance of This Document

This file is a living technical report. Any substantive code change that
alters assumptions, math, interfaces, or outputs must be reflected here and
in `docs/design/ROADMAP.md` (workflow requirement). Include a brief summary of
what changed, why, and any migration notes for configs.



