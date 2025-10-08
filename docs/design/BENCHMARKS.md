# Benchmarks & Metrics (Design)

This note formalizes baseline methods and evaluation metrics for DS comparison.

## Baseline Methods

Let bands be ordered: [B02,B03,B04,B08,B05,B06,B07,B8A,B11,B12] (indices 0..9).

- NDVI = (B08 − B04) / (B08 + B04); ΔNDVI = NDVI(t2) − NDVI(t1)
- NBR  = (B08 − B12) / (B08 + B12); ΔNBR  = NBR(t2)  − NBR(t1)
- NDWI = (B03 − B08) / (B03 + B08); ΔNDWI = NDWI(t2) − NDWI(t1)
- MNDWI = (B03 − B11) / (B03 + B11); ΔMNDWI = MNDWI(t2) − MNDWI(t1)
- ΔNIR = B08(t2) − B08(t1); ΔSWIR = B12(t2) − B12(t1)
- CVA (Change Vector Analysis): ∥Δx∥₂; optionally standardize by per‑band σ pooled across t1/t2.
- PCA‑difference: fit PCs on Z = [Z1; Z2], project Δx onto top‑k PCs, score ∥Proj_PC(Δx)∥².
- (Stretch) MAD / IR‑MAD: CCA‑based difference transforms; iterative reweighting for robustness.

## Metrics

- Pixel/patch classification: ROC‑AUC, PR‑AUC, F1 at Otsu/percentiles; IoU if masks exist.
- Operational: area above threshold (%), time‑to‑detection (sliding windows), runtime, memory.
- Interpretability: DS rank; principal‑angle histogram; ΔNBR severity correlation (wildfires).

## Thresholds & Fairness

- Report standard thresholds (Otsu, 95th percentile) for each method.
- Apply identical masks to all methods; compute metrics only on valid pixels.

## Implementation Plan

- Module `src/multisenge_ds/benchmarks.py` with:
  - `delta_ndvi/…/delta_mndwi`, `delta_nir`, `delta_swir`
  - `cva_magnitude(X1,X2, standardize=False)`
  - `pca_difference_energy(X1,X2, k=None)`
- Optional: metrics helpers under `eval/` or re‑use existing plotting utils for histograms.
