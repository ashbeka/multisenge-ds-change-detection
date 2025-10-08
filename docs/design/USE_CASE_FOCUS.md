# Use‑Case Focus (DRAFT): Disaster Damage Triage with DS

This document anchors the project’s current focus while keeping other applications viable.

## Problem
Given two co‑registered Sentinel‑2 acquisitions (t1, t2) over a disaster area (flood or wildfire), highlight pixels whose spectra changed in a physically meaningful, interpretable way, robust to nuisance variation, and without supervised training.

## Method (Summary)
- PCA on each date → subspaces Φ, Ψ (k components).
- Canonical angles of ΦᵀΨ → cosines s; build difference basis D.
- Change scores: projection energy ∥Proj_D(Δx)∥² and cross‑residual.

## Baselines
- Index deltas: ΔNDVI, ΔNBR (fire), ΔNDWI/MNDWI (flood), ΔNIR, ΔSWIR.
- Vector: CVA magnitude ∥Δx∥ (optionally standardized); PCA‑difference (pooled PCs, no canonical‑angle normalization).
- (Optional later) MAD/IR‑MAD; supervised references (xBD/xView2/U‑Net) when labels available.

## Metrics
- Pixel/patch: ROC‑AUC, PR‑AUC, F1 at Otsu/percentiles; IoU when segmentation masks exist.
- Operational: area‑above‑threshold, time‑to‑detection with sliding windows, runtime and memory.
- Interpretability: DS rank, principal‑angle spectrum; correlation of DS with ΔNBR severity (wildfire).

## Protocol
- Event curation: 2–3 floods/wildfires; record tile/date/coords; fetch public footprints (Copernicus EMS, burned‑area products) when possible.
- Preprocessing: reflectance scaling (÷10000), optional cloud/shadow masking; identical masks for all methods.
- Thresholds: report Otsu, 95th percentile, and Youden’s J for fairness.

## Outputs per Event
- RGB t1/t2, DS maps (proj/cross‑resid), index‑delta maps, histograms, principal‑angles, per‑event metrics (CSV/JSON), qualitative panels.

This focus is compatible with future tracks (deforestation, urban, agriculture) using the same benchmarking framework.
