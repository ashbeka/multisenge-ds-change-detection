# Sliding-Window Difference Subspace Design

This document outlines the proposed design for extending the Difference Subspace (DS) prototype from bi-temporal analysis to multi-temporal, sliding-window change detection on Sentinel-2 patches.

## Objectives

- Support DS processing across ordered acquisition dates for a fixed spatial coordinate (tile + x_y suffix).
- Produce per-window change maps, summary metrics, and aggregate visualizations without duplicating the existing bi-temporal pipeline logic.
- Preserve reproducibility by documenting inputs (dates, bands, PCA dimension) and outputs for each window.

## Data Model

- **Input cube**: aligned Sentinel-2 tiles for one spatial coordinate. Each tile is a 256?256?10 array in the documented band order (B02,B03,B04,B08,B05,B06,B07,B8A,B11,B12).
- **Date ordering**: extracted from filenames; sorted chronologically using the `yyyymmdd` token.
- **Window definition**: consecutive pairs of dates by default. Future options include stride > 1 or window length > 2.

## API Sketch

```
def list_available_dates(data_root: Path, tile_prefix: str, spatial_coords: str | None = None) -> list[PatchInfo]

def run_sliding_windows(
    images: list[Path],
    window: int = 2,
    stride: int = 1,
    k: int = 6,
    score: str = "proj",
    out_dir: Path,
    make_gif: bool = False,
) -> SlidingWindowResult
```

- `PatchInfo` records date, filepath, and optional metadata.
- `SlidingWindowResult` aggregates per-window outputs (file paths, metrics, logs).

## Processing Steps per Window

1. Load two (or more) dates using existing loader utilities.
2. Compute PCA bases with consistent `subspace_k` and centering.
3. Derive DS basis via `difference_subspace`.
4. Generate change maps (projection energy + optional cross-residual).
5. Persist artifacts under a window-specific directory (`out_dir/date_i__date_j/`).
6. Compute summary statistics: max/mean score, percentile threshold suggestions.

## Aggregate Outputs

- **Time series**: CSV/JSON with per-window metrics (dates, mean/max scores, thresholds).
- **Plot**: matplotlib line chart of mean/max scores vs. window index.
- **Pairwise heatmap** (optional advanced mode): if all pair combinations requested.
- **Animation** (when `make_gif=True`): GIF of DS projection maps ordered by time.

## Configuration Additions

Extend YAML schema:

```yaml
sliding_window:
  enabled: true
  window: 2
  stride: 1
  score: proj  # proj | cross | both
  make_gif: false
  metrics_out: outputs/real_s2_demo/window_metrics.csv
  include_cross_residual: false
  aggregate_plots: true
```

- Default behaviour preserves current bi-temporal run when `sliding_window.enabled` is false or omitted.
- When enabled, `date_indices` is ignored; the script derives ordered windows automatically.

## Logging & Validation

- Emit clear console log per window with file names, principal angles summary, and basic stats.
- Assert consistent shapes/band ordering; fail fast if discrepancies detected.
- Optionally cache the PCA means/variance to allow reuse when windows overlap heavily (future optimisation).

## Testing Strategy

- Synthetic multi-date fixture: generate random walk changes across ?4 dates; compare scores to known pattern.
- Small real patch sample: run sliding window and verify output directories & metrics.
- Unit tests for helper functions (date parsing, window enumeration).

## Open Questions

- Should we support dynamic reference windows (rolling baseline vs. all previous images)?
- How to handle missing dates or clouds (need masks/metadata)?
- Do we expose partial-windows at the start/end when `window > 2`?

Once this design is approved, implementation can proceed on the `feature/sliding-window-ds` branch.
