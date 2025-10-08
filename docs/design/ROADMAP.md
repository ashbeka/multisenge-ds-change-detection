# Project Roadmap

This roadmap is the master plan for the MultiSenGE Difference Subspace project. It combines all planning notes and serves as the reference when scheduling work, fixing issues, or proposing new features.

## Baseline Checks

- **Toolchain**: system Python 3.13.2; project virtualenv `.venv` on Python 3.12.3.
- **Dependencies**: `numpy`, `matplotlib`, `pyyaml`, `rasterio==1.4.3` installed in the venv; optional `torch` for GPU acceleration.
- **Launch**: VS Code configs set `PYTHONPATH=${workspaceFolder}/src`; scripts also auto-append `src` so direct CLI runs work.

## Project Diagnosis

- DS core math implemented in `src/multisenge_ds/diffsubspace.py`; PCA utilities in `src/multisenge_ds/pca.py`.
- Data access via `src/multisenge_ds/loader.py` (`.npy` native, GeoTIFF via rasterio).
- CLI entrypoint `scripts/run_ds_demo.py` supports synthetic + real bi-temporal runs.
- Visual outputs produced by `src/multisenge_ds/vis/plots.py`.
- Documentation reorganized under `docs/` (index, core specs, design notes, guides, examples, references).

## Action Roadmap

1. **Environment readiness**
   - Capture dependencies in a requirements/pyproject file and document optional extras (torch).
   - Evaluate packaging strategy (editable install vs. PYTHONPATH) as the project matures.
2. **Baseline hardening**
   - Exercise `load_s2_files_by_date_pattern` with more real tiles; add robust logging and shape/band assertions.
   - Add lightweight smoke tests: deterministic synthetic case + small S2 sample (kept outside repo).
3. **Sliding-window feature**
   - Implement design in `docs/design/SLIDING_WINDOW_DESIGN.md`: window enumeration, metrics outputs, config schema, GIF option.
   - Ensure backward compatibility when `sliding_window.enabled` is false.
4. **Visual analytics**
   - Add time-series plots, optional pairwise heatmap, animated DS map (GIF/MP4) exports.
5. **Performance & scalability**
   - Profile PCA/SVD for larger windows; explore incremental/rand SVD and optional PyTorch on GPU.
6. **QA / Testing**
   - Build deterministic synthetic fixtures, regression tests for PCA & DS routines, CLI integration harness.

## Branch Ideas

- `feature/sliding-window-ds`: multi-date DS workflow + metrics.
- `feature/data-ingest`: filename parsing robustness, metadata exports, better error reporting.
- `feature/visualization-suite`: richer plots, animations, notebook upgrades.
- `chore/package-structure`: finalize packaging (pyproject, linting, CI).
- `docs/research-alignment`: ADRs, experiment templates, research traceability.

## Documentation Strategy

- Maintain living docs (spec, runbook, assumptions, data management, design notes, roadmap) via `docs/README.md` index.
- Add CHANGELOG or ADRs for major decisions (e.g., windowing strategy, scaling choices).
- Provide Windows-first quickstart with cross-platform notes; include dependency commands (`pip install rasterio matplotlib pyyaml`).
- Prepare GitHub-ready files before release: updated README (with images from outputs), CONTRIBUTING, issue/PR templates.

## Implementation Playbook (Step-by-Step)

Use this short list to keep work items clear for both humans and future LLM agents. Move through one item at a time; after each item, re-run the relevant demo(s), record results, and update the change log.

1. **Benchmark integration (current focus)**
   - Inputs: YAML keys `benchmarks`, `metrics_mask`, `thresholds`, `metrics_out`, `save_benchmark_maps`.
   - Outputs: DS + baseline maps, metrics CSV/JSON (ROC-AUC, PR-AUC, F1@Otsu/p95/Youden, runtime).
   - Required code modules: `scripts/run_ds_demo.py`, `src/multisenge_ds/eval/benchmarks.py`, metrics helpers (new if needed).
   - Verification: run synthetic + real configs; confirm CSV exists when a mask is provided; log runtimes.
2. **Sliding-window DS**
   - Enumerate consecutive pairs, reuse benchmark scorers, emit per-window CSV (mean/max/percentiles) and optional GIF.
   - Metrics: time-to-detection, area above threshold per window.
3. **Smoke/regression tests**
   - Deterministic synthetic fixture (tiny array assertions) + tiny real patch sanity test.
   - Add to CI/local scripts once packaging ready.
4. **Packaging & release prep**
   - `requirements.txt` / `pyproject.toml`, lint configuration, GitHub repo initialization, documentation polish.

## Exploration Tracks (Options & Focal Paths)

- Primary recommendation: Rapid disaster damage triage on Sentinel‑2 using Difference Subspace (DS); extend to multi‑temporal monitoring with sliding windows.

- Alternative tracks (keep on radar):
  - Deforestation/vegetation loss monitoring (handle phenology; multi‑date trend focus).
  - Urban expansion/infrastructure change (impervious growth; quarterly summaries).
  - Agricultural health/crop rotation (phenology cycles; field‑scale metrics).

### Scope & Deliverables (Phase 1)
- Problem statement (TECH_REPORT): “Rapid, interpretable triage of flood/wildfire damage from Sentinel‑2 using DS.” Add a “Use‑Case Focus” box with metrics and baselines.
- Event curation: 2–3 real events (flood/wildfire), with pre/post dates; record tile/date/coords; fetch public footprints where available (e.g., Copernicus EMS, burned‑area products).
- Metrics:
  - Pixel/patch ROC–AUC, PR–AUC, F1 at Otsu/percentiles; report principal angles to interpret subspace shifts.
  - Aggregate DS scores within polygons (when footprints exist) for region‑level evaluation.
- Baselines to compare:
  - ΔNDVI, ΔNBR, simple band differencing (ΔNIR, ΔSWIR).
  - Simple PCA‑difference without canonical‑angle normalization (sanity comparator).
- Robustness features:
  - Optional cloud/shadow masking; reflectance normalization checks; per‑band percentile clipping.
- Outputs per event:
  - t1/t2 RGBs, DS projection and cross‑residual maps, histograms, principal‑angle bars, concise metrics report.

### Phase 2 (Immediate Next)
- Sliding windows: DS per window; time‑series metrics; optional GIF; CSV/JSON summaries.
- Minimal tests: deterministic synthetic fixtures + tiny real S2 sample; NaN‑free checks; stable metrics.
- Packaging: requirements/pyproject; basic CI; prep GitHub.

### Stretch Goals (Later)
- Second‑order DS; SSC + U‑Net integration (when labels available); MCDA skeleton for priority ranking.

### Baselines & Evaluation Summary
- Band/index deltas (ΔNDVI/ΔNBR/ΔNIR/ΔSWIR); PCA‑difference; optional CVA; supervised references (xBD/xView2/U‑Net) later.
- Metrics: ROC‑AUC, PR‑AUC, F1 (Otsu/percentiles), area‑above‑threshold, time‑to‑detection, principal‑angle distributions.

### Risks & Mitigations
- Labels scarce → start with proxy metrics; add public footprints later.
- Clouds/smoke → masking hooks; document limitations; report cloudiness.
- Sensor mismatch → defer; keep DS outputs stable for future ML.

### Decision Needed
- Select anchor focus; choose 2–3 events; approve TECH_REPORT “Use‑Case Focus”.

## Release Checklist (first public iteration)

- [ ] Install required dependencies in `.venv` (including `rasterio`) and validate synthetic + real S2 runs.
- [ ] Ensure VS Code launch configs run both demos successfully.
- [ ] Confirm `.gitignore` keeps datasets/outputs out of version control.
- [ ] Produce baseline outputs and document them in README or docs.
- [ ] Review all docs for consistency (module paths, assumptions, run instructions).
- [ ] Initialize git, craft initial commit, and push to GitHub (`multisenge-ds-change-detection`).

## Workflow Requirement

For every development task:
1. Execute the change.
2. Verify the result (tests, runs, or inspection as appropriate).
3. Update this roadmap with any plan adjustments and log what changed relative to the prior iteration.

This checklist must be followed and documented before considering the task complete.

## Change Log
- 2025-09-20: Documented step-by-step implementation playbook (benchmarks → sliding windows → tests → packaging) for clarity and LLM/human handoff.
- 2025-09-20: Refined run_ds_demo.py typing (config parsing, index casts) to satisfy Pylance Problems and keep runtime behavior unchanged.

- 2025-09-20: Added Exploration Tracks section (options, baselines, metrics, risks) to capture alternate paths without changing core plan.

- 2025-09-20: Added VS Code python.analysis.extraPaths='src' to fix Problems tab import warnings.
- 2025-09-20: Updated CLI defaults so running without --config falls back to demo_config.yaml; README reflects the behavior.
- 2025-09-20: Repository reorganized into `src/multisenge_ds/`, docs restructured (`docs/core`, `docs/design`, `docs/guides`, `docs/examples`), synthetic demo outputs archived under docs, README and launch configs updated to reference new layout.
- 2025-09-20: Structural audit (pre-reorg) recommended moving `ds/` and `vis/` into a package, archiving `synthetic_demo_outputs/`, and reorganizing docs.









