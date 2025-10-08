# Data Management Guidelines

This project depends on large Sentinel-2 tiles from the MultiSenGE dataset. The complete archive easily exceeds 100 GB, so we **do not commit raw data** to version control. Instead, follow the practices below.

## Local storage

- Keep raw tiles outside the repository (for example, `D:/data/MultiSenGE/s2`).
- Adjust the `data_root` entry in your YAML configs (`real_s2_config.yaml` or custom files) to point at your local data directory.
- If you prefer to keep a lightweight pointer in the repo, create a symlink named `s2` that targets your external storage. The `.gitignore` entry prevents the actual data from being tracked.

## Minimal working set

- For code review or quick smoke tests, maintain a **tiny sample** (e.g. a single tile with two dates) in a separate folder such as `sample_data/`. Keep the sample under 50 MB so it can be shared if needed.
- Note the source and preprocessing of any shared samples in this document to preserve provenance.

## Reproducibility notes

- Document the exact MultiSenGE release (date/hash) and any preprocessing steps in experiment logs or notebooks.
- Include statistical summaries (band means/standard deviations) in reports when sharing results without the raw data.

## Cloud or remote storage

- If you need to collaborate, place the dataset on shared storage (cloud bucket, NAS, etc.) and update `data_root` accordingly.
- Avoid placing credentials in config files; rely on environment variables or credential managers instead.

Following these guidelines keeps the repository lightweight for GitHub while ensuring experiments remain reproducible.
