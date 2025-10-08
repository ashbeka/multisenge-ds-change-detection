Runbook for the Difference Subspace Prototype

This runbook provides practical instructions for running the DS prototype, along with a record of decisions and open questions. Keeping this log up to date helps track changes to the code and assumptions.

Environment setup

Create a Python virtual environment in the project root:

python -m venv .venv
source .venv/bin/activate  # Windows users: .venv\Scripts\activate
pip install --upgrade pip
pip install numpy scipy matplotlib rasterio rioxarray tqdm pyyaml


Set the PYTHONPATH so that modules under src/ can be imported:

export PYTHONPATH=src:${PYTHONPATH}


You can add this to a .env file or your shell configuration for convenience.

Running the synthetic demo

The repository includes a synthetic demonstration script that does not require real data. To verify that the DS code functions correctly, run:

python scripts/run_ds_demo.py --config demo_config.yaml


This will create an outputs/synthetic_demo/ directory with the following files:

t1_rgb.png and t2_rgb.png – simple synthetic RGB images used for demonstration.

ds_proj_map.png and ds_xresid_map.png – projection energy and cross‑residual change maps.

ds_proj_hist.png – histogram of projection scores.

ds_principal_angles.png – bar plot of canonical angles.

Processing real MultiSenGE S2 data

To run the DS prototype on actual Sentinel‑2 data from MultiSenGE:

Place your extracted MultiSenGE S2 patches in a directory such as /data/MultiSenGE/s2/. Each patch directory should contain multi‑band GeoTIFFs or band‑stacked arrays for each date.

Create a configuration file (for example config.yaml) based on the template below:

data_root: /data/MultiSenGE
s2_dir: s2
patch_id: PATCH_XYZ
timestamps: ["2018-04-01", "2018-07-15"]
rgb_bands: [2, 1, 0]  # B04,B03,B02 in the band list
subspace_k: 6
change_score: proj_energy  # or residual_norm
out_dir: outputs/PATCH_XYZ
use_cloud_mask: false


Adjust patch_id and timestamps to match your patch directory and the available dates. rgb_bands can be modified if you wish to view different colour composites.

Run the DS demo script with your configuration:

python scripts/run_ds_demo.py --config config.yaml


The script will load the two specified Sentinel‑2 images, compute the DS basis, generate change maps, and save them to the directory specified by out_dir.

Decisions log

This section records important decisions made during development:

Band order: We assume that each Sentinel‑2 image stores bands in the order B02,B03,B04,B08,B05,B06,B07,B8A,B11,B12. This order is based on the MultiSenGE documentation and is used by the loader.

Scaling: Pixel values are divided by 10 000 to convert them to reflectance. This is consistent with Sentinel‑2 Level‑2A convention.

No cloud mask: MultiSenGE does not provide per‑pixel cloud or shadow masks, so our prototype processes all pixels. Cloud handling may be added later if appropriate masks become available.

Full subspace dimension: The default uses all ten bands (subspace_k=10). For noisy data, using fewer components (e.g. 6–8) can improve robustness.

Scoring method: Initial testing uses the projection energy score. The residual‑based score can be enabled by setting change_score to residual_norm in the configuration.

Please append additional decisions as they arise.

Open questions

Temporal alignment: Should we apply Randomised Time Warping (RTW) or another alignment technique to the time series before computing DS? For now, this is considered optional and left for later investigation.

Cloud masking: How should we handle clouds or missing data? If masks become available, they can be applied prior to subspace computation.

Threshold selection: What is the best way to threshold change scores? Future work could involve Otsu’s method, percentiles, or a learned threshold.

Add further questions here as issues are identified.