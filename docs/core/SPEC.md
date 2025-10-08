Difference Subspace (DS) Prototype Specification

This document summarises the Difference Subspace (DS) method for change detection in multi‑temporal Sentinel‑2 imagery and describes how it will be applied to the MultiSenGE dataset. Capturing these details in a specification keeps the rationale and assumptions clear as development proceeds.

Overview

The DS prototype detects changes between two images of the same geographic patch acquired on different dates. Each patch is a 256×256 pixel tile covering roughly 1.2 km × 1.2 km. Sentinel‑2 provides multiple spectral bands per pixel; in MultiSenGE ten bands are included. By modelling each image as a subspace in spectral space, DS computes directions that differentiate the pre‑event and post‑event subspaces. Change scores for each pixel indicate how strongly the pixel’s spectral difference aligns with those directions.

Data Format

Patch directory – holds several Sentinel‑2 images (multi‑band GeoTIFFs or stacked arrays) for one location at different dates. Filenames encode the patch ID and acquisition date.

Spatial size – 256 rows × 256 columns (approx. 1.2 km² at 10 m resolution).

Bands – ten bands in this order: B02 (Blue, 490 nm), B03 (Green, 560 nm), B04 (Red, 665 nm), B08 (NIR, 842 nm), B05 (Red Edge 1, 705 nm), B06 (Red Edge 2, 740 nm), B07 (Red Edge 3, 783 nm), B8A (NIR narrow, 865 nm), B11 (SWIR 1, 1610 nm), B12 (SWIR 2, 2190 nm).

Scaling – values are scaled surface reflectance (typically integers divided by 10 000). Convert to float and divide by 10 000 to obtain reflectance values in the range 0–1.

Registration – images within a patch are spatially aligned. No additional alignment is necessary.

Subspace Construction

For two dates, flatten each image to a matrix of shape (N, B), where N = 256 × 256 (one row per pixel) and B = 10 (one column per band). Standardise each column by subtracting its mean (and optionally dividing by the standard deviation). Apply Principal Component Analysis to each matrix to obtain an orthonormal basis representing the major spectral variance for that date. Using all ten bands produces a basis of dimension 10; a smaller number can be chosen to reduce noise.

Canonical angles

To measure how the two subspaces differ, compute the singular values of the matrix ΦᵀΨ where Φ and Ψ are the basis matrices for the first and second images. If σ is a singular value, the corresponding principal angle is given by arccos(σ). Small angles mean the subspaces share similar directions; large angles indicate that the directions are distinct.

Difference and common subspaces

From the singular value decomposition of ΦᵀΨ, obtain matrices U, Σ, and V. Build raw difference vectors by subtracting the matching basis vectors: diff = ΦU – ΨV. To normalise these vectors, scale each column by the reciprocal of sqrt(2 (1 – σ)) for the corresponding singular value σ. The resulting columns form an orthonormal basis for the difference subspace – the part of spectral space unique to one date or the other. A similar construction using sums ΦU + ΨV with a scaling of sqrt(2 (1 + σ)) yields a basis for the common subspace, which represents directions present in both images. In this prototype we focus on the difference subspace for change detection.

Change scoring methods

Once a basis for the difference subspace is available, compute a change score per pixel using one of two methods:

Projection energy score – For each pixel, compute the difference between its spectral vectors at the two dates (Δx = x₂ – x₁). Project this difference onto the difference subspace and compute the squared Euclidean norm of the projection. Large values indicate changes aligned with the dominant difference directions.

Cross‑residual score – For each pixel, project its spectral vector at one date onto the subspace of the other date and measure the reconstruction error. Let x̂₂ be the projection of x₂ onto the first image’s subspace and x̂₁ the projection of x₁ onto the second image’s subspace. Compute r₁ = ∥x₂ – x̂₂∥ and r₂ = ∥x₁ – x̂₁∥. The residual score is (r₁² + r₂²)/2. Pixels that do not fit well within either subspace are likely to have changed.

These scores form a change intensity map over the patch. A threshold (for example, Otsu’s method or a percentile cut) can convert the continuous map to a binary change mask.

Parameters and configuration

The prototype exposes several parameters:

subspace_k – number of principal components retained when constructing the subspaces. 10 uses all bands; lower values reduce noise.

change_score – method for scoring changes: proj_energy (projection energy) or residual_norm (cross‑residual).

rgb_bands – list of band indices used for RGB composites. The default is [2, 1, 0] corresponding to red, green, blue bands in the order above.

use_cloud_mask – whether to apply a cloud or no‑data mask. MultiSenGE does not provide masks, so this is false by default.

Expected outputs

Executing the DS pipeline on a pair of Sentinel‑2 images produces the following artefacts:

An RGB composite for each date, generated from bands B04, B03, and B02 (red, green, blue) for qualitative inspection.

A change map showing the projection energy or cross‑residual scores. High intensities indicate likely change.

A histogram of change scores to assist in selecting a threshold.

A principal angle plot summarising the canonical angles between the two subspaces.

These outputs help visualise and assess the detected changes. For quantitative evaluation, ground‑truth change maps would be required, which are not currently included in MultiSenGE.