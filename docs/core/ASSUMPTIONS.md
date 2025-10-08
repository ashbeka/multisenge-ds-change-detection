Assumptions for the DS Prototype

This file lists the assumptions made while implementing the Difference Subspace (DS) prototype. Recording these assumptions helps identify potential sources of error and guides future refinements.

Data and pre‑processing

Band order – Sentinel‑2 images in MultiSenGE S2 subsets are stored with the ten bands in the following order: B02,B03,B04,B08,B05,B06,B07,B8A,B11,B12. The loader reads the bands in this order.

Reflectance scaling – Input values are surface reflectance scaled by 10 000. They are converted to float32 and divided by 10 000 to yield approximate reflectance values between 0 and 1.

No cloud or shadow masks – No per‑pixel cloud mask is applied because MultiSenGE does not provide one. All pixels are treated as valid. This may result in change detections on cloudy areas.

Spatial alignment – All images in a patch are assumed to be perfectly co‑registered. Pixel (i,j) in the first date corresponds to the same physical location as pixel (i,j) in the second date.

Patch size – Each patch is 256×256 pixels at 10 m resolution, covering approximately 1.2 km × 1.2 km.

Number of dates – Each patch may contain multiple acquisition dates. The prototype selects two dates at a time for change detection. No time‑series modelling (e.g. alignment or interpolation) is performed.

Subspace computation

Principal component dimension – By default, all bands are used to form the subspaces (subspace_k = 10). Lower values may be used to suppress noise.

Mean centring – Each band is centred by subtracting its mean before PCA. Optionally bands could be standardised by dividing by the standard deviation, but this is not currently implemented.

Orthogonality – The PCA bases are assumed to be orthonormal. Computation uses numerical SVD, which yields nearly orthogonal bases. Tiny deviations are ignored.

Singular value cut‑off – In computing the difference subspace, singular values extremely close to 1 or 0 are ignored to avoid numerical instability. This means that common or null directions may be dropped from the difference basis.

Change scoring

Projection energy vs. residual – The prototype supports two scoring methods: projection energy and cross‑residual. Projection energy highlights changes aligned with the major difference directions. Residual highlights pixels that do not fit well into the other date’s subspace.

Thresholding – No thresholding is baked into the code. Interpreting the change map is left to the user or subsequent processing (e.g. Otsu’s method or percentile thresholds).

Interpretation – High change scores indicate spectral change, which can result from actual land‑cover changes, sensor differences, or atmospheric conditions. Without ground truth, scores should be interpreted cautiously.

The assumptions above are current as of the initial implementation. Should new evidence or requirements arise (for example, availability of per‑pixel quality masks or changes in band order), update this file accordingly.New-Item -ItemType