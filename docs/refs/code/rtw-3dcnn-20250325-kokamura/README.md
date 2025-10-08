# RTW with 3DCNN(X3D)

These scripts implement the Randomized Time Warping (RTW) techniques for action recognition research.
`VideoLoaderFactory` class in `features.py` is used to generate features.

※ These scripts require additional preparations, including dataset setup (Something-Something v2), training of the X3D model, feature extraction, and mutual subspace model fitting. So you may not be able to execute the scripts.

<br />

> In my experiment, executed the following procedure.

1. Prepare the dataset (Something-Something v2)
2. Train X3D model (`train_x3d.py`)
   - Here, use `sample_frames()` in `dataset.py`, but only 1 sample on training 3dcnn, so it is not RTW.
   - It is just random sampling with temporal sequence.
3. Extract features from the dataset (`extract_features.py`)
4. Fit mutual subspace model (`mutual_subspace.py`)
   - 3 and 4 are generated features 1000 times and fit mutual subspace model.
   - Something-Something v2 is a large dataset, so in my experiment, save features as npy files locally after extracting features. Then, fit mutual subspace model with npy files.
