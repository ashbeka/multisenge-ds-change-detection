# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Machine Learning research project with two main components:

### 1. Bump Detection in Floor Surfaces (Main Focus)
The primary goal is detecting tiny bumps in floor surfaces using shape and difference subspace methods:
- Started with synthetic uniform point clouds (10,000 points) simulating flat floors with artificial bumps
- Now using real .ply files which contain more noise than synthetic data
- Uses patch-based analysis to detect local geometric anomalies

### 2. Reference Implementation (src/ folder)
The src/ folder contains reference implementations of subspace methods from video action recognition:
- Used as a reference for subspace generation techniques
- Implements various Mutual Subspace Methods (MSM, CMSM, etc.)
- **Important**: When implementing new subspace-related code, refer to src/cvt_custom/ for mathematical approaches

## Development Commands

### Environment Setup
```bash
# Install dependencies using UV (recommended)
uv pip install -e .

# Or install specific dependencies
uv pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118
```

### Code Quality
```bash
# Run linting (configured in ruff.toml)
ruff check .

# Format code
ruff format .
```

### Running the Application
```bash
# Point cloud processing
python main.py

# Test X3D model
python src/test_x3d.py
```

## Architecture Overview

### Bump Detection Pipeline (Main Project)
- **main.py**: Core entry point for bump detection in floor surfaces
  - `generate_subspace()`: Creates subspace from patch using SVD on centered data
  - `gen_shape_difference_subspace()`: Computes difference subspace between two patches
  - `calculate_patch_differences_with_ds()`: Main detection using difference subspace projection
  - Supports both 4-neighbor and 8-neighbor analysis
- **pointlib.py**: Utility functions for point cloud operations
- **grid_patch_division_with_stride.py**: Alternative patch division with overlapping patches
- **config.py**: Configuration parameters (subspace dimensions, patch sizes, etc.)

### Reference Subspace Implementations (src/)
- **src/cvt_custom/utils/base.py**: Core `subspace_bases()` function using PCA
- **src/cvt_custom/models/**: Various subspace method implementations
  - `msm.py`: Mutual Subspace Method using canonical angles
  - `cmsm.py`: Constrained MSM with Generalized Difference Subspace
  - Kernel versions (KMSM, KCMSM) for non-linear representations

### Detection Methodology
1. **Patch Division**: Floor point cloud divided into grid of patches (e.g., 10×10)
2. **Subspace Generation**: Each patch → PCA/SVD → shape subspace representation
3. **Difference Subspace**: Between neighboring patches, captures geometric differences
4. **Anomaly Detection**: Project patch onto difference subspace, measure magnitude
5. **Visualization**: Heatmap showing anomaly scores across patch grid

### Key Configuration
- Model and training parameters are in `config.py`
- Uses PyTorch Lightning for training orchestration
- Supports Weights & Biases (wandb) for experiment tracking

## Important Notes
- The project uses UV as the package manager (not pip directly)
- CUDA 11.8 is required for GPU support
- Point cloud files should be in .ply format for real data
- Synthetic data generated with `generate_synthetic_floor()` for testing
- Default patch grid: 10×10, configurable via `n_patches_x` and `n_patches_y`
- Subspace dimension typically set to 3 (configurable in config.py)