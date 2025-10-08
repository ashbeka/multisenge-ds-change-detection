import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig, svd, eigh
from config import Confing
import open3d as o3d
import pandas as pd
import pointlib as Pointlib
from mpl_toolkits.mplot3d import Axes3D
import scipy
import os
from datetime import datetime
import sys


class Logger:
    """Simple logger to write to both console and file."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()

class Confing:
    def __init__(self):
        self.subspace_dim = 3

def generate_subspace(data_segment, cfg):
    """Generate a subspace from a segment using SVD."""
    if len(data_segment) < cfg.subspace_dim + 1:
        return np.zeros((data_segment.shape[0], cfg.subspace_dim))

    X = data_segment
    mv = np.mean(X, axis=0)
    X_centered = X - mv
    
    U, s, Vt = svd(X_centered, full_matrices=False)
    
    return U[:, 0:3]

def gen_shape_difference_subspace(S1, S2, cfg):
    """Generate the Difference Subspace (DS) between two subspaces."""
    n_rows = max(S1.shape[0], S2.shape[0])
    
    if S1.shape[0] < n_rows:
        S1_padded = np.zeros((n_rows, S1.shape[1]))
        S1_padded[:S1.shape[0], :] = S1
        S1 = S1_padded
    
    if S2.shape[0] < n_rows:
        S2_padded = np.zeros((n_rows, S2.shape[1]))
        S2_padded[:S2.shape[0], :] = S2
        S2 = S2_padded
    
    G = S1 @ S1.T + S2 @ S2.T
    eigen_val, eigen_vec = eig(G)
    idx = np.where((1e-6 < eigen_val) & (eigen_val < 1))[0]
    return eigen_vec[:, idx]

def calculate_subspace_distance(S1, S2):
    """Calculate the Frobenius norm of the difference of projection matrices."""
    n_rows = max(S1.shape[0], S2.shape[0])
    
    if S1.shape[0] < n_rows:
        S1_padded = np.zeros((n_rows, S1.shape[1]))
        S1_padded[:S1.shape[0], :] = S1
        S1 = S1_padded
    
    if S2.shape[0] < n_rows:
        S2_padded = np.zeros((n_rows, S2.shape[1]))
        S2_padded[:S2.shape[0], :] = S2
        S2 = S2_padded

    if S1.shape[1] == 0 or S2.shape[1] == 0:
        return 0.0
    
    try:
        P1 = S1 @ S1.T
        P2 = S2 @ S2.T
        return np.linalg.norm(P1 - P2, 'fro')
    except np.linalg.LinAlgError as e:
        print(f"LinAlgError in calculate_subspace_distance: {e}")
        return 0.0
    except Exception as e:
        print(f"Error in calculate_subspace_distance: {e}")
        return 0.0

def extract_patches_with_stride(points, patch_size=(50, 50), stride=(25, 25)):
    """
    Extract patches from point cloud using stride-based approach like convolution.
    
    Args:
        points: numpy array of shape (N, 3) containing point cloud data
        patch_size: tuple (patch_width, patch_height) defining patch dimensions
        stride: tuple (stride_x, stride_y) defining step size between patches
        
    Returns:
        list of patches, each containing subset of points
        patch_centers: list of (x, y) coordinates for patch centers
        patch_grid_info: dict containing grid information
    """
    # Get point cloud bounds
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    
    print(f"Point cloud bounds: X[{x_min:.3f}, {x_max:.3f}], Y[{y_min:.3f}, {y_max:.3f}]")
    
    # Calculate patch dimensions in world coordinates
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # Calculate number of patches that can fit
    n_patches_x = int((x_range - patch_size[0]) // stride[0]) + 1
    n_patches_y = int((y_range - patch_size[1]) // stride[1]) + 1
    
    print(f"Will create {n_patches_x} x {n_patches_y} = {n_patches_x * n_patches_y} patches")
    print(f"Patch size: {patch_size[0]:.3f} x {patch_size[1]:.3f}")
    print(f"Stride: {stride[0]:.3f} x {stride[1]:.3f}")
    
    patches = []
    patch_centers = []
    patch_info = []
    
    for i in range(n_patches_x):
        for j in range(n_patches_y):
            # Calculate patch boundaries
            patch_x_start = x_min + i * stride[0]
            patch_x_end = patch_x_start + patch_size[0]
            patch_y_start = y_min + j * stride[1]
            patch_y_end = patch_y_start + patch_size[1]
            
            # Extract points within this patch
            mask = ((points[:, 0] >= patch_x_start) & (points[:, 0] < patch_x_end) &
                    (points[:, 1] >= patch_y_start) & (points[:, 1] < patch_y_end))
            
            patch_points = points[mask]
            
            # Store patch center
            center_x = (patch_x_start + patch_x_end) / 2
            center_y = (patch_y_start + patch_y_end) / 2
            
            patches.append(patch_points)
            patch_centers.append((center_x, center_y))
            patch_info.append({
                'id': len(patches) - 1,
                'grid_i': i,
                'grid_j': j,
                'bounds': (patch_x_start, patch_x_end, patch_y_start, patch_y_end),
                'center': (center_x, center_y),
                'n_points': len(patch_points)
            })
    
    patch_grid_info = {
        'n_patches_x': n_patches_x,
        'n_patches_y': n_patches_y,
        'patch_size': patch_size,
        'stride': stride,
        'x_range': (x_min, x_max),
        'y_range': (y_min, y_max)
    }
    
    # Print statistics
    patch_sizes = [len(patch) for patch in patches]
    print(f"Patch statistics:")
    print(f"  Points per patch - Min: {min(patch_sizes)}, Max: {max(patch_sizes)}, Avg: {np.mean(patch_sizes):.1f}")
    print(f"  Empty patches: {sum(1 for size in patch_sizes if size == 0)}")
    
    return patches, patch_centers, patch_info, patch_grid_info

def calculate_stride_patch_differences(patches, patch_info, patch_grid_info, cfg):
    """
    Calculate average difference subspace magnitude for each patch using stride-based neighbors.
    Similar to convolution, considers overlapping neighborhoods.
    """
    n_patches_x = patch_grid_info['n_patches_x']
    n_patches_y = patch_grid_info['n_patches_y']
    
    # Generate subspaces for all patches
    print("Generating subspaces for all patches...")
    patch_subspaces = []
    for i, patch in enumerate(patches):
        if len(patch) > 0:
            subspace = generate_subspace(patch, cfg)
        else:
            subspace = np.zeros((1, cfg.subspace_dim))
        patch_subspaces.append(subspace)
    
    # Calculate differences for each patch
    average_magnitudes = []
    
    def get_stride_neighbors(grid_i, grid_j, radius=1):
        """Get neighboring patches within radius, considering stride pattern."""
        neighbors = []
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                if di == 0 and dj == 0:
                    continue
                    
                ni, nj = grid_i + di, grid_j + dj
                if (0 <= ni < n_patches_x) and (0 <= nj < n_patches_y):
                    neighbor_idx = ni * n_patches_y + nj
                    neighbors.append(neighbor_idx)
        return neighbors
    
    print("Calculating patch differences...")
    for patch_idx, info in enumerate(patch_info):
        grid_i, grid_j = info['grid_i'], info['grid_j']
        
        # Get neighboring patches
        neighbor_indices = get_stride_neighbors(grid_i, grid_j, radius=1)
        
        if len(neighbor_indices) == 0 or len(patches[patch_idx]) == 0:
            average_magnitudes.append(0.0)
            continue
        
        # Calculate difference subspace magnitudes with neighbors
        S1 = patch_subspaces[patch_idx]
        magnitudes = []
        
        for neighbor_idx in neighbor_indices:
            if len(patches[neighbor_idx]) > 0:
                S2 = patch_subspaces[neighbor_idx]
                
                # Calculate difference subspace
                DS = gen_shape_difference_subspace(S1, S2, cfg)
                
                # Calculate magnitude (number of difference dimensions)
                if DS.shape[1] > 0:
                    # Option 1: Use dimension count
                    magnitude = DS.shape[1]
                    
                    # Option 2: Use sum of singular values for weighted measure
                    # _, s, _ = svd(DS, full_matrices=False)
                    # magnitude = np.sum(s)
                else:
                    magnitude = 0
                
                magnitudes.append(magnitude)
        
        # Average the magnitudes
        if magnitudes:
            avg_magnitude = np.mean(magnitudes)
        else:
            avg_magnitude = 0.0
            
        average_magnitudes.append(avg_magnitude)
    
    return average_magnitudes

def visualize_stride_patches(patches, patch_info, patch_grid_info, differences=None):
    """Visualize stride-based patches with optional difference coloring."""
    fig = plt.figure(figsize=(15, 10))
    
    # 3D visualization
    ax1 = fig.add_subplot(121, projection='3d')
    
    if differences is not None:
        # Color by differences
        vmin, vmax = np.min(differences), np.max(differences)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.hot
        
        for patch_idx, patch in enumerate(patches):
            if len(patch) > 0:
                color_val = differences[patch_idx]
                colors = np.array([cmap(norm(color_val))] * len(patch))
                ax1.scatter(patch[:, 0], patch[:, 1], patch[:, 2],
                           c=colors, alpha=0.6, s=1)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm, ax=ax1, label='Average DS Magnitude')
        ax1.set_title('Stride Patches colored by DS Magnitude')
    else:
        # Color by patch index
        colors = plt.cm.rainbow(np.linspace(0, 1, len(patches)))
        for patch, color in zip(patches, colors):
            if len(patch) > 0:
                ax1.scatter(patch[:, 0], patch[:, 1], patch[:, 2],
                           c=[color], alpha=0.6, s=1)
        ax1.set_title('Stride Patches (colored by patch)')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # 2D heatmap of differences
    if differences is not None:
        ax2 = fig.add_subplot(122)
        
        # Create grid for heatmap
        n_patches_x = patch_grid_info['n_patches_x']
        n_patches_y = patch_grid_info['n_patches_y']
        
        heatmap = np.zeros((n_patches_y, n_patches_x))
        for patch_idx, info in enumerate(patch_info):
            grid_i, grid_j = info['grid_i'], info['grid_j']
            heatmap[grid_j, grid_i] = differences[patch_idx]
        
        im = ax2.imshow(heatmap, cmap='hot', aspect='equal', origin='lower')
        ax2.set_title('Stride-based Difference Heatmap')
        ax2.set_xlabel('Grid X')
        ax2.set_ylabel('Grid Y')
        plt.colorbar(im, ax=ax2, label='Average DS Magnitude')
        
        # Add grid lines
        ax2.set_xticks(np.arange(-.5, n_patches_x, 1), minor=True)
        ax2.set_yticks(np.arange(-.5, n_patches_y, 1), minor=True)
        ax2.grid(which="minor", color="w", linestyle='-', linewidth=1)
    
    plt.tight_layout()
    
    # Save figure
    if not os.path.exists('graph_output'):
        os.makedirs('graph_output')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join('graph_output', f'stride_patches_{timestamp}.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def sample_points_from_mesh(ply_path, sample_size=10000, sample_method='uniform'):
    """Load 3D mesh from PLY file and sample points from the surface."""
    mesh = o3d.io.read_triangle_mesh(ply_path, enable_post_processing=False)
    
    print(f"Loaded mesh from PLY file:")
    print(f"  Number of vertices: {len(mesh.vertices)}")
    print(f"  Number of triangles: {len(mesh.triangles)}")
    
    if sample_method == 'uniform':
        print(f"\nSampling {sample_size} points uniformly from mesh surface...")
        pcd = mesh.sample_points_uniformly(number_of_points=sample_size)
        sampled_points = np.asarray(pcd.points)
        print(f"Uniformly sampled {len(sampled_points)} points from mesh surface")
        
    elif sample_method == 'poisson':
        print(f"\nSampling {sample_size} points using Poisson disk from mesh surface...")
        pcd = mesh.sample_points_poisson_disk(number_of_points=sample_size)
        sampled_points = np.asarray(pcd.points)
        print(f"Poisson disk sampled {len(sampled_points)} points from mesh surface")
    
    else:
        raise ValueError(f"Unknown sampling method: {sample_method}")
    
    print(f"\nFinal data shape: {sampled_points.shape}")
    print(f"X range: [{sampled_points[:, 0].min():.2f}, {sampled_points[:, 0].max():.2f}]")
    print(f"Y range: [{sampled_points[:, 1].min():.2f}, {sampled_points[:, 1].max():.2f}]")
    print(f"Z range: [{sampled_points[:, 2].min():.2f}, {sampled_points[:, 2].max():.2f}]")
    
    return sampled_points

def save_stride_analysis_results(patch_info, differences, filename):
    """Save stride-based analysis results to CSV."""
    data = []
    for i, info in enumerate(patch_info):
        row = {
            'patch_id': info['id'],
            'grid_i': info['grid_i'],
            'grid_j': info['grid_j'],
            'center_x': info['center']['0'],
            'center_y': info['center']['1'],
            'n_points': info['n_points'],
            'avg_ds_magnitude': differences[i],
            'bounds_x_start': info['bounds'][0],
            'bounds_x_end': info['bounds'][1],
            'bounds_y_start': info['bounds'][2],
            'bounds_y_end': info['bounds'][3]
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Saved stride analysis results to: {filename}")

# Main Workflow
if __name__ == "__main__":
    # Create output directory and setup logging
    if not os.path.exists('graph_output'):
        os.makedirs('graph_output')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join('graph_output', f'stride_analysis_log_{timestamp}.txt')
    logger = Logger(log_filename)
    sys.stdout = logger
    
    print(f"Stride-based Analysis started at: {datetime.now()}")
    print(f"All outputs will be saved to: graph_output/")
    print("="*60)
    
    # Load mesh and sample points
    ply_path = "C:\\Users\\ibrah\\OneDrive\\Documents\\newpc3.ply"
    
    # Sample parameters
    sample_size = 30000
    sample_method = 'uniform'
    
    print("Loading and sampling points from mesh...")
    sampled_points = sample_points_from_mesh(ply_path, sample_size, sample_method)
    
    # Configuration
    cfg = Confing()
    cfg.subspace_dim = 3
    
    # Stride parameters (similar to convolution)
    # Calculate reasonable patch size based on point cloud dimensions
    x_range = sampled_points[:, 0].max() - sampled_points[:, 0].min()
    y_range = sampled_points[:, 1].max() - sampled_points[:, 1].min()
    
    # Set patch size to be ~1/10 of the total range
    patch_width = x_range / 10
    patch_height = y_range / 10
    
    # Set stride to be half of patch size for 50% overlap
    stride_x = patch_width / 2
    stride_y = patch_height / 2
    
    patch_size = (patch_width, patch_height)
    stride = (stride_x, stride_y)
    
    print(f"\nStride Configuration:")
    print(f"Patch size: {patch_size}")
    print(f"Stride: {stride}")
    print(f"Overlap: ~50%")
    
    # Extract patches using stride
    print("\nExtracting patches with stride...")
    patches, patch_centers, patch_info, patch_grid_info = extract_patches_with_stride(
        sampled_points, patch_size, stride)
    
    # Visualize patches without differences first
    print("\nVisualizing stride-based patches...")
    visualize_stride_patches(patches, patch_info, patch_grid_info)
    
    # Calculate stride-based patch differences
    print("\nCalculating stride-based patch differences...")
    stride_differences = calculate_stride_patch_differences(patches, patch_info, patch_grid_info, cfg)
    
    # Visualize with differences
    print("\nVisualizing results with difference coloring...")
    visualize_stride_patches(patches, patch_info, patch_grid_info, stride_differences)
    
    # Save analysis results
    results_filename = os.path.join('graph_output', f'stride_analysis_results_{timestamp}.csv')
    save_stride_analysis_results(patch_info, stride_differences, results_filename)
    
    # Print statistics
    print(f"\nStride Analysis Statistics:")
    print(f"Total patches: {len(patches)}")
    print(f"Grid size: {patch_grid_info['n_patches_x']} x {patch_grid_info['n_patches_y']}")
    print(f"Difference range: [{np.min(stride_differences):.6f}, {np.max(stride_differences):.6f}]")
    print(f"Mean difference: {np.mean(stride_differences):.6f}")
    print(f"Std difference: {np.std(stride_differences):.6f}")
    
    # Find top anomalous patches
    top_patches_idx = np.argsort(stride_differences)[-5:]
    print(f"\nTop 5 patches with highest DS magnitude:")
    for rank, patch_idx in enumerate(reversed(top_patches_idx)):
        info = patch_info[patch_idx]
        print(f"{rank+1}. Patch {patch_idx} (Grid {info['grid_i']}, {info['grid_j']}) - "
              f"DS Magnitude: {stride_differences[patch_idx]:.6f}, "
              f"Points: {info['n_points']}, "
              f"Center: ({info['center'][0]:.3f}, {info['center'][1]:.3f})")
    
    # Create summary report
    summary_filename = os.path.join('graph_output', f'stride_analysis_summary_{timestamp}.txt')
    with open(summary_filename, 'w') as f:
        f.write(f"Stride-based Surface Analysis Summary\n")
        f.write(f"{'='*60}\n")
        f.write(f"Analysis Date: {datetime.now()}\n")
        f.write(f"Input File: {ply_path}\n")
        f.write(f"Sample Size: {sample_size} points\n")
        f.write(f"Patch Size: {patch_size}\n")
        f.write(f"Stride: {stride}\n")
        f.write(f"Grid Size: {patch_grid_info['n_patches_x']} x {patch_grid_info['n_patches_y']}\n")
        f.write(f"Total Patches: {len(patches)}\n")
        f.write(f"\nDifference Statistics:\n")
        f.write(f"  Range: [{np.min(stride_differences):.6f}, {np.max(stride_differences):.6f}]\n")
        f.write(f"  Mean: {np.mean(stride_differences):.6f}\n")
        f.write(f"  Std: {np.std(stride_differences):.6f}\n")
    
    print(f"\nSummary report saved as: {summary_filename}")
    
    print(f"\n{'='*60}")
    print("STRIDE ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Analysis completed at: {datetime.now()}")
    
    # Close logger
    sys.stdout = sys.__stdout__
    logger.close()