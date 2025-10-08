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

class Confing: # Keeping original typo Confing as in your code
    def __init__(self):
        self.subspace_dim = 3 # Default value


def generate_subspace(data_segment, cfg):
    """Generate a subspace from a segment using SVD."""
    if len(data_segment) < cfg.subspace_dim + 1: # Ensure enough points for meaningful SVD
        # Return a zero matrix with correct dimensions if not enough points
        return np.zeros((data_segment.shape[0], cfg.subspace_dim))

    X = data_segment
    mv = np.mean(X, axis=0)
    X_centered = X - mv
    
    # Use economy mode SVD to handle different sized patches
    U, s, Vt = svd(X_centered, full_matrices=False) # Get singular values (s) and V transpose (Vt)
    
    # Ensure we don't exceed the number of available components
    # We might consider selecting components based on explained variance, but for now, top 'subspace_dim'
    
    return U[:, 0:3]


def generate_local_shape_subspace(patch, cfg, k_neighbors=10):
    """Generate a subspace from local shape features (curvature) instead of raw coordinates."""
    from sklearn.neighbors import NearestNeighbors
    
    # Ensure we have enough points for k-nearest neighbors and covariance
    if len(patch) < k_neighbors + 1:
        # Reduce k_neighbors if patch is too small, but ensure at least 3 for 3D covariance
        k_neighbors = max(3, len(patch) - 1)
        if len(patch) < 3: # Not enough points even for basic covariance
            return np.zeros((len(patch), cfg.subspace_dim)) # Return empty or zero subspace
    
    # Initialize nearest neighbors finder
    nbrs = NearestNeighbors(n_neighbors=k_neighbors+1, algorithm='ball_tree').fit(patch)
    
    # Compute local features for each point
    local_features = []
    
    for i, point in enumerate(patch):
        # Find k nearest neighbors (excluding the point itself)
        distances, indices = nbrs.kneighbors([point])
        neighbor_indices = indices[0][1:]  # Exclude self
        
        # Get neighbor points
        neighbors = patch[neighbor_indices]
        
        # Compute local covariance
        if len(neighbors) >= 3: # Need at least 3 points for 3D covariance
            try:
                local_center = np.mean(neighbors, axis=0)
                centered_neighbors = neighbors - local_center
                
                # Compute eigenvalues and eigenvectors of local covariance (these indicate local shape)
                # Use eigh for symmetric covariance matrix, returns sorted eigenvalues and corresponding eigenvectors
                eigenvalues, eigenvectors = eigh(np.cov(centered_neighbors.T))
                
                # Sort eigenvalues in descending order for common curvature features
                sorted_indices = np.argsort(eigenvalues)[::-1]
                eigenvalues_sorted = eigenvalues[sorted_indices]
                eigenvectors_sorted = eigenvectors[:, sorted_indices] # Eigenvectors are also sorted now

                # Features based on eigenvalues (using sorted_eigenvalues)
                lambda1, lambda2, lambda3 = eigenvalues_sorted[0], eigenvalues_sorted[1], eigenvalues_sorted[2]
                
                # Avoid division by zero
                sum_eigenvalues = lambda1 + lambda2 + lambda3
                if sum_eigenvalues > 1e-10:
                    linearity = (lambda1 - lambda2) / lambda1
                    planarity = (lambda2 - lambda3) / lambda1
                    sphericity = lambda3 / lambda1
                    anisotropy = (lambda1 - lambda3) / lambda1
                else:
                    linearity = planarity = sphericity = anisotropy = 0
                
                # Local height variation
                z_std = np.std(neighbors[:, 2])
                z_range = np.max(neighbors[:, 2]) - np.min(neighbors[:, 2])
                
                # Distance from local plane (normal is eigenvector corresponding to smallest eigenvalue, lambda3)
                # Smallest eigenvalue corresponds to the direction of least variance, which is the normal for a plane
                normal_vector = eigenvectors_sorted[:, 2] # This is the eigenvector for lambda3
                plane_distance = abs(np.dot(point - local_center, normal_vector))
                
                features = [
                    lambda1,          # Largest eigenvalue (spread)
                    lambda2,          # Medium eigenvalue
                    lambda3,          # Smallest eigenvalue (local flatness)
                    linearity,        # Linear structure
                    planarity,        # Planar structure
                    sphericity,       # Spherical structure
                    anisotropy,       # Overall anisotropy
                    z_std,            # Height variation
                    z_range,          # Height range
                    plane_distance    # Distance from local plane fit
                ]
            except np.linalg.LinAlgError:
                # Handle singular matrix or other linear algebra errors
                features = [0] * 10
            except Exception as e:
                # Catch any other unexpected errors
                print(f"Error in local shape feature calculation for point {i}: {e}")
                features = [0] * 10
        else:
            features = [0] * 10 # Not enough neighbors for meaningful features
            
        local_features.append(features)
    
    # Convert to array
    local_features = np.array(local_features)
    
    # Normalize features - only if they have variance
    for i in range(local_features.shape[1]):
        col = local_features[:, i]
        if np.std(col) > 1e-10: # Avoid division by zero
            local_features[:, i] = (col - np.mean(col)) / np.std(col)
        else:
            local_features[:, i] = 0 # If no variance, set to 0 (or keep as is if already 0)
    
    # Handle cases where all features might be zero or have no variance
    if np.all(local_features == 0):
        return np.zeros((patch.shape[0], cfg.subspace_dim))

    # Apply SVD to local features
    U, _, _ = svd(local_features, full_matrices=False)
    
    # Ensure we don't exceed the number of available components
    n_components = min(cfg.subspace_dim, U.shape[1])
    return U[:, 0:n_components]

def gen_shape_difference_subspace(S1, S2, cfg):
    """Generate the Difference Subspace (DS) between two subspaces."""
    # Ensure both subspaces have the same number of rows by padding with zeros if necessary
    n_rows = max(S1.shape[0], S2.shape[0])
    
    # Pad S1 if necessary
    if S1.shape[0] < n_rows:
        S1_padded = np.zeros((n_rows, S1.shape[1]))
        S1_padded[:S1.shape[0], :] = S1
        S1 = S1_padded
    
    # Pad S2 if necessary
    if S2.shape[0] < n_rows:
        S2_padded = np.zeros((n_rows, S2.shape[1]))
        S2_padded[:S2.shape[0], :] = S2
        S2 = S2_padded
    
    G = S1 @ S1.T + S2 @ S2.T
    eigen_val, eigen_vec = eig(G)
    idx = np.where((1e-6 < eigen_val) & (eigen_val < 1))[0]
    return eigen_vec[:, idx]

def calculate_subspace_distance(S1, S2):
    """
    Calculate the Frobenius norm of the difference of projection matrices 
    between two subspaces S1 and S2. This is a common and robust metric
    for subspace distance.
    """
    # Ensure both subspaces have the same number of rows by padding with zeros if necessary
    # This padding is necessary because projection matrix P = S @ S.T requires S to be N x D
    # where N is number of points, D is subspace dimension.
    # The projection matrix P will be N x N.
    n_rows = max(S1.shape[0], S2.shape[0])
    
    # Pad S1 if necessary
    if S1.shape[0] < n_rows:
        S1_padded = np.zeros((n_rows, S1.shape[1]))
        S1_padded[:S1.shape[0], :] = S1
        S1 = S1_padded
    
    # Pad S2 if necessary
    if S2.shape[0] < n_rows:
        S2_padded = np.zeros((n_rows, S2.shape[1]))
        S2_padded[:S2.shape[0], :] = S2
        S2 = S2_padded

    # If either subspace is effectively empty (zero columns), return max difference or zero
    if S1.shape[1] == 0 or S2.shape[1] == 0:
        return 0.0 # Or some other default, depending on what 'difference' means for empty spaces.
                    # For flat, non-existent patches, 0 might be appropriate if they are common.
                    # For a clear difference, could be np.inf or 1.0 (normalized).
    
    try:
        P1 = S1 @ S1.T
        P2 = S2 @ S2.T
        return np.linalg.norm(P1 - P2, 'fro') # Frobenius norm
    except np.linalg.LinAlgError as e:
        print(f"LinAlgError in calculate_subspace_distance: {e}")
        return 0.0 # Or handle appropriately
    except Exception as e:
        print(f"Error in calculate_subspace_distance: {e}")
        return 0.0


def calculate_patch_differences_with_ds_8neighbors(patches, cfg, n_patches_x=5, n_patches_y=5):
    """Calculate patch differences using the magnitude of actual Difference Subspace, considering all 8 neighbors."""
    average_magnitudes = []
    # Using the original generate_subspace for 'ds_magnitude'
    patch_subspaces = [generate_subspace(patch, cfg) for patch in patches]
    
    def get_neighbors_8(idx):
        row = idx // n_patches_x
        col = idx % n_patches_x
        neighbors = []
        weights = []
        
        # Check all 8 surrounding positions
        neighbor_positions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1)
        ]
        
        for dr, dc in neighbor_positions:
            new_row = row + dr
            new_col = col + dc
            if (0 <= new_row < n_patches_y) and (0 <= new_col < n_patches_x):
                neighbor_idx = new_row * n_patches_x + new_col
                neighbors.append(neighbor_idx)
                weights.append(1.0)
                
        return neighbors, weights
    
    for i in range(len(patches)):
        neighbors, weights = get_neighbors_8(i)
        patch_magnitudes = []
        
        S1 = patch_subspaces[i]
        
        for neighbor_idx, weight in zip(neighbors, weights):
            S2 = patch_subspaces[neighbor_idx]
            
            # Calculate the actual Difference Subspace
            DS = gen_shape_difference_subspace(S1, S2, cfg)
            
            # Calculate magnitude of DS (number of dimensions in difference subspace)
            # This represents how many directions are different between the two patches
            if DS.shape[1] > 0:  # If there are difference dimensions
                # Option 1: Use the number of difference dimensions (simplest)
                magnitude = DS.shape[1]
                
                # Option 2: Weight by the eigenvalues (if you want to consider strength)
                # We can get this from the SVD of DS
                # _, s, _ = svd(DS, full_matrices=False)
                # magnitude = np.sum(s)  # Sum of singular values
            else:
                magnitude = 0  # No difference between subspaces
                
            patch_magnitudes.append(magnitude)
                
        if patch_magnitudes:
            # Average magnitudes, typically sum(weights) is just count of neighbors here
            avg_magnitude = sum(patch_magnitudes) / len(patch_magnitudes) 
            average_magnitudes.append(avg_magnitude)
        else:
            average_magnitudes.append(0) # No neighbors or all calculations failed
            
    return average_magnitudes

def sample_points_from_mesh(ply_path, sample_size=10000, sample_method='uniform'):
    """
    Load 3D mesh from PLY file and sample points from the surface.
    
    Args:
        ply_path: Path to the PLY file containing a 3D mesh
        sample_size: Number of points to sample
        sample_method: Sampling method - 'uniform' or 'poisson'
        
    Returns:
        numpy array of shape (N, 3) containing the sampled points
    """
    # Load the PLY file as a mesh using Open3D
    mesh = o3d.io.read_triangle_mesh(ply_path, enable_post_processing=False)
    
    # Get mesh information
    print(f"Loaded mesh from PLY file:")
    print(f"  Number of vertices: {len(mesh.vertices)}")
    print(f"  Number of triangles: {len(mesh.triangles)}")
    print(f"  Has vertex normals: {mesh.has_vertex_normals()}")
    print(f"  Has vertex colors: {mesh.has_vertex_colors()}")
    
    # Sample points from mesh surface
    if sample_method == 'uniform':
        # Uniform sampling from mesh surface
        print(f"\nSampling {sample_size} points uniformly from mesh surface...")
        pcd = mesh.sample_points_uniformly(number_of_points=sample_size)
        sampled_points = np.asarray(pcd.points)
        print(f"Uniformly sampled {len(sampled_points)} points from mesh surface")
        
    elif sample_method == 'poisson':
        # Poisson disk sampling from mesh surface
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


def save_sampled_points_to_csv(points, output_path):
    """
    Save sampled points to CSV file.
    
    Args:
        points: numpy array of shape (N, 3) containing point cloud data
        output_path: Path to save the CSV file
    """
    df = pd.DataFrame(points, columns=['x', 'y', 'z'])
    df.to_csv(output_path, index=False)
    print(f"Saved {len(points)} points to {output_path}")


def numpy_to_pointcloud(array):
    """Convert a NumPy array to an Open3D PointCloud object."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(array)
    return pcd


def divide_into_equal_patches(points, n_patches_x=5, n_patches_y=5):
    """Divide points into patches with approximately equal number of points."""
    n_points = len(points)
    points_sorted_x = points[np.argsort(points[:, 0])]
    points_per_strip = n_points // n_patches_x

    patches = []
    for i in range(n_patches_x):
        start_idx = i * points_per_strip
        end_idx = start_idx + points_per_strip if i < n_patches_x - 1 else n_points
        strip = points_sorted_x[start_idx:end_idx]

        # Sort strip by y coordinate
        strip_sorted_y = strip[np.argsort(strip[:, 1])]
        points_per_patch = len(strip_sorted_y) // n_patches_y

        for j in range(n_patches_y):
            start_idx_y = j * points_per_patch
            end_idx_y = start_idx_y + points_per_patch if j < n_patches_y - 1 else len(strip_sorted_y)
            patch = strip_sorted_y[start_idx_y:end_idx_y]
            patches.append(patch)

    return patches


def visualize_patches(patches, n_patches_x=5, n_patches_y=5, case_type="mesh_sampled"):
    """Visualize the patches with different colors."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.rainbow(np.linspace(0, 1, len(patches)))

    for patch, color in zip(patches, colors):
        if len(patch) > 0:  # Check if patch contains points
            ax.scatter(patch[:, 0], patch[:, 1], patch[:, 2],
                      c=[color], alpha=0.6, s=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(f'Mesh divided into {n_patches_x}x{n_patches_y} patches')
    
    # Create output directory if it doesn't exist
    if not os.path.exists('graph_output'):
        os.makedirs('graph_output')
    
    plt.savefig(os.path.join('graph_output', f'{case_type}_patches.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()  # Free memory


def analyze_z_distribution(points, n_bins=50, timestamp=""):
    """Analyze Z-coordinate distribution to detect bumps."""
    z_values = points[:, 2]
    
    plt.figure(figsize=(15, 5))
    
    # Histogram
    plt.subplot(1, 3, 1)
    plt.hist(z_values, bins=n_bins, edgecolor='black')
    plt.xlabel('Z coordinate')
    plt.ylabel('Count')
    plt.title('Z-coordinate Distribution')
    
    # Scatter plot colored by Z
    plt.subplot(1, 3, 2)
    plt.scatter(points[:, 0], points[:, 1], c=z_values, cmap='viridis', s=1)
    plt.colorbar(label='Z coordinate')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Point Cloud colored by height')
    
    # 3D view
    ax = plt.subplot(1, 3, 3, projection='3d')
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                        c=z_values, cmap='viridis', s=1)
    plt.colorbar(scatter, label='Z coordinate')
    
    # Fix aspect ratio to show true proportions
    # Calculate the ranges
    x_range = points[:, 0].max() - points[:, 0].min()
    y_range = points[:, 1].max() - points[:, 1].min()
    z_range = points[:, 2].max() - points[:, 2].min()
    
    # Set equal aspect ratio based on the maximum range
    max_range = max(x_range, y_range, z_range)
    
    # Set the aspect ratio
    ax.set_box_aspect([x_range/max_range, y_range/max_range, z_range/max_range])
    
    # Alternative: Set limits to be equal
    mid_x = (points[:, 0].max() + points[:, 0].min()) / 2
    mid_y = (points[:, 1].max() + points[:, 1].min()) / 2
    mid_z = (points[:, 2].max() + points[:, 2].min()) / 2
    
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D view colored by height (True Scale)')
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    if not os.path.exists('graph_output'):
        os.makedirs('graph_output')
    
    filename = f'z_distribution_analysis_{timestamp}.png' if timestamp else 'z_distribution_analysis.png'
    plt.savefig(os.path.join('graph_output', filename), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()  # Free memory
    
    print(f"Z range: {z_values.min():.6f} to {z_values.max():.6f}")
    print(f"Z standard deviation: {z_values.std():.6f}")


def calculate_height_based_differences(patches, n_patches_x=10, n_patches_y=10):
    """Calculate differences based on height statistics."""
    patch_stats = []
    
    # Calculate statistics for each patch
    for patch in patches:
        if len(patch) > 0:
            z_values = patch[:, 2]
            stats = {
                'mean_z': np.mean(z_values),
                'std_z': np.std(z_values),
                'max_z': np.max(z_values),
                'min_z': np.min(z_values),
                'range_z': np.max(z_values) - np.min(z_values),
                'percentile_95': np.percentile(z_values, 95),
                'percentile_5': np.percentile(z_values, 5)
            }
            patch_stats.append(stats)
        else:
            patch_stats.append({
                'mean_z': 0, 'std_z': 0, 'max_z': 0, 'min_z': 0,
                'range_z': 0, 'percentile_95': 0, 'percentile_5': 0
            })
    
    # Calculate differences from neighbors
    differences = []
    
    for i in range(len(patches)):
        row = i // n_patches_x
        col = i % n_patches_x
        neighbors = []
        
        # Get all valid neighbors (8-connectivity)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < n_patches_y and 0 <= new_col < n_patches_x:
                    neighbor_idx = new_row * n_patches_x + new_col
                    neighbors.append(neighbor_idx)
        
        if neighbors:
            # Calculate difference in mean height
            current_mean = patch_stats[i]['mean_z']
            neighbor_means = [patch_stats[n]['mean_z'] for n in neighbors]
            avg_neighbor_mean = np.mean(neighbor_means)
            
            # Height difference metric
            height_diff = abs(current_mean - avg_neighbor_mean)
            
            # Also consider maximum height difference
            current_max = patch_stats[i]['max_z']
            neighbor_maxs = [patch_stats[n]['max_z'] for n in neighbors]
            avg_neighbor_max = np.mean(neighbor_maxs)
            max_diff = abs(current_max - avg_neighbor_max)
            
            # Standard deviation (surface roughness)
            std_diff = patch_stats[i]['std_z']
            
            # Combined metric (you can adjust weights)
            combined_diff = height_diff + 0.3 * max_diff + 0.2 * std_diff
            differences.append(combined_diff)
        else:
            differences.append(0)
    
    return differences, patch_stats


def visualize_height_based_differences(patches, differences, patch_stats, n_patches_x=10, n_patches_y=10, case_type="height_based"):
    """Visualize the height-based differences with enhanced visualization."""
    fig = plt.figure(figsize=(20, 12))
    
    # Convert differences to numpy array and reshape
    heatmap = np.array(differences).reshape(n_patches_y, n_patches_x)
    
    # 1. 3D scatter plot colored by differences
    ax_3d = fig.add_subplot(2, 3, 1, projection='3d')
    
    # Create color normalization
    vmin = np.min(heatmap)
    vmax = np.max(heatmap)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.hot
    
    # Plot patches with colors based on differences
    for idx, patch in enumerate(patches):
        if len(patch) > 0:
            color_val = differences[idx]
            colors = np.array([cmap(norm(color_val))] * len(patch))
            ax_3d.scatter(patch[:, 0], patch[:, 1], patch[:, 2],
                         c=colors, alpha=0.6, s=1)
    
    ax_3d.set_title('Patches colored by height-based differences')
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax_3d, label='Height Difference')
    
    # 2. Height difference heatmap
    ax_heat = fig.add_subplot(2, 3, 2)
    im = ax_heat.imshow(heatmap, cmap='hot', aspect='equal')
    ax_heat.set_title('Height-Based Difference Heatmap')
    plt.colorbar(im, ax=ax_heat, label='Combined Difference Metric')
    
    # Add grid lines
    ax_heat.set_xticks(np.arange(-.5, n_patches_x, 1), minor=True)
    ax_heat.set_yticks(np.arange(-.5, n_patches_y, 1), minor=True)
    ax_heat.grid(which="minor", color="w", linestyle='-', linewidth=1)
    
    # 3. Mean height heatmap
    ax_mean = fig.add_subplot(2, 3, 3)
    mean_heights = np.array([stats['mean_z'] for stats in patch_stats]).reshape(n_patches_y, n_patches_x)
    im_mean = ax_mean.imshow(mean_heights, cmap='terrain', aspect='equal')
    ax_mean.set_title('Mean Height per Patch')
    plt.colorbar(im_mean, ax=ax_mean, label='Mean Z')
    
    # 4. Max height heatmap
    ax_max = fig.add_subplot(2, 3, 4)
    max_heights = np.array([stats['max_z'] for stats in patch_stats]).reshape(n_patches_y, n_patches_x)
    im_max = ax_max.imshow(max_heights, cmap='terrain', aspect='equal')
    ax_max.set_title('Maximum Height per Patch')
    plt.colorbar(im_max, ax=ax_max, label='Max Z')
    
    # 5. Standard deviation heatmap
    ax_std = fig.add_subplot(2, 3, 5)
    std_heights = np.array([stats['std_z'] for stats in patch_stats]).reshape(n_patches_y, n_patches_x)
    im_std = ax_std.imshow(std_heights, cmap='viridis', aspect='equal')
    ax_std.set_title('Height Standard Deviation per Patch')
    plt.colorbar(im_std, ax=ax_std, label='Std Z')
    
    # 6. Detected anomalies (patches with high differences)
    ax_anomaly = fig.add_subplot(2, 3, 6)
    threshold = np.percentile(differences, 80)  # Top 20% as anomalies
    anomaly_map = (heatmap > threshold).astype(float)
    im_anomaly = ax_anomaly.imshow(anomaly_map, cmap='RdBu_r', aspect='equal', vmin=0, vmax=1)
    ax_anomaly.set_title(f'Detected Anomalies (top 20%)')
    plt.colorbar(im_anomaly, ax=ax_anomaly, label='Anomaly')
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    if not os.path.exists('graph_output'):
        os.makedirs('graph_output')
    
    plt.savefig(os.path.join('graph_output', f'{case_type}_height_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()  # Free memory
    
    # Print statistics
    print("\nHeight-based analysis statistics:")
    print(f"Difference range: [{np.min(differences):.6f}, {np.max(differences):.6f}]")
    print(f"Mean difference: {np.mean(differences):.6f}")
    print(f"Std difference: {np.std(differences):.6f}")
    print(f"Number of anomalous patches (top 20%): {np.sum(anomaly_map)}")
    
    return heatmap


def visualize_ds_differences(patches, ds_differences, n_patches_x=10, n_patches_y=10, case_type="ds_method"):
    """Visualize the DS differences with similar style to height-based visualization."""
    fig = plt.figure(figsize=(20, 8))
    
    # Convert differences to numpy array and reshape
    heatmap = np.array(ds_differences).reshape(n_patches_y, n_patches_x)
    
    # Normalize for better visualization
    heatmap_normalized = (heatmap - np.mean(heatmap)) / (np.std(heatmap) + 1e-8)
    
    # 1. 3D scatter plot colored by DS differences
    ax_3d = fig.add_subplot(1, 4, 1, projection='3d')
    
    # Create color normalization
    vmin = np.min(heatmap)
    vmax = np.max(heatmap)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.hot
    
    # Plot patches with colors based on DS differences
    for idx, patch in enumerate(patches):
        if len(patch) > 0:
            color_val = ds_differences[idx]
            colors = np.array([cmap(norm(color_val))] * len(patch))
            ax_3d.scatter(patch[:, 0], patch[:, 1], patch[:, 2],
                         c=colors, alpha=0.6, s=1)
    
    ax_3d.set_title('Patches colored by DS magnitude')
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax_3d, label='DS Magnitude')
    
    # 2. Raw DS heatmap
    ax_raw = fig.add_subplot(1, 4, 2)
    im_raw = ax_raw.imshow(heatmap, cmap='hot', aspect='equal')
    ax_raw.set_title('DS Magnitude Heatmap (Raw)')
    plt.colorbar(im_raw, ax=ax_raw, label='DS Magnitude')
    
    # Add grid lines
    ax_raw.set_xticks(np.arange(-.5, n_patches_x, 1), minor=True)
    ax_raw.set_yticks(np.arange(-.5, n_patches_y, 1), minor=True)
    ax_raw.grid(which="minor", color="w", linestyle='-', linewidth=1)
    
    # 3. Normalized DS heatmap
    ax_norm = fig.add_subplot(1, 4, 3)
    im_norm = ax_norm.imshow(heatmap_normalized, cmap='hot', aspect='equal')
    ax_norm.set_title('DS Magnitude Heatmap (Normalized)')
    plt.colorbar(im_norm, ax=ax_norm, label='Normalized DS Magnitude')
    
    # Add grid lines
    ax_norm.set_xticks(np.arange(-.5, n_patches_x, 1), minor=True)
    ax_norm.set_yticks(np.arange(-.5, n_patches_y, 1), minor=True)
    ax_norm.grid(which="minor", color="w", linestyle='-', linewidth=1)
    
    # 4. Detected anomalies using DS
    ax_anomaly = fig.add_subplot(1, 4, 4)
    threshold = np.percentile(ds_differences, 80)  # Top 20% as anomalies
    anomaly_map = (heatmap > threshold).astype(float)
    im_anomaly = ax_anomaly.imshow(anomaly_map, cmap='RdBu_r', aspect='equal', vmin=0, vmax=1)
    ax_anomaly.set_title('DS Detected Anomalies (top 20%)')
    plt.colorbar(im_anomaly, ax=ax_anomaly, label='Anomaly')
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    if not os.path.exists('graph_output'):
        os.makedirs('graph_output')
    
    plt.savefig(os.path.join('graph_output', f'{case_type}_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()  # Free memory
    
    # Print statistics
    print("\nDS analysis statistics:")
    print(f"DS magnitude range: [{np.min(ds_differences):.6f}, {np.max(ds_differences):.6f}]")
    print(f"Mean DS magnitude: {np.mean(ds_differences):.6f}")
    print(f"Std DS magnitude: {np.std(ds_differences):.6f}")
    print(f"Number of anomalous patches (top 20%): {np.sum(anomaly_map)}")
    
    return heatmap


def compare_height_and_ds_heatmaps(height_differences, ds_differences, n_patches_x=10, n_patches_y=10):
    """Create side-by-side comparison of height-based and DS methods."""
    fig = plt.figure(figsize=(20, 10))
    
    # Convert to heatmaps
    height_heatmap = np.array(height_differences).reshape(n_patches_y, n_patches_x)
    ds_heatmap = np.array(ds_differences).reshape(n_patches_y, n_patches_x)
    
    # Normalize both for fair comparison
    height_norm = (height_heatmap - np.min(height_heatmap)) / (np.max(height_heatmap) - np.min(height_heatmap))
    ds_norm = (ds_heatmap - np.min(ds_heatmap)) / (np.max(ds_heatmap) - np.min(ds_heatmap))
    
    # 1. Height-based heatmap
    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.imshow(height_norm, cmap='hot', aspect='equal')
    ax1.set_title('Height-Based Detection (Normalized)', fontsize=14)
    plt.colorbar(im1, ax=ax1)
    
    # 2. DS heatmap
    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.imshow(ds_norm, cmap='hot', aspect='equal')
    ax2.set_title('DS-Based Detection (Normalized)', fontsize=14)
    plt.colorbar(im2, ax=ax2)
    
    # 3. Difference between methods
    ax3 = fig.add_subplot(2, 3, 3)
    difference = height_norm - ds_norm
    im3 = ax3.imshow(difference, cmap='RdBu', aspect='equal', vmin=-1, vmax=1)
    ax3.set_title('Difference (Height - DS)', fontsize=14)
    plt.colorbar(im3, ax=ax3)
    
    # 4. Height-based anomalies
    ax4 = fig.add_subplot(2, 3, 4)
    height_threshold = np.percentile(height_differences, 80)
    height_anomalies = (height_heatmap > height_threshold).astype(float)
    im4 = ax4.imshow(height_anomalies, cmap='RdBu_r', aspect='equal', vmin=0, vmax=1)
    ax4.set_title('Height-Based Anomalies (Top 20%)', fontsize=14)
    
    # 5. DS anomalies
    ax5 = fig.add_subplot(2, 3, 5)
    ds_threshold = np.percentile(ds_differences, 80)
    ds_anomalies = (ds_heatmap > ds_threshold).astype(float)
    im5 = ax5.imshow(ds_anomalies, cmap='RdBu_r', aspect='equal', vmin=0, vmax=1)
    ax5.set_title('DS Anomalies (Top 20%)', fontsize=14)
    
    # 6. Agreement map
    ax6 = fig.add_subplot(2, 3, 6)
    agreement = ((height_anomalies == 1) & (ds_anomalies == 1)).astype(float)
    disagreement = ((height_anomalies != ds_anomalies)).astype(float) * 0.5
    combined = agreement + disagreement
    im6 = ax6.imshow(combined, cmap='RdYlGn', aspect='equal', vmin=0, vmax=1)
    ax6.set_title('Agreement Map (Green=Agree, Yellow=Disagree)', fontsize=14)
    
    # Add grid lines to all subplots
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_xticks(np.arange(-.5, n_patches_x, 1), minor=True)
        ax.set_yticks(np.arange(-.5, n_patches_y, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    if not os.path.exists('graph_output'):
        os.makedirs('graph_output')
    
    plt.savefig(os.path.join('graph_output', f'height_vs_ds_comparison_{n_patches_x}x{n_patches_y}.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()  # Free memory
    
    # Print comparison statistics
    print("\nComparison Statistics:")
    print(f"Patches detected by both methods: {np.sum(agreement)}")
    print(f"Patches detected only by height: {np.sum((height_anomalies == 1) & (ds_anomalies == 0))}")
    print(f"Patches detected only by DS: {np.sum((height_anomalies == 0) & (ds_anomalies == 1))}")
    print(f"Agreement percentage: {np.sum(height_anomalies == ds_anomalies) / len(height_anomalies.flatten()) * 100:.1f}%")


def generate_height_deviation_subspace(data_segment, cfg):
    """Generate a subspace from height deviations instead of raw coordinates."""
    if len(data_segment) < cfg.subspace_dim + 1: # Ensure enough points
        return np.zeros((data_segment.shape[0], cfg.subspace_dim))

    # Extract z coordinates
    z_values = data_segment[:, 2]
    
    # Calculate mean height
    mean_z = np.mean(z_values)
    
    # Create height deviation features
    height_deviations = z_values - mean_z
    
    # Add some spatial context - include x,y gradients
    x_values = data_segment[:, 0]
    y_values = data_segment[:, 1]
    
    # Create feature matrix with height deviations and spatial info
    # Adding squared deviations to emphasize larger variations
    features = np.column_stack([
        height_deviations,
        x_values - np.mean(x_values), # Centered x
        y_values - np.mean(y_values), # Centered y
        height_deviations**2 # Emphasize large deviations
    ])

    # Normalize features to prevent one feature from dominating SVD
    for i in range(features.shape[1]):
        col = features[:, i]
        if np.std(col) > 1e-10:
            features[:, i] = (col - np.mean(col)) / np.std(col)
        else:
            features[:, i] = 0

    # Apply SVD to features
    U, _, _ = svd(features, full_matrices=False)
    
    # Ensure we don't exceed the number of available components
    n_components = min(cfg.subspace_dim, U.shape[1])
    return U[:, 0:n_components]

def calculate_patch_differences_with_modified_ds(patches, cfg, n_patches_x=5, n_patches_y=5):
    """Calculate patch differences using modified DS based on height deviations."""
    average_magnitudes = []
    # Using the new generate_height_deviation_subspace
    patch_subspaces = [generate_height_deviation_subspace(patch, cfg) for patch in patches]
    
    def get_neighbors_8(idx):
        row = idx // n_patches_x
        col = idx % n_patches_x
        neighbors = []
        weights = []
        
        # Check all 8 surrounding positions
        neighbor_positions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1)
        ]
        
        for dr, dc in neighbor_positions:
            new_row = row + dr
            new_col = col + dc
            if (0 <= new_row < n_patches_y) and (0 <= new_col < n_patches_x):
                neighbor_idx = new_row * n_patches_x + new_col
                neighbors.append(neighbor_idx)
                weights.append(1.0)
                
        return neighbors, weights
    
    for i in range(len(patches)):
        neighbors, weights = get_neighbors_8(i)
        patch_magnitudes = []
        
        S1 = patch_subspaces[i]
        
        for neighbor_idx, weight in zip(neighbors, weights):
            S2 = patch_subspaces[neighbor_idx]
            
            # Use the robust subspace distance directly
            magnitude = calculate_subspace_distance(S1, S2)
            patch_magnitudes.append(magnitude)
                
        if patch_magnitudes:
            avg_magnitude = sum(patch_magnitudes) / len(patch_magnitudes)
            average_magnitudes.append(avg_magnitude)
        else:
            average_magnitudes.append(0)
            
    return average_magnitudes


def calculate_patch_differences_with_local_shape_ds(patches, cfg, n_patches_x=5, n_patches_y=5):
    """Calculate patch differences using local shape features DS."""
    average_magnitudes = []
    # Using the generate_local_shape_subspace
    patch_subspaces = [generate_local_shape_subspace(patch, cfg) for patch in patches]
    
    def get_neighbors_8(idx):
        row = idx // n_patches_x
        col = idx % n_patches_x
        neighbors = []
        weights = []
        
        # Check all 8 surrounding positions
        neighbor_positions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1)
        ]
        
        for dr, dc in neighbor_positions:
            new_row = row + dr
            new_col = col + dc
            if (0 <= new_row < n_patches_y) and (0 <= new_col < n_patches_x):
                neighbor_idx = new_row * n_patches_x + new_col
                neighbors.append(neighbor_idx)
                weights.append(1.0)
                
        return neighbors, weights
    
    for i in range(len(patches)):
        neighbors, weights = get_neighbors_8(i)
        patch_magnitudes = []
        
        S1 = patch_subspaces[i]
        
        for neighbor_idx, weight in zip(neighbors, weights):
            S2 = patch_subspaces[neighbor_idx]
            
            # Use the robust subspace distance directly
            magnitude = calculate_subspace_distance(S1, S2)
            patch_magnitudes.append(magnitude)
                
        if patch_magnitudes:
            avg_magnitude = sum(patch_magnitudes) / len(patch_magnitudes)
            average_magnitudes.append(avg_magnitude)
        else:
            average_magnitudes.append(0)
            
    return average_magnitudes


def compare_all_methods(height_differences, ds_differences, modified_ds_differences, 
                       local_shape_ds_differences, n_patches_x=10, n_patches_y=10):
    """Compare height-based, original DS, modified DS, and local shape DS methods."""
    fig = plt.figure(figsize=(20, 15))
    
    # Convert to heatmaps
    height_heatmap = np.array(height_differences).reshape(n_patches_y, n_patches_x)
    ds_heatmap = np.array(ds_differences).reshape(n_patches_y, n_patches_x)
    modified_ds_heatmap = np.array(modified_ds_differences).reshape(n_patches_y, n_patches_x)
    local_shape_ds_heatmap = np.array(local_shape_ds_differences).reshape(n_patches_y, n_patches_x)
    
    # Normalize all for fair comparison
    height_norm = (height_heatmap - np.min(height_heatmap)) / (np.max(height_heatmap) - np.min(height_heatmap) + 1e-8)
    ds_norm = (ds_heatmap - np.min(ds_heatmap)) / (np.max(ds_heatmap) - np.min(ds_heatmap) + 1e-8)
    modified_ds_norm = (modified_ds_heatmap - np.min(modified_ds_heatmap)) / (np.max(modified_ds_heatmap) - np.min(modified_ds_heatmap) + 1e-8)
    local_shape_norm = (local_shape_ds_heatmap - np.min(local_shape_ds_heatmap)) / (np.max(local_shape_ds_heatmap) - np.min(local_shape_ds_heatmap) + 1e-8)
    
    # 1. Height-based heatmap
    ax1 = fig.add_subplot(3, 4, 1)
    im1 = ax1.imshow(height_norm, cmap='hot', aspect='equal')
    ax1.set_title('Height-Based (Normalized)', fontsize=12)
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    # 2. Original DS heatmap
    ax2 = fig.add_subplot(3, 4, 2)
    im2 = ax2.imshow(ds_norm, cmap='hot', aspect='equal')
    ax2.set_title('Original DS (Normalized)', fontsize=12)
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # 3. Modified DS heatmap
    ax3 = fig.add_subplot(3, 4, 3)
    im3 = ax3.imshow(modified_ds_norm, cmap='hot', aspect='equal')
    ax3.set_title('Modified DS - Height Deviations (Normalized)', fontsize=12)
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    # 4. Local Shape DS heatmap
    ax4 = fig.add_subplot(3, 4, 4)
    im4 = ax4.imshow(local_shape_norm, cmap='hot', aspect='equal')
    ax4.set_title('Local Shape DS (Normalized)', fontsize=12)
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    # Raw value heatmaps (second row)
    ax5 = fig.add_subplot(3, 4, 5)
    im5 = ax5.imshow(height_heatmap, cmap='hot', aspect='equal')
    ax5.set_title('Height-Based (Raw Values)', fontsize=12)
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    
    ax6 = fig.add_subplot(3, 4, 6)
    im6 = ax6.imshow(ds_heatmap, cmap='hot', aspect='equal')
    ax6.set_title('Original DS (Raw Values)', fontsize=12)
    plt.colorbar(im6, ax=ax6, fraction=0.046)
    
    ax7 = fig.add_subplot(3, 4, 7)
    im7 = ax7.imshow(modified_ds_heatmap, cmap='hot', aspect='equal')
    ax7.set_title('Modified DS (Raw Values)', fontsize=12)
    plt.colorbar(im7, ax=ax7, fraction=0.046)
    
    ax8 = fig.add_subplot(3, 4, 8)
    im8 = ax8.imshow(local_shape_ds_heatmap, cmap='hot', aspect='equal')
    ax8.set_title('Local Shape DS (Raw Values)', fontsize=12)
    plt.colorbar(im8, ax=ax8, fraction=0.046)
    
    # Anomaly detection comparison (third row)
    threshold_percentile = 80
    
    ax9 = fig.add_subplot(3, 4, 9)
    height_threshold = np.percentile(height_differences, threshold_percentile)
    height_anomalies = (height_heatmap > height_threshold).astype(float)
    im9 = ax9.imshow(height_anomalies, cmap='RdBu_r', aspect='equal', vmin=0, vmax=1)
    ax9.set_title('Height Anomalies (Top 20%)', fontsize=12)
    
    ax10 = fig.add_subplot(3, 4, 10)
    ds_threshold = np.percentile(ds_differences, threshold_percentile)
    ds_anomalies = (ds_heatmap > ds_threshold).astype(float)
    im10 = ax10.imshow(ds_anomalies, cmap='RdBu_r', aspect='equal', vmin=0, vmax=1)
    ax10.set_title('Original DS Anomalies (Top 20%)', fontsize=12)
    
    ax11 = fig.add_subplot(3, 4, 11)
    modified_ds_threshold = np.percentile(modified_ds_differences, threshold_percentile)
    modified_ds_anomalies = (modified_ds_heatmap > modified_ds_threshold).astype(float)
    im11 = ax11.imshow(modified_ds_anomalies, cmap='RdBu_r', aspect='equal', vmin=0, vmax=1)
    ax11.set_title('Modified DS Anomalies (Top 20%)', fontsize=12)
    
    ax12 = fig.add_subplot(3, 4, 12)
    local_shape_threshold = np.percentile(local_shape_ds_differences, threshold_percentile)
    local_shape_anomalies = (local_shape_ds_heatmap > local_shape_threshold).astype(float)
    im12 = ax12.imshow(local_shape_anomalies, cmap='RdBu_r', aspect='equal', vmin=0, vmax=1)
    ax12.set_title('Local Shape DS Anomalies (Top 20%)', fontsize=12)
    
    # Add grid lines to all subplots
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]:
        ax.set_xticks(np.arange(-.5, n_patches_x, 1), minor=True)
        ax.set_yticks(np.arange(-.5, n_patches_y, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    if not os.path.exists('graph_output'):
        os.makedirs('graph_output')
    
    plt.savefig(os.path.join('graph_output', f'all_methods_comparison_{n_patches_x}x{n_patches_y}.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()  # Free memory
    
    # Calculate correlations
    corr_height_ds = np.corrcoef(height_norm.flatten(), ds_norm.flatten())[0, 1]
    corr_height_modified = np.corrcoef(height_norm.flatten(), modified_ds_norm.flatten())[0, 1]
    corr_height_local_shape = np.corrcoef(height_norm.flatten(), local_shape_norm.flatten())[0, 1]
    
    # Print comparison statistics
    print("\nComparison Statistics:")
    print(f"Correlation with Height-Based Method:")
    print(f"  - Original DS: {corr_height_ds:.3f}")
    print(f"  - Modified DS: {corr_height_modified:.3f}")
    print(f"  - Local Shape DS: {corr_height_local_shape:.3f}")
    
    print(f"\nLocal Shape DS Statistics:")
    print(f"Range: [{np.min(local_shape_ds_differences):.6f}, {np.max(local_shape_ds_differences):.6f}]")
    print(f"Mean: {np.mean(local_shape_ds_differences):.6f}")
    print(f"Std: {np.std(local_shape_ds_differences):.6f}")


def save_analysis_results(results_dict, filename):
    """Save analysis results to CSV file."""
    rows = []
    for patch_idx, data in results_dict.items():
        row = {'patch_idx': patch_idx}
        row.update(data)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Saved analysis results to: {filename}")


def save_open3d_visualization(geometry, window_name, filename):
    """Save Open3D visualization to file."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, visible=False)
    vis.add_geometry(geometry)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(filename)
    vis.destroy_window()
    print(f"Saved visualization: {filename}")


# Main Workflow
if __name__ == "__main__":
    # Create output directory and setup logging
    if not os.path.exists('graph_output'):
        os.makedirs('graph_output')
    
    # Setup logger to save console output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join('graph_output', f'analysis_log_{timestamp}.txt')
    logger = Logger(log_filename)
    sys.stdout = logger
    
    print(f"Analysis started at: {datetime.now()}")
    print(f"All outputs will be saved to: graph_output/")
    print("="*60)
    
    # Load mesh and sample points
    ply_path = "C:\\Users\\ibrah\\OneDrive\\Documents\\newpc3.ply"
    
    # Sample parameters
    sample_size = 50000  # Increased for better resolution
    sample_method = 'uniform'  # 'uniform' or 'poisson'
    
    # Load mesh and visualize
    mesh = o3d.io.read_triangle_mesh(ply_path, enable_post_processing=False)
    print("Visualizing original mesh...")
    # Save mesh visualization
    save_open3d_visualization(mesh, "Original Mesh", 
                            os.path.join('graph_output', f'original_mesh_{timestamp}.png'))
    # Also show interactively
    o3d.visualization.draw_geometries([mesh], window_name="Original Mesh")
    
    # Sample points from mesh surface
    sampled_points = sample_points_from_mesh(ply_path, sample_size, sample_method)
    
    # Convert to point cloud for visualization
    sampled_pcd = numpy_to_pointcloud(sampled_points)
    print("\nVisualizing sampled point cloud...")
    # Save point cloud visualization
    save_open3d_visualization(sampled_pcd, "Sampled Point Cloud",
                            os.path.join('graph_output', f'sampled_pointcloud_{timestamp}.png'))
    # Also show interactively
    o3d.visualization.draw_geometries([sampled_pcd], window_name="Sampled Point Cloud", point_show_normal=False)
    
    # Analyze Z distribution to understand the surface
    print("\nAnalyzing Z distribution...")
    analyze_z_distribution(sampled_points, timestamp=timestamp)
    
    # Optionally save sampled points to CSV
    save_csv = True
    if save_csv:
        csv_output_path = os.path.join('graph_output', f'sampled_points_{timestamp}.csv')
        save_sampled_points_to_csv(sampled_points, csv_output_path)
    
    # Configuration
    cfg = Confing()
    cfg.subspace_dim = 3  # Keep for comparison if needed
    
    # Fixed patch resolution (10x10)
    n_patches_x, n_patches_y = 30, 30
    
    print(f"\n{'='*60}")
    print(f"Analyzing with {n_patches_x}x{n_patches_y} patches")
    print(f"{'='*60}")
    
    # Divide into patches
    print(f"\nDividing sampled points into {n_patches_x}x{n_patches_y} patches...")
    print(f"Expected points per patch: ~{len(sampled_points) // (n_patches_x * n_patches_y)}")
    
    patches = divide_into_equal_patches(sampled_points, n_patches_x, n_patches_y)
    
    # Check actual patch sizes
    patch_sizes = [len(patch) for patch in patches]
    print(f"Actual patch sizes - Min: {min(patch_sizes)}, Max: {max(patch_sizes)}, Avg: {np.mean(patch_sizes):.1f}")
    
    # Visualize patches
    visualize_patches(patches, n_patches_x, n_patches_y, f"height_based_{n_patches_x}x{n_patches_y}")
    
    # Calculate height-based differences
    print("\nCalculating height-based differences...")
    differences, patch_stats = calculate_height_based_differences(patches, n_patches_x, n_patches_y)
    
    # Visualize the results
    heatmap = visualize_height_based_differences(patches, differences, patch_stats, 
                                                 n_patches_x, n_patches_y, 
                                                 f"height_based_{n_patches_x}x{n_patches_y}")
    
    # Find and analyze patches with highest differences (potential bumps)
    top_patches = np.argsort(differences)[-5:]  # Top 5 patches
    print(f"\nTop 5 patches with highest differences (potential bumps):")
    
    # Save analysis results
    analysis_results = {}
    for patch_idx in range(len(patches)):
        analysis_results[patch_idx] = {
            'height_difference': differences[patch_idx],
            'mean_z': patch_stats[patch_idx]['mean_z'],
            'max_z': patch_stats[patch_idx]['max_z'],
            'std_z': patch_stats[patch_idx]['std_z'],
            'range_z': patch_stats[patch_idx]['range_z']
        }
    
    for rank, patch_idx in enumerate(reversed(top_patches)):
        row = patch_idx // n_patches_x
        col = patch_idx % n_patches_x
        print(f"{rank+1}. Patch {patch_idx} (Row {row}, Col {col}) - "
              f"Difference: {differences[patch_idx]:.6f}, "
              f"Mean Z: {patch_stats[patch_idx]['mean_z']:.6f}, "
              f"Max Z: {patch_stats[patch_idx]['max_z']:.6f}")
    
    # Compare with DS method
    print("\n\nComparing with DS method...")
    ds_differences = calculate_patch_differences_with_ds_8neighbors(patches, cfg, n_patches_x, n_patches_y)
    
    # Also run modified DS method
    print("\nRunning Modified DS method (height deviations)...")
    modified_ds_differences = calculate_patch_differences_with_modified_ds(patches, cfg, n_patches_x, n_patches_y)        
    
    # Run local shape DS method
    print("\nRunning Local Shape DS method (curvature features)...")
    local_shape_ds_differences = calculate_patch_differences_with_local_shape_ds(patches, cfg, n_patches_x, n_patches_y)
    
    # Add DS results to analysis_results
    for patch_idx in range(len(patches)):
        analysis_results[patch_idx]['ds_magnitude'] = ds_differences[patch_idx]
        analysis_results[patch_idx]['modified_ds_magnitude'] = modified_ds_differences[patch_idx]
        analysis_results[patch_idx]['local_shape_ds_magnitude'] = local_shape_ds_differences[patch_idx]
    
    # Save complete analysis results
    results_filename = os.path.join('graph_output', 
                                  f'analysis_results_{n_patches_x}x{n_patches_y}_{timestamp}.csv')
    save_analysis_results(analysis_results, results_filename)
    
    # Visualize DS results
    ds_heatmap = visualize_ds_differences(patches, ds_differences, n_patches_x, n_patches_y, 
                           f"ds_method_{n_patches_x}x{n_patches_y}")
    
    # Visualize Modified DS results
    modified_ds_heatmap = visualize_ds_differences(patches, modified_ds_differences, n_patches_x, n_patches_y, 
                                     f"modified_ds_method_{n_patches_x}x{n_patches_y}")
    local_shape_ds_heatmap = visualize_ds_differences(patches, local_shape_ds_differences, n_patches_x, n_patches_y, 
                                 f"local_shape_ds_method_{n_patches_x}x{n_patches_y}")

    # Compare all methods
    compare_all_methods(differences, ds_differences, modified_ds_differences, 
               local_shape_ds_differences, n_patches_x, n_patches_y)
    
    # Quick comparison
    height_norm = (differences - np.min(differences)) / (np.max(differences) - np.min(differences) + 1e-8)
    ds_norm = (ds_differences - np.min(ds_differences)) / (np.max(ds_differences) - np.min(ds_differences) + 1e-8)
    modified_ds_norm = (modified_ds_differences - np.min(modified_ds_differences)) / (np.max(modified_ds_differences) - np.min(modified_ds_differences) + 1e-8)
    local_shape_norm = (local_shape_ds_differences - np.min(local_shape_ds_differences)) / (np.max(local_shape_ds_differences) - np.min(local_shape_ds_differences) + 1e-8)
    correlation_original = np.corrcoef(height_norm, ds_norm)[0, 1]
    correlation_modified = np.corrcoef(height_norm, modified_ds_norm)[0, 1]
    correlation_local_shape = np.corrcoef(height_norm, local_shape_norm)[0, 1]
    print(f"\nCorrelation between height-based and original DS: {correlation_original:.3f}")
    print(f"Correlation between height-based and modified DS: {correlation_modified:.3f}")
    print(f"Correlation between height-based and local shape DS: {correlation_local_shape:.3f}")
    print(f"Improvement (modified vs original): {correlation_modified - correlation_original:.3f}")

    # Side-by-side comparison (original method)
    compare_height_and_ds_heatmaps(differences, ds_differences, n_patches_x, n_patches_y)
    
    # Create a comprehensive summary report
    summary_filename = os.path.join('graph_output', f'analysis_summary_{timestamp}.txt')
    with open(summary_filename, 'w') as f:
        f.write(f"3D Mesh Surface Analysis Summary\n")
        f.write(f"{'='*60}\n")
        f.write(f"Analysis Date: {datetime.now()}\n")
        f.write(f"Input File: {ply_path}\n")
        f.write(f"Sample Size: {sample_size} points\n")
        f.write(f"Sample Method: {sample_method}\n")
        f.write(f"Patch Resolution: {n_patches_x}x{n_patches_y}\n")
        f.write(f"\nKey Findings:\n")
        f.write(f"  - Z range: {sampled_points[:, 2].min():.6f} to {sampled_points[:, 2].max():.6f}\n")
        f.write(f"  - Z standard deviation: {sampled_points[:, 2].std():.6f}\n")
        f.write(f"\nMethod Correlations with Height-Based:\n")
        f.write(f"  - Original DS: {correlation_original:.3f}\n")
        f.write(f"  - Modified DS: {correlation_modified:.3f}\n") 
        f.write(f"  - Local Shape DS: {correlation_local_shape:.3f}\n")
        f.write(f"\nOutput Files Generated:\n")
        for file in sorted(os.listdir('graph_output')):
            if timestamp in file:
                f.write(f"  - {file}\n")
    
    print(f"\nSummary report saved as: {summary_filename}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Analysis completed at: {datetime.now()}")
    print(f"\nAll outputs have been saved to: graph_output/")
    print("\nGenerated files:")
    for file in sorted(os.listdir('graph_output')):
        if file.startswith(('original_mesh_', 'sampled_', 'z_distribution', 
                          'height_', 'ds_', 'modified_', 'all_methods', 
                          'patch_', 'local_', 'global_', 'analysis_log_')):
            print(f"  - {file}")
    
    # Close logger
    print(f"\nLog file saved as: {log_filename}")
    sys.stdout = sys.__stdout__
    logger.close()