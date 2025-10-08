import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy.linalg import svd
from config import Confing
import open3d as o3d
import pandas as pd
import pointlib as Pointlib
from mpl_toolkits.mplot3d import Axes3D
import scipy
import os


def generate_subspace(data_segment, cfg):
    """Generate a subspace from a segment using SVD."""
    X = data_segment
    mv = np.mean(X, axis=0)
    X_centered = X - mv
    
    # Use economy mode SVD to handle different sized patches
    U, _, _ = svd(X_centered, full_matrices=False)
    
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


def calculate_patch_differences_with_ds_8neighbors(patches, cfg, n_patches_x=5, n_patches_y=5):
    """Calculate patch differences using difference subspace, considering all 8 neighbors."""
    average_magnitudes = []
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
        
        for neighbor_idx, weight in zip(neighbors, weights):
            # Get the current patch subspace
            S1 = patch_subspaces[i]
            S2 = patch_subspaces[neighbor_idx]
            
            # Generate difference subspace
            DS = gen_shape_difference_subspace(S1, S2, cfg)
            if DS.shape[1] > 0:
                # Ensure DS has the same number of rows as S1
                if DS.shape[0] > S1.shape[0]:
                    DS = DS[:S1.shape[0], :]
                elif DS.shape[0] < S1.shape[0]:
                    DS_padded = np.zeros((S1.shape[0], DS.shape[1]))
                    DS_padded[:DS.shape[0], :] = DS
                    DS = DS_padded
                
                P = DS @ DS.T
                V = P @ S1
                magnitude = np.linalg.norm(V)
                patch_magnitudes.append(magnitude)
        
        if patch_magnitudes:
            avg_magnitude = sum(patch_magnitudes) / sum(weights)
            average_magnitudes.append(avg_magnitude)
        else:
            average_magnitudes.append(0)
    
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


def analyze_z_distribution(points, n_bins=50):
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
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D view colored by height')
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    if not os.path.exists('graph_output'):
        os.makedirs('graph_output')
    
    plt.savefig(os.path.join('graph_output', 'z_distribution_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
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
    
    # Print comparison statistics
    print("\nComparison Statistics:")
    print(f"Patches detected by both methods: {np.sum(agreement)}")
    print(f"Patches detected only by height: {np.sum((height_anomalies == 1) & (ds_anomalies == 0))}")
    print(f"Patches detected only by DS: {np.sum((height_anomalies == 0) & (ds_anomalies == 1))}")
    print(f"Agreement percentage: {np.sum(height_anomalies == ds_anomalies) / len(height_anomalies.flatten()) * 100:.1f}%")


def visualize_specific_patch_height_analysis(patches, patch_idx, patch_stats, n_patches_x=10, n_patches_y=10):
    """Detailed visualization of a specific patch and its neighbors with height analysis."""
    row = patch_idx // n_patches_x
    col = patch_idx % n_patches_x
    
    # Get all 8 neighbors
    neighbors = []
    directions = []
    for dr, dc, direction in [(-1,-1,'NW'), (-1,0,'N'), (-1,1,'NE'), 
                              (0,-1,'W'), (0,1,'E'),
                              (1,-1,'SW'), (1,0,'S'), (1,1,'SE')]:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < n_patches_y and 0 <= new_col < n_patches_x:
            neighbor_idx = new_row * n_patches_x + new_col
            neighbors.append(neighbor_idx)
            directions.append(direction)
    
    fig = plt.figure(figsize=(15, 10))
    
    # 3D visualization
    ax_3d = fig.add_subplot(221, projection='3d')
    
    # Plot central patch
    central_patch = patches[patch_idx]
    ax_3d.scatter(central_patch[:, 0], central_patch[:, 1], central_patch[:, 2],
                 c='red', s=2, label=f'Patch {patch_idx}')
    
    # Plot neighbors
    colors = plt.cm.tab10(np.linspace(0, 1, len(neighbors)))
    for neighbor_idx, direction, color in zip(neighbors, directions, colors):
        neighbor_patch = patches[neighbor_idx]
        ax_3d.scatter(neighbor_patch[:, 0], neighbor_patch[:, 1], neighbor_patch[:, 2],
                     c=[color], s=2, label=f'{direction} ({neighbor_idx})')
    
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.set_title('Patch and Neighbors 3D View')
    ax_3d.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Height statistics comparison
    ax_stats = fig.add_subplot(222)
    
    # Prepare data for bar chart
    labels = ['Central'] + directions
    mean_heights = [patch_stats[patch_idx]['mean_z']] + [patch_stats[n]['mean_z'] for n in neighbors]
    max_heights = [patch_stats[patch_idx]['max_z']] + [patch_stats[n]['max_z'] for n in neighbors]
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax_stats.bar(x - width/2, mean_heights, width, label='Mean Height')
    bars2 = ax_stats.bar(x + width/2, max_heights, width, label='Max Height')
    
    ax_stats.set_xlabel('Patch')
    ax_stats.set_ylabel('Height (Z)')
    ax_stats.set_title('Height Comparison')
    ax_stats.set_xticks(x)
    ax_stats.set_xticklabels(labels, rotation=45)
    ax_stats.legend()
    
    # Grid visualization with values
    ax_grid = fig.add_subplot(223)
    grid_values = np.zeros((n_patches_y, n_patches_x))
    
    # Fill grid with mean heights
    for i in range(len(patches)):
        r = i // n_patches_x
        c = i % n_patches_x
        grid_values[r, c] = patch_stats[i]['mean_z']
    
    im = ax_grid.imshow(grid_values, cmap='terrain')
    
    # Highlight central patch and neighbors
    rect = plt.Rectangle((col-0.5, row-0.5), 1, 1, fill=False, edgecolor='red', linewidth=3)
    ax_grid.add_patch(rect)
    
    for neighbor_idx, direction in zip(neighbors, directions):
        n_row = neighbor_idx // n_patches_x
        n_col = neighbor_idx % n_patches_x
        rect = plt.Rectangle((n_col-0.5, n_row-0.5), 1, 1, fill=False, edgecolor='blue', linewidth=2)
        ax_grid.add_patch(rect)
    
    ax_grid.set_title('Mean Height Grid (Red: Central, Blue: Neighbors)')
    plt.colorbar(im, ax=ax_grid, label='Mean Height')
    
    # Height difference analysis
    ax_diff = fig.add_subplot(224)
    
    central_mean = patch_stats[patch_idx]['mean_z']
    height_diffs = [abs(patch_stats[n]['mean_z'] - central_mean) for n in neighbors]
    
    bars = ax_diff.bar(directions, height_diffs, color='orange')
    ax_diff.set_xlabel('Neighbor Direction')
    ax_diff.set_ylabel('Absolute Height Difference')
    ax_diff.set_title(f'Height Differences from Patch {patch_idx}')
    ax_diff.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, diff in zip(bars, height_diffs):
        height = bar.get_height()
        ax_diff.text(bar.get_x() + bar.get_width()/2., height,
                    f'{diff:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    if not os.path.exists('graph_output'):
        os.makedirs('graph_output')
    
    plt.savefig(os.path.join('graph_output', f'patch_{patch_idx}_height_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed statistics
    print(f"\nDetailed height analysis for Patch {patch_idx} (Row {row}, Col {col}):")
    print(f"Central patch - Mean Z: {patch_stats[patch_idx]['mean_z']:.6f}, "
          f"Max Z: {patch_stats[patch_idx]['max_z']:.6f}, "
          f"Std Z: {patch_stats[patch_idx]['std_z']:.6f}")
    print("\nNeighbor analysis:")
    for neighbor_idx, direction in zip(neighbors, directions):
        diff = abs(patch_stats[neighbor_idx]['mean_z'] - patch_stats[patch_idx]['mean_z'])
        print(f"{direction} (Patch {neighbor_idx}) - "
              f"Mean Z: {patch_stats[neighbor_idx]['mean_z']:.6f}, "
              f"Difference: {diff:.6f}")


def extract_local_region(points, center_row, center_col, n_patches_x, n_patches_y, expansion_factor=2):
    """Extract points around a specific patch location with expanded area."""
    # Calculate approximate bounds for the patch
    x_sorted = np.sort(points[:, 0])
    y_sorted = np.sort(points[:, 1])
    
    x_min, x_max = x_sorted[0], x_sorted[-1]
    y_min, y_max = y_sorted[0], y_sorted[-1]
    
    patch_width = (x_max - x_min) / n_patches_x
    patch_height = (y_max - y_min) / n_patches_y
    
    # Calculate center and bounds with expansion
    center_x = x_min + (center_col + 0.5) * patch_width
    center_y = y_min + (center_row + 0.5) * patch_height
    
    roi_width = patch_width * expansion_factor
    roi_height = patch_height * expansion_factor
    
    # Extract points within ROI
    mask = ((points[:, 0] >= center_x - roi_width/2) & 
            (points[:, 0] <= center_x + roi_width/2) &
            (points[:, 1] >= center_y - roi_height/2) & 
            (points[:, 1] <= center_y + roi_height/2))
    
    return points[mask], center_x, center_y


def hybrid_focused_ds_analysis(points, height_differences, patch_stats, cfg, 
                             n_patches_global=10, n_patches_local=5, top_k=5):
    """
    Perform focused DS analysis on regions identified by height-based detection.
    
    Args:
        points: Original point cloud
        height_differences: Results from height-based analysis
        patch_stats: Statistics from height-based analysis
        cfg: Configuration with subspace_dim
        n_patches_global: Number of patches in global analysis
        n_patches_local: Number of patches for local DS analysis
        top_k: Number of top anomalies to analyze
    """
    print(f"\n{'='*60}")
    print("HYBRID FOCUSED DS ANALYSIS")
    print(f"{'='*60}")
    
    # Find top anomalous patches
    top_patches = np.argsort(height_differences)[-top_k:]
    
    results = []
    
    for rank, patch_idx in enumerate(reversed(top_patches)):
        row = patch_idx // n_patches_global
        col = patch_idx % n_patches_global
        
        print(f"\n--- Analyzing Anomaly {rank+1}: Patch {patch_idx} (Row {row}, Col {col}) ---")
        print(f"Height difference score: {height_differences[patch_idx]:.6f}")
        print(f"Mean Z: {patch_stats[patch_idx]['mean_z']:.6f}")
        
        # Extract local region
        local_points, center_x, center_y = extract_local_region(
            points, row, col, n_patches_global, n_patches_global, expansion_factor=3
        )
        
        print(f"Extracted {len(local_points)} points for local analysis")
        
        if len(local_points) < 100:  # Not enough points
            print("Warning: Not enough points for local analysis")
            continue
        
        # Create local patches
        local_patches = divide_into_equal_patches(local_points, n_patches_local, n_patches_local)
        
        # Run DS analysis on local patches
        local_ds_differences = calculate_patch_differences_with_ds_8neighbors(
            local_patches, cfg, n_patches_local, n_patches_local
        )
        
        # Visualize local DS analysis
        visualize_local_ds_analysis(local_patches, local_ds_differences, n_patches_local, 
                                   patch_idx, center_x, center_y)
        
        # Store results
        results.append({
            'global_patch_idx': patch_idx,
            'center': (center_x, center_y),
            'local_ds_differences': local_ds_differences,
            'local_patches': local_patches,
            'height_score': height_differences[patch_idx]
        })
    
    return results


def visualize_local_ds_analysis(local_patches, ds_differences, n_patches_local, 
                               global_patch_idx, center_x, center_y):
    """Visualize the focused DS analysis results for a local region."""
    fig = plt.figure(figsize=(15, 5))
    
    # Convert to heatmap
    ds_heatmap = np.array(ds_differences).reshape(n_patches_local, n_patches_local)
    
    # 1. 3D view of local patches
    ax_3d = fig.add_subplot(131, projection='3d')
    colors = plt.cm.hot(ds_differences / np.max(ds_differences))
    
    for idx, patch in enumerate(local_patches):
        if len(patch) > 0:
            ax_3d.scatter(patch[:, 0], patch[:, 1], patch[:, 2],
                         c=[colors[idx]], alpha=0.8, s=5)
    
    ax_3d.set_title(f'Local DS Analysis around Patch {global_patch_idx}')
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    
    # 2. Local DS heatmap
    ax_heat = fig.add_subplot(132)
    im = ax_heat.imshow(ds_heatmap, cmap='hot', aspect='equal')
    ax_heat.set_title(f'Local DS Heatmap ({n_patches_local}x{n_patches_local})')
    plt.colorbar(im, ax=ax_heat, label='DS Magnitude')
    
    # Add grid
    ax_heat.set_xticks(np.arange(-.5, n_patches_local, 1), minor=True)
    ax_heat.set_yticks(np.arange(-.5, n_patches_local, 1), minor=True)
    ax_heat.grid(which="minor", color="w", linestyle='-', linewidth=1)
    
    # 3. DS profile plot
    ax_profile = fig.add_subplot(133)
    
    # Get center patch and its DS value
    center_patch_idx = (n_patches_local // 2) * n_patches_local + (n_patches_local // 2)
    
    # Plot radial profile
    distances = []
    ds_values = []
    
    for i in range(n_patches_local):
        for j in range(n_patches_local):
            patch_idx = i * n_patches_local + j
            dist = np.sqrt((i - n_patches_local//2)**2 + (j - n_patches_local//2)**2)
            distances.append(dist)
            ds_values.append(ds_differences[patch_idx])
    
    ax_profile.scatter(distances, ds_values, alpha=0.6)
    ax_profile.set_xlabel('Distance from Center (patches)')
    ax_profile.set_ylabel('DS Magnitude')
    ax_profile.set_title('DS Magnitude vs Distance from Anomaly Center')
    
    # Fit and plot trend line
    if len(distances) > 3:
        z = np.polyfit(distances, ds_values, 2)
        p = np.poly1d(z)
        x_trend = np.linspace(0, max(distances), 100)
        ax_profile.plot(x_trend, p(x_trend), 'r-', alpha=0.8, label='Trend')
        ax_profile.legend()
    
    plt.tight_layout()
    
    # Save figure
    if not os.path.exists('graph_output'):
        os.makedirs('graph_output')
    
    plt.savefig(os.path.join('graph_output', f'local_ds_analysis_patch_{global_patch_idx}.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()


def compare_global_local_ds(global_ds_differences, local_results, n_patches_global):
    """Compare global and local DS analysis results."""
    fig, axes = plt.subplots(2, len(local_results), figsize=(5*len(local_results), 10))
    
    if len(local_results) == 1:
        axes = axes.reshape(2, 1)
    
    for idx, result in enumerate(local_results):
        patch_idx = result['global_patch_idx']
        row = patch_idx // n_patches_global
        col = patch_idx % n_patches_global
        
        # Global context
        ax_global = axes[0, idx]
        global_heatmap = np.array(global_ds_differences).reshape(n_patches_global, n_patches_global)
        im1 = ax_global.imshow(global_heatmap, cmap='hot')
        
        # Highlight the analyzed patch
        rect = plt.Rectangle((col-0.5, row-0.5), 1, 1, fill=False, 
                           edgecolor='cyan', linewidth=3)
        ax_global.add_patch(rect)
        
        ax_global.set_title(f'Global DS (Patch {patch_idx} highlighted)')
        plt.colorbar(im1, ax=ax_global, fraction=0.046)
        
        # Local detail
        ax_local = axes[1, idx]
        local_heatmap = np.array(result['local_ds_differences']).reshape(
            int(np.sqrt(len(result['local_ds_differences']))), -1
        )
        im2 = ax_local.imshow(local_heatmap, cmap='hot')
        ax_local.set_title(f'Local DS Detail')
        plt.colorbar(im2, ax=ax_local, fraction=0.046)
        
        # Add grid
        for ax in [ax_global, ax_local]:
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    if not os.path.exists('graph_output'):
        os.makedirs('graph_output')
    
    plt.savefig(os.path.join('graph_output', 'global_vs_local_ds_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()


# Main Workflow
if __name__ == "__main__":
    # Load mesh and sample points
    ply_path = "C:\\Users\\ibrah\\OneDrive\\Documents\\newpc.ply"
    
    # Sample parameters
    sample_size = 50000  # Increased for better resolution
    sample_method = 'uniform'  # 'uniform' or 'poisson'
    
    # Load mesh and visualize
    mesh = o3d.io.read_triangle_mesh(ply_path, enable_post_processing=False)
    print("Visualizing original mesh...")
    o3d.visualization.draw_geometries([mesh], window_name="Original Mesh")
    
    # Sample points from mesh surface
    sampled_points = sample_points_from_mesh(ply_path, sample_size, sample_method)
    
    # Convert to point cloud for visualization
    sampled_pcd = numpy_to_pointcloud(sampled_points)
    print("\nVisualizing sampled point cloud...")
    o3d.visualization.draw_geometries([sampled_pcd], window_name="Sampled Point Cloud", point_show_normal=False)
    
    # Analyze Z distribution to understand the surface
    print("\nAnalyzing Z distribution...")
    analyze_z_distribution(sampled_points)
    
    # Optionally save sampled points to CSV
    save_csv = True
    if save_csv:
        csv_output_path = "C:\\Users\\ibrah\\OneDrive\\Desktop\\Realsense\\height_analysis_sampled_points.csv"
        save_sampled_points_to_csv(sampled_points, csv_output_path)
    
    # Configuration
    cfg = Confing()
    cfg.subspace_dim = 3  # Keep for comparison if needed
    
    # Try different patch resolutions
    patch_resolutions = [(10, 10), (20, 20), (30, 30)]
    
    for n_patches_x, n_patches_y in patch_resolutions:
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
        for rank, patch_idx in enumerate(reversed(top_patches)):
            row = patch_idx // n_patches_x
            col = patch_idx % n_patches_x
            print(f"{rank+1}. Patch {patch_idx} (Row {row}, Col {col}) - "
                  f"Difference: {differences[patch_idx]:.6f}, "
                  f"Mean Z: {patch_stats[patch_idx]['mean_z']:.6f}, "
                  f"Max Z: {patch_stats[patch_idx]['max_z']:.6f}")
        
        # Analyze the patch with highest difference in detail
        if len(top_patches) > 0:
            most_anomalous_patch = top_patches[-1]
            print(f"\nAnalyzing most anomalous patch (#{most_anomalous_patch}) in detail...")
            visualize_specific_patch_height_analysis(patches, most_anomalous_patch, patch_stats, 
                                                   n_patches_x, n_patches_y)
        
        # Compare with DS method
        print("\n\nComparing with DS method...")
        ds_differences = calculate_patch_differences_with_ds_8neighbors(patches, cfg, n_patches_x, n_patches_y)
        
        # Visualize DS results
        ds_heatmap = visualize_ds_differences(patches, ds_differences, n_patches_x, n_patches_y, 
                               f"ds_method_{n_patches_x}x{n_patches_y}")
        
        # Quick comparison
        height_norm = (differences - np.min(differences)) / (np.max(differences) - np.min(differences) + 1e-8)
        ds_norm = (ds_differences - np.min(ds_differences)) / (np.max(ds_differences) - np.min(ds_differences) + 1e-8)
        correlation = np.corrcoef(height_norm, ds_norm)[0, 1]
        print(f"Correlation between height-based and DS methods: {correlation:.3f}")
        
        # Side-by-side comparison
        compare_height_and_ds_heatmaps(differences, ds_differences, n_patches_x, n_patches_y)
    
    # Run hybrid focused analysis on the finest resolution that still has enough points
    if n_patches_x == 20 and n_patches_y == 20:  # Run on 20x20 for good balance
        print(f"\n{'='*60}")
        print("RUNNING HYBRID FOCUSED DS ANALYSIS")
        print(f"{'='*60}")
        
        # Perform focused DS analysis on top anomalies
        local_results = hybrid_focused_ds_analysis(
            sampled_points,  # Use original points, not patches
            differences, 
            patch_stats,
            cfg,
            n_patches_global=n_patches_x,
            n_patches_local=5,  # 5x5 local patches
            top_k=3  # Analyze top 3 anomalies
        )
        
        # Compare global vs local DS
        if local_results:
            compare_global_local_ds(ds_differences, local_results, n_patches_x)