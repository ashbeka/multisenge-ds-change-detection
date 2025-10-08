import numpy as np
import matplotlib.pyplot as plt
from grid_patch_division_with_stride import divide_into_grid_patches_with_stride, visualize_grid_patches_with_overlap
from scipy.linalg import svd
import os


class Config:
    def __init__(self):
        self.subspace_dim = 3


def generate_subspace(data_segment, cfg):
    """Generate a subspace from a segment using SVD."""
    if len(data_segment) < cfg.subspace_dim + 1:
        return np.zeros((data_segment.shape[0], cfg.subspace_dim))
    
    X = data_segment
    mv = np.mean(X, axis=0)
    X_centered = X - mv
    
    U, _, _ = svd(X_centered, full_matrices=False)
    n_components = min(cfg.subspace_dim, U.shape[1])
    return U[:, 0:n_components]


def calculate_height_based_differences_grid(patches, patch_info):
    """
    Calculate height-based differences for grid-based patches with overlap support.
    
    Args:
        patches: list of patches
        patch_info: metadata from divide_into_grid_patches_with_stride
    
    Returns:
        differences: array of difference scores
        patch_stats: statistics for each patch
    """
    patch_stats = []
    n_patches_x_stride, n_patches_y_stride = patch_info['grid_size']
    
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
                'n_points': len(patch)
            }
            patch_stats.append(stats)
        else:
            patch_stats.append({
                'mean_z': 0, 'std_z': 0, 'max_z': 0, 'min_z': 0,
                'range_z': 0, 'n_points': 0
            })
    
    # Calculate differences from neighbors
    differences = []
    
    for idx, (i, j) in enumerate(patch_info['indices']):
        neighbors = []
        
        # Get all valid neighbors (8-connectivity)
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < n_patches_y_stride and 0 <= nj < n_patches_x_stride:
                    neighbor_idx = ni * n_patches_x_stride + nj
                    if neighbor_idx < len(patches):
                        neighbors.append(neighbor_idx)
        
        if neighbors:
            # Calculate difference in mean height
            current_mean = patch_stats[idx]['mean_z']
            neighbor_means = [patch_stats[n]['mean_z'] for n in neighbors]
            avg_neighbor_mean = np.mean(neighbor_means)
            
            # Height difference metric
            height_diff = abs(current_mean - avg_neighbor_mean)
            
            # Also consider maximum height difference
            current_max = patch_stats[idx]['max_z']
            neighbor_maxs = [patch_stats[n]['max_z'] for n in neighbors]
            avg_neighbor_max = np.mean(neighbor_maxs)
            max_diff = abs(current_max - avg_neighbor_max)
            
            # Standard deviation (surface roughness)
            std_diff = patch_stats[idx]['std_z']
            
            # Combined metric
            combined_diff = height_diff + 0.3 * max_diff + 0.2 * std_diff
            differences.append(combined_diff)
        else:
            differences.append(0)
    
    return differences, patch_stats


def visualize_anomaly_detection_with_overlap(patches, patch_info, differences, patch_stats):
    """Visualize anomaly detection results for overlapping patches."""
    fig = plt.figure(figsize=(20, 10))
    
    n_patches_x_stride, n_patches_y_stride = patch_info['grid_size']
    
    # Convert differences to 2D grid
    diff_grid = np.zeros((n_patches_y_stride, n_patches_x_stride))
    for idx, (i, j) in enumerate(patch_info['indices']):
        diff_grid[i, j] = differences[idx]
    
    # 1. 3D visualization colored by anomaly score
    ax1 = fig.add_subplot(231, projection='3d')
    
    # Normalize differences for coloring
    norm_diffs = (differences - np.min(differences)) / (np.max(differences) - np.min(differences) + 1e-8)
    colors = plt.cm.hot(norm_diffs)
    
    for idx, patch in enumerate(patches):
        if len(patch) > 0:
            ax1.scatter(patch[:, 0], patch[:, 1], patch[:, 2],
                       c=[colors[idx]], alpha=0.6, s=1)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D View Colored by Anomaly Score')
    
    # 2. Anomaly heatmap
    ax2 = fig.add_subplot(232)
    im = ax2.imshow(diff_grid, cmap='hot', aspect='equal', origin='lower')
    ax2.set_title('Anomaly Score Heatmap')
    ax2.set_xlabel('Patch X')
    ax2.set_ylabel('Patch Y')
    plt.colorbar(im, ax=ax2, label='Anomaly Score')
    
    # 3. Height statistics
    ax3 = fig.add_subplot(233)
    mean_heights = np.array([s['mean_z'] for s in patch_stats])
    mean_grid = np.zeros((n_patches_y_stride, n_patches_x_stride))
    for idx, (i, j) in enumerate(patch_info['indices']):
        mean_grid[i, j] = mean_heights[idx]
    
    im3 = ax3.imshow(mean_grid, cmap='terrain', aspect='equal', origin='lower')
    ax3.set_title('Mean Height per Patch')
    ax3.set_xlabel('Patch X')
    ax3.set_ylabel('Patch Y')
    plt.colorbar(im3, ax=ax3, label='Mean Z')
    
    # 4. Point density (showing overlap effect)
    ax4 = fig.add_subplot(234)
    point_counts = np.array([s['n_points'] for s in patch_stats])
    count_grid = np.zeros((n_patches_y_stride, n_patches_x_stride))
    for idx, (i, j) in enumerate(patch_info['indices']):
        count_grid[i, j] = point_counts[idx]
    
    im4 = ax4.imshow(count_grid, cmap='viridis', aspect='equal', origin='lower')
    ax4.set_title('Points per Patch')
    ax4.set_xlabel('Patch X')
    ax4.set_ylabel('Patch Y')
    plt.colorbar(im4, ax=ax4, label='Point Count')
    
    # 5. Top anomalies
    ax5 = fig.add_subplot(235)
    threshold = np.percentile(differences, 80)
    anomaly_mask = diff_grid > threshold
    ax5.imshow(anomaly_mask, cmap='RdYlBu_r', aspect='equal', origin='lower')
    ax5.set_title('Detected Anomalies (Top 20%)')
    ax5.set_xlabel('Patch X')
    ax5.set_ylabel('Patch Y')
    
    # 6. Overlap visualization
    ax6 = fig.add_subplot(236)
    ax6.text(0.1, 0.8, f"Grid size: {n_patches_x_stride} x {n_patches_y_stride}", transform=ax6.transAxes)
    ax6.text(0.1, 0.7, f"Total patches: {len(patches)}", transform=ax6.transAxes)
    ax6.text(0.1, 0.6, f"Overlap X: {patch_info['overlap_x']*100:.0f}%", transform=ax6.transAxes)
    ax6.text(0.1, 0.5, f"Overlap Y: {patch_info['overlap_y']*100:.0f}%", transform=ax6.transAxes)
    ax6.text(0.1, 0.4, f"Stride: ({patch_info['stride'][0]:.3f}, {patch_info['stride'][1]:.3f})", transform=ax6.transAxes)
    ax6.text(0.1, 0.3, f"Anomalies found: {np.sum(anomaly_mask)}", transform=ax6.transAxes)
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.set_xticks([])
    ax6.set_yticks([])
    ax6.set_title('Analysis Summary')
    
    plt.tight_layout()
    plt.show()
    
    return diff_grid, anomaly_mask


def compare_stride_strategies(points, n_patches_x=10, n_patches_y=10):
    """Compare anomaly detection results with different stride strategies."""
    
    stride_configs = [
        (1.0, 1.0, "No Overlap"),
        (0.8, 0.8, "20% Overlap"),
        (0.5, 0.5, "50% Overlap"),
        (0.3, 0.3, "70% Overlap")
    ]
    
    results = []
    
    for stride_x, stride_y, label in stride_configs:
        print(f"\n{'-'*60}")
        print(f"Testing: {label}")
        
        # Divide into patches
        patches, patch_info = divide_into_grid_patches_with_stride(
            points, n_patches_x, n_patches_y, stride_x, stride_y
        )
        
        # Calculate anomaly scores
        differences, patch_stats = calculate_height_based_differences_grid(patches, patch_info)
        
        # Find anomalies
        threshold = np.percentile(differences, 80)
        n_anomalies = np.sum(np.array(differences) > threshold)
        
        # Store results
        results.append({
            'label': label,
            'stride': (stride_x, stride_y),
            'n_patches': len(patches),
            'differences': differences,
            'patch_stats': patch_stats,
            'patch_info': patch_info,
            'n_anomalies': n_anomalies,
            'max_score': np.max(differences),
            'mean_score': np.mean(differences),
            'std_score': np.std(differences)
        })
        
        print(f"  Patches: {len(patches)}")
        print(f"  Anomalies detected: {n_anomalies}")
        print(f"  Max anomaly score: {np.max(differences):.4f}")
        print(f"  Mean anomaly score: {np.mean(differences):.4f}")
    
    # Visualization comparing all strategies
    fig, axes = plt.subplots(2, len(stride_configs), figsize=(20, 10))
    
    for idx, result in enumerate(results):
        # Anomaly heatmap
        ax_heat = axes[0, idx]
        n_x, n_y = result['patch_info']['grid_size']
        diff_grid = np.zeros((n_y, n_x))
        
        for j, (i, k) in enumerate(result['patch_info']['indices']):
            diff_grid[i, k] = result['differences'][j]
        
        im = ax_heat.imshow(diff_grid, cmap='hot', aspect='equal')
        ax_heat.set_title(f"{result['label']}\n{result['n_patches']} patches")
        ax_heat.set_xlabel('X')
        ax_heat.set_ylabel('Y')
        
        # Anomaly detection
        ax_anom = axes[1, idx]
        threshold = np.percentile(result['differences'], 80)
        anomaly_mask = diff_grid > threshold
        ax_anom.imshow(anomaly_mask, cmap='RdBu_r', aspect='equal')
        ax_anom.set_title(f"Anomalies: {result['n_anomalies']}")
        ax_anom.set_xlabel('X')
        ax_anom.set_ylabel('Y')
    
    plt.suptitle('Comparison of Different Stride Strategies', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return results


# Example usage
if __name__ == "__main__":
    # Generate sample data with anomalies
    np.random.seed(42)
    
    # Create base surface
    x = np.linspace(0, 20, 200)
    y = np.linspace(0, 20, 200)
    X, Y = np.meshgrid(x, y)
    
    # Base surface with some variations
    Z = 0.1 * np.sin(0.5 * X) + 0.1 * np.cos(0.5 * Y)
    
    # Add anomalies (bumps)
    anomalies = [
        (5, 5, 1.0, 1.0),   # (x, y, height, width)
        (15, 8, 0.8, 0.8),
        (10, 15, 1.2, 1.5),
        (3, 18, 0.6, 0.7)
    ]
    
    for ax, ay, height, width in anomalies:
        Z += height * np.exp(-((X - ax)**2 + (Y - ay)**2) / (2 * width**2))
    
    # Convert to point cloud
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # Add noise
    points[:, 2] += np.random.normal(0, 0.02, len(points))
    
    print("Testing grid-based anomaly detection with overlapping patches")
    print("="*60)
    
    # Test 1: Single configuration with visualization
    print("\nDetailed analysis with 30% overlap:")
    patches, patch_info = divide_into_grid_patches_with_stride(
        points, n_patches_x=15, n_patches_y=15, stride_x=0.7, stride_y=0.7
    )
    
    differences, patch_stats = calculate_height_based_differences_grid(patches, patch_info)
    diff_grid, anomaly_mask = visualize_anomaly_detection_with_overlap(
        patches, patch_info, differences, patch_stats
    )
    
    # Test 2: Compare different stride strategies
    print("\nComparing different stride strategies:")
    results = compare_stride_strategies(points, n_patches_x=12, n_patches_y=12)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Effect of stride on anomaly detection")
    print("="*60)
    for result in results:
        print(f"\n{result['label']}:")
        print(f"  Total patches: {result['n_patches']}")
        print(f"  Anomalies detected: {result['n_anomalies']}")
        print(f"  Detection sensitivity: {result['std_score']:.4f}")
        print(f"  Computational cost factor: {result['n_patches'] / results[0]['n_patches']:.2f}x")