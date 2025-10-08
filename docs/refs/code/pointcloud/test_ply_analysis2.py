import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig, svd
import open3d as o3d
import os

# Import functions from main.py
from main import (generate_subspace, gen_shape_difference_subspace, 
                  visualize_patch_differences, divide_into_equal_patches,
                  calculate_patch_differences_with_ds_8neighbors)
from grid_patch_division_with_stride import divide_into_grid_patches_with_stride
from config import Confing
import pointlib as Pointlib

def load_ply_file(filename):
    """Load PLY file and return points as numpy array."""
    try:
        pcd = o3d.io.read_point_cloud(filename)
        if len(pcd.points) == 0:
            raise ValueError("PLY file is empty or couldn't be read")
        
        points = np.asarray(pcd.points)
        print(f"Loaded PLY file: {filename}")
        print(f"  Number of points: {len(points)}")
        print(f"  X range: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
        print(f"  Y range: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
        print(f"  Z range: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
        
        return points, pcd
    except Exception as e:
        print(f"Error loading PLY file: {e}")
        return None, None

def preprocess_points(points, downsample_factor=1):
    """Preprocess points - optional downsampling and centering."""
    if downsample_factor > 1:
        # Simple downsampling by taking every nth point
        points = points[::downsample_factor]
        print(f"Downsampled to {len(points)} points")
    
    # Center the point cloud
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    
    print(f"Centered point cloud at origin")
    print(f"  New X range: [{points_centered[:, 0].min():.3f}, {points_centered[:, 0].max():.3f}]")
    print(f"  New Y range: [{points_centered[:, 1].min():.3f}, {points_centered[:, 1].max():.3f}]")
    print(f"  New Z range: [{points_centered[:, 2].min():.3f}, {points_centered[:, 2].max():.3f}]")
    
    return points_centered

def analyze_ply_with_patches(points, cfg, n_patches_x=15, n_patches_y=15):
    """Analyze PLY file using fixed patches approach - now with 15x15 grid."""
    print(f"\n{'='*60}")
    print(f"FIXED PATCHES ANALYSIS ({n_patches_x}x{n_patches_y})")
    print(f"{'='*60}")
    
    # Divide into patches
    patches = divide_into_equal_patches(points, n_patches_x, n_patches_y)
    
    # Filter patches with minimum points
    min_points = 20
    valid_patches = [p for p in patches if len(p) >= min_points]
    print(f"Valid patches: {len(valid_patches)} out of {len(patches)}")
    
    if len(valid_patches) == 0:
        print("No valid patches found!")
        return None, None
    
    # Calculate differences
    differences = calculate_patch_differences_with_ds_8neighbors(
        valid_patches, cfg, n_patches_x, n_patches_y
    )
    
    # Statistics
    print(f"Difference statistics:")
    print(f"  Max magnitude: {max(differences):.6f}")
    print(f"  Mean magnitude: {sum(differences)/len(differences):.6f}")
    print(f"  Min magnitude: {min(differences):.6f}")
    print(f"  Std deviation: {np.std(differences):.6f}")
    
    return valid_patches, differences

def analyze_ply_with_sliding_window(points, cfg, n_patches_x=15, n_patches_y=15, overlap=0.0):
    """Analyze PLY file using sliding window approach - now with 0% overlap for fair comparison."""
    print(f"\n{'='*60}")
    print(f"SLIDING WINDOW ANALYSIS ({overlap*100}% overlap)")
    print(f"{'='*60}")
    
    stride = 1.0 - overlap
    
    # Divide into sliding windows
    sliding_patches, patch_info = divide_into_grid_patches_with_stride(
        points, n_patches_x=n_patches_x, n_patches_y=n_patches_y,
        stride_x=stride, stride_y=stride
    )
    
    # Filter patches with minimum points
    min_points = 20
    valid_patches = [p for p in sliding_patches if len(p) >= min_points]
    print(f"Valid windows: {len(valid_patches)} out of {len(sliding_patches)}")
    
    if len(valid_patches) == 0:
        print("No valid windows found!")
        return None, None, None
    
    # Calculate differences
    n_x, n_y = patch_info['grid_size']
    differences = calculate_patch_differences_with_ds_8neighbors(
        valid_patches, cfg, n_x, n_y
    )
    
    # Statistics
    print(f"Difference statistics:")
    print(f"  Max magnitude: {max(differences):.6f}")
    print(f"  Mean magnitude: {sum(differences)/len(differences):.6f}")
    print(f"  Min magnitude: {min(differences):.6f}")
    print(f"  Std deviation: {np.std(differences):.6f}")
    
    return valid_patches, differences, patch_info

def analyze_ply_with_sliding_window_overlap(points, cfg, n_patches_x=15, n_patches_y=15, overlap=0.5):
    """Analyze PLY file using sliding window approach with overlap - FORCED to exact 15x15 grid."""
    print(f"\n{'='*60}")
    print(f"SLIDING WINDOW WITH OVERLAP ANALYSIS (EXACT {n_patches_x}x{n_patches_y} GRID)")
    print(f"{'='*60}")
    
    stride = 1.0 - overlap
    
    # Divide into sliding windows - FORCE EXACT GRID SIZE
    sliding_patches, patch_info = divide_into_grid_patches_with_stride(
        points, n_patches_x=n_patches_x, n_patches_y=n_patches_y,
        stride_x=stride, stride_y=stride, force_exact_grid=True
    )
    
    # Filter patches with minimum points
    min_points = 20
    valid_patches = [p for p in sliding_patches if len(p) >= min_points]
    print(f"Valid windows: {len(valid_patches)} out of {len(sliding_patches)}")
    
    if len(valid_patches) == 0:
        print("No valid windows found!")
        return None, None, None
    
    # Calculate differences
    n_x, n_y = patch_info['grid_size']
    differences = calculate_patch_differences_with_ds_8neighbors(
        valid_patches, cfg, n_x, n_y
    )
    
    # Statistics
    print(f"Difference statistics:")
    print(f"  Max magnitude: {max(differences):.6f}")
    print(f"  Mean magnitude: {sum(differences)/len(differences):.6f}")
    print(f"  Min magnitude: {min(differences):.6f}")
    print(f"  Std deviation: {np.std(differences):.6f}")
    
    return valid_patches, differences, patch_info

def create_three_way_comparison_visualization(points, fixed_patches, fixed_differences, 
                                            sliding_patches, sliding_differences, patch_info,
                                            sliding_overlap_patches, sliding_overlap_differences, patch_info_overlap,
                                            n_patches_x, n_patches_y):
    """Create three-way comparison visualization."""
    
    fig = plt.figure(figsize=(20, 15))
    
    # Original point cloud
    ax1 = fig.add_subplot(2, 4, 1, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
               c=points[:, 2], cmap='viridis', s=0.1, alpha=0.6)
    ax1.set_title('Original Point Cloud')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Fixed patches visualization
    ax2 = fig.add_subplot(2, 4, 2)
    if fixed_differences:
        fixed_heatmap = np.array(fixed_differences).reshape(n_patches_y, n_patches_x)
        im2 = ax2.imshow(fixed_heatmap, cmap='PuBu', aspect='equal')
        ax2.set_title(f'Fixed Patches ({n_patches_x}×{n_patches_y})')
        plt.colorbar(im2, ax=ax2, label='DS Magnitude')
    
    # Sliding window (no overlap) visualization
    ax3 = fig.add_subplot(2, 4, 3)
    if sliding_differences and patch_info:
        n_x, n_y = patch_info['grid_size']
        sliding_heatmap = np.array(sliding_differences).reshape(n_y, n_x)
        im3 = ax3.imshow(sliding_heatmap, cmap='PuBu', aspect='equal')
        ax3.set_title(f'Sliding Window No Overlap ({n_x}×{n_y})')
        plt.colorbar(im3, ax=ax3, label='DS Magnitude')
    
    # Sliding window (with overlap) visualization
    ax4 = fig.add_subplot(2, 4, 4)
    if sliding_overlap_differences and patch_info_overlap:
        n_x_overlap, n_y_overlap = patch_info_overlap['grid_size']
        sliding_overlap_heatmap = np.array(sliding_overlap_differences).reshape(n_y_overlap, n_x_overlap)
        im4 = ax4.imshow(sliding_overlap_heatmap, cmap='PuBu', aspect='equal')
        ax4.set_title(f'Sliding Window Exact 15×15 ({n_x_overlap}×{n_y_overlap})')
        plt.colorbar(im4, ax=ax4, label='DS Magnitude')
    
    # Statistics comparison
    ax5 = fig.add_subplot(2, 4, 5)
    if fixed_differences and sliding_differences and sliding_overlap_differences:
        methods = ['Fixed\nPatches', 'Sliding\nNo Overlap', 'Sliding\nExact 15×15']
        max_vals = [max(fixed_differences), max(sliding_differences), max(sliding_overlap_differences)]
        mean_vals = [np.mean(fixed_differences), np.mean(sliding_differences), np.mean(sliding_overlap_differences)]
        
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax5.bar(x - width/2, max_vals, width, label='Max Magnitude', alpha=0.8)
        bars2 = ax5.bar(x + width/2, mean_vals, width, label='Mean Magnitude', alpha=0.8)
        
        ax5.set_ylabel('DS Magnitude')
        ax5.set_title('Comparison of Methods')
        ax5.set_xticks(x)
        ax5.set_xticklabels(methods)
        ax5.legend()
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        for bar in bars2:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
    
    # Histogram comparison
    ax6 = fig.add_subplot(2, 4, 6)
    if fixed_differences and sliding_differences and sliding_overlap_differences:
        ax6.hist(fixed_differences, bins=20, alpha=0.7, label='Fixed Patches', density=True)
        ax6.hist(sliding_differences, bins=20, alpha=0.7, label='Sliding No Overlap', density=True)
        ax6.hist(sliding_overlap_differences, bins=20, alpha=0.7, label='Sliding 50% Overlap', density=True)
        ax6.set_xlabel('DS Magnitude')
        ax6.set_ylabel('Density')
        ax6.set_title('Distribution of DS Magnitudes')
        ax6.legend()
    
    # Resolution comparison
    ax7 = fig.add_subplot(2, 4, 7)
    if fixed_differences and sliding_differences and sliding_overlap_differences:
        methods = ['Fixed', 'Sliding\nNo Overlap', 'Sliding\n50% Overlap']
        resolutions = [len(fixed_differences), len(sliding_differences), len(sliding_overlap_differences)]
        
        bars = ax7.bar(methods, resolutions, color=['blue', 'orange', 'green'], alpha=0.7)
        ax7.set_ylabel('Number of Analysis Points')
        ax7.set_title('Spatial Resolution Comparison')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
    
    # Summary text
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.axis('off')
    
    summary_text = "FAIR COMPARISON ANALYSIS\n\n"
    if fixed_differences and sliding_differences and sliding_overlap_differences:
        summary_text += f"Fixed Patches (15×15 = {len(fixed_patches)} patches):\n"
        summary_text += f"  Max: {max(fixed_differences):.6f}\n"
        summary_text += f"  Mean: {np.mean(fixed_differences):.6f}\n\n"
        
        summary_text += f"Sliding Window No Overlap (15×15 = {len(sliding_patches)} windows):\n"
        summary_text += f"  Max: {max(sliding_differences):.6f}\n"
        summary_text += f"  Mean: {np.mean(sliding_differences):.6f}\n\n"
        
        n_x_overlap, n_y_overlap = patch_info_overlap['grid_size']
        summary_text += f"Sliding Window Exact 15×15 with Overlap ({n_x_overlap}×{n_y_overlap} = {len(sliding_overlap_patches)} windows):\n"
        summary_text += f"  Max: {max(sliding_overlap_differences):.6f}\n"
        summary_text += f"  Mean: {np.mean(sliding_overlap_differences):.6f}\n\n"
        
        fair_improvement = (max(sliding_differences) - max(fixed_differences)) / max(fixed_differences) * 100
        overlap_improvement = (max(sliding_overlap_differences) - max(fixed_differences)) / max(fixed_differences) * 100
        
        summary_text += f"Fair Comparison (same resolution):\n"
        summary_text += f"  Performance change: {fair_improvement:+.1f}%\n\n"
        summary_text += f"Sliding Window Advantage:\n"
        summary_text += f"  With overlap: {overlap_improvement:+.1f}%\n"
        summary_text += f"  Resolution gain: {len(sliding_overlap_differences)/len(fixed_differences):.1f}x\n"
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.suptitle('Fair Comparison: Fixed Patches vs Sliding Window (Same Grid Size)', fontsize=16)
    plt.tight_layout()
    
    return fig

def main():
    """Main analysis function with fair comparison."""
    
    # Configuration
    cfg = Confing()
    cfg.subspace_dim = 3
    
    # Analysis parameters - NOW USING 15x15 for fair comparison
    n_patches_x = 15
    n_patches_y = 15
    overlap = 0.5
    
    # Create output directory
    os.makedirs('output/ply_fair_comparison', exist_ok=True)
    
    # Load PLY file
    print("Loading PLY file...")
    ply_filename = "newpc.ply"
    points, original_pcd = load_ply_file(ply_filename)
    
    if points is None:
        print("Failed to load PLY file!")
        return
    
    # Visualize original point cloud
    print("\nVisualizing original point cloud...")
    Pointlib.visualize([original_pcd], point_size=5)
    
    # Preprocess points
    print("\nPreprocessing points...")
    points_processed = preprocess_points(points, downsample_factor=1)
    
    # Analyze with fixed patches (15x15 for fair comparison)
    fixed_patches, fixed_differences = analyze_ply_with_patches(
        points_processed, cfg, n_patches_x, n_patches_y
    )
    
    if fixed_differences:
        # Visualize fixed patches results
        visualize_patch_differences(fixed_patches, fixed_differences, 
                                   n_patches_x, n_patches_y, "ply_fixed_15x15")
        plt.savefig('output/ply_fair_comparison/fixed_patches_15x15.png', dpi=150, bbox_inches='tight')
    
    # Analyze with sliding window (no overlap for fair comparison)
    sliding_patches, sliding_differences, patch_info = analyze_ply_with_sliding_window(
        points_processed, cfg, n_patches_x, n_patches_y, overlap=0.0
    )
    
    if sliding_differences and patch_info:
        # Visualize sliding window results (no overlap)
        n_x, n_y = patch_info['grid_size']
        visualize_patch_differences(sliding_patches, sliding_differences, 
                                   n_x, n_y, "ply_sliding_no_overlap")
        plt.savefig('output/ply_fair_comparison/sliding_window_no_overlap.png', dpi=150, bbox_inches='tight')
    
    # Analyze with sliding window (50% overlap to show advantage)
    sliding_overlap_patches, sliding_overlap_differences, patch_info_overlap = analyze_ply_with_sliding_window_overlap(
        points_processed, cfg, n_patches_x, n_patches_y, overlap=0.5
    )
    
    if sliding_overlap_differences and patch_info_overlap:
        # Visualize sliding window results (with overlap)
        n_x_overlap, n_y_overlap = patch_info_overlap['grid_size']
        visualize_patch_differences(sliding_overlap_patches, sliding_overlap_differences, 
                                   n_x_overlap, n_y_overlap, "ply_sliding_50pct_overlap")
        plt.savefig('output/ply_fair_comparison/sliding_window_50pct_overlap.png', dpi=150, bbox_inches='tight')
    
    # Create three-way comparison visualization
    if fixed_differences and sliding_differences and sliding_overlap_differences:
        print("\nCreating three-way comparison visualization...")
        comparison_fig = create_three_way_comparison_visualization(
            points_processed, fixed_patches, fixed_differences,
            sliding_patches, sliding_differences, patch_info,
            sliding_overlap_patches, sliding_overlap_differences, patch_info_overlap,
            n_patches_x, n_patches_y
        )
        comparison_fig.savefig('output/ply_fair_comparison/three_way_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print final comparison
        print(f"\n{'='*60}")
        print("FAIR COMPARISON RESULTS")
        print(f"{'='*60}")
        print(f"Point cloud: {ply_filename} ({len(points)} points)")
        print(f"Analysis grid: {n_patches_x}×{n_patches_y}")
        print()
        print(f"Fixed Patches (15×15):")
        print(f"  Valid patches: {len(fixed_patches)}")
        print(f"  Max DS magnitude: {max(fixed_differences):.6f}")
        print(f"  Mean DS magnitude: {np.mean(fixed_differences):.6f}")
        print()
        print(f"Sliding Window No Overlap (15×15):")
        print(f"  Valid windows: {len(sliding_patches)}")
        print(f"  Max DS magnitude: {max(sliding_differences):.6f}")
        print(f"  Mean DS magnitude: {np.mean(sliding_differences):.6f}")
        print()
        n_x_overlap, n_y_overlap = patch_info_overlap['grid_size']
        print(f"Sliding Window 50% Overlap ({n_x_overlap}×{n_y_overlap}):")
        print(f"  Valid windows: {len(sliding_overlap_patches)}")
        print(f"  Max DS magnitude: {max(sliding_overlap_differences):.6f}")
        print(f"  Mean DS magnitude: {np.mean(sliding_overlap_differences):.6f}")
        print()
        
        # Fair comparison (same resolution)
        fair_improvement = (max(sliding_differences) - max(fixed_differences)) / max(fixed_differences) * 100
        print(f"Fair comparison (same 15×15 grid):")
        print(f"  Peak detection change: {fair_improvement:+.1f}%")
        
        # Overlap advantage
        overlap_improvement = (max(sliding_overlap_differences) - max(fixed_differences)) / max(fixed_differences) * 100
        resolution_increase = len(sliding_overlap_differences) / len(fixed_differences)
        print(f"\nSliding window with overlap advantage:")
        print(f"  Peak detection improvement: {overlap_improvement:+.1f}%")
        print(f"  Resolution increase: {resolution_increase:.1f}x")

if __name__ == "__main__":
    main()