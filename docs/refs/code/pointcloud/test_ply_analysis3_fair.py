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
    """Analyze PLY file using fixed patches approach."""
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

def analyze_ply_with_sliding_window_fair(points, cfg, n_patches_x=15, n_patches_y=15, overlap=0.5):
    """Analyze PLY file using sliding window approach - FORCED to exact 15x15 grid for fair comparison."""
    print(f"\n{'='*60}")
    print(f"SLIDING WINDOW FAIR COMPARISON (EXACT {n_patches_x}x{n_patches_y} GRID)")
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
    print(f"  Actual overlap achieved: {patch_info['overlap_x']*100:.1f}%")
    
    return valid_patches, differences, patch_info

def create_fair_comparison_visualization(points, fixed_patches, fixed_differences, 
                                       sliding_patches, sliding_differences, patch_info,
                                       n_patches_x, n_patches_y):
    """Create fair comparison visualization with exactly the same grid sizes."""
    
    fig = plt.figure(figsize=(16, 12))
    
    # Original point cloud
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
               c=points[:, 2], cmap='viridis', s=0.1, alpha=0.6)
    ax1.set_title('Original Point Cloud')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Fixed patches visualization
    ax2 = fig.add_subplot(2, 3, 2)
    if fixed_differences:
        fixed_heatmap = np.array(fixed_differences).reshape(n_patches_y, n_patches_x)
        im2 = ax2.imshow(fixed_heatmap, cmap='PuBu', aspect='equal')
        ax2.set_title(f'Fixed Patches ({n_patches_x}×{n_patches_y})')
        plt.colorbar(im2, ax=ax2, label='DS Magnitude')
    
    # Sliding window visualization
    ax3 = fig.add_subplot(2, 3, 3)
    if sliding_differences and patch_info:
        n_x, n_y = patch_info['grid_size']
        sliding_heatmap = np.array(sliding_differences).reshape(n_y, n_x)
        im3 = ax3.imshow(sliding_heatmap, cmap='PuBu', aspect='equal')
        overlap_pct = patch_info['overlap_x'] * 100
        ax3.set_title(f'Sliding Window ({n_x}×{n_y}, {overlap_pct:.1f}% overlap)')
        plt.colorbar(im3, ax=ax3, label='DS Magnitude')
    
    # Statistics comparison
    ax4 = fig.add_subplot(2, 3, 4)
    if fixed_differences and sliding_differences:
        methods = ['Fixed Patches', 'Sliding Window']
        max_vals = [max(fixed_differences), max(sliding_differences)]
        mean_vals = [np.mean(fixed_differences), np.mean(sliding_differences)]
        
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, max_vals, width, label='Max Magnitude', alpha=0.8)
        bars2 = ax4.bar(x + width/2, mean_vals, width, label='Mean Magnitude', alpha=0.8)
        
        ax4.set_ylabel('DS Magnitude')
        ax4.set_title('Fair Comparison (Same Grid Size)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(methods)
        ax4.legend()
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        for bar in bars2:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
    
    # Histogram comparison
    ax5 = fig.add_subplot(2, 3, 5)
    if fixed_differences and sliding_differences:
        ax5.hist(fixed_differences, bins=20, alpha=0.7, label='Fixed Patches', density=True)
        ax5.hist(sliding_differences, bins=20, alpha=0.7, label='Sliding Window', density=True)
        ax5.set_xlabel('DS Magnitude')
        ax5.set_ylabel('Density')
        ax5.set_title('Distribution of DS Magnitudes')
        ax5.legend()
    
    # Summary text
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = "TRULY FAIR COMPARISON\\n\\n"
    if fixed_differences and sliding_differences:
        summary_text += f"Fixed Patches ({n_patches_x}×{n_patches_y} = {len(fixed_patches)} patches):\\n"
        summary_text += f"  Max: {max(fixed_differences):.6f}\\n"
        summary_text += f"  Mean: {np.mean(fixed_differences):.6f}\\n\\n"
        
        n_x, n_y = patch_info['grid_size']
        overlap_pct = patch_info['overlap_x'] * 100
        summary_text += f"Sliding Window ({n_x}×{n_y} = {len(sliding_patches)} windows):\\n"
        summary_text += f"  Max: {max(sliding_differences):.6f}\\n"
        summary_text += f"  Mean: {np.mean(sliding_differences):.6f}\\n"
        summary_text += f"  Overlap: {overlap_pct:.1f}%\\n\\n"
        
        improvement = (max(sliding_differences) - max(fixed_differences)) / max(fixed_differences) * 100
        mean_change = (np.mean(sliding_differences) - np.mean(fixed_differences)) / np.mean(fixed_differences) * 100
        
        summary_text += f"Fair Comparison Results:\\n"
        summary_text += f"  Peak detection change: {improvement:+.1f}%\\n"
        summary_text += f"  Mean magnitude change: {mean_change:+.1f}%\\n\\n"
        
        if improvement > 0:
            summary_text += f"✓ Sliding window shows {improvement:.1f}% improvement\\n"
            summary_text += f"  with {overlap_pct:.1f}% overlap at same resolution"
        else:
            summary_text += f"✗ Fixed patches perform {abs(improvement):.1f}% better\\n"
            summary_text += f"  for this dataset"
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.suptitle('Fair Comparison: Fixed Patches vs Sliding Window (Exact Same Grid Size)', fontsize=14)
    plt.tight_layout()
    
    return fig

def main():
    """Main analysis function with truly fair comparison."""
    
    # Configuration
    cfg = Confing()
    cfg.subspace_dim = 3
    
    # Analysis parameters - EXACT SAME GRID SIZE
    n_patches_x = 15
    n_patches_y = 15
    overlap = 0.5  # Request 50% overlap, but force exact 15x15 output
    
    # Create output directory
    os.makedirs('output/ply_truly_fair_comparison', exist_ok=True)
    
    # Load PLY file
    print("Loading PLY file...")
    ply_filename = "newpc.ply"
    points, original_pcd = load_ply_file(ply_filename)
    
    if points is None:
        print("Failed to load PLY file!")
        return
    
    # Visualize original point cloud
    print("\\nVisualizing original point cloud...")
    Pointlib.visualize([original_pcd], point_size=5)
    
    # Preprocess points
    print("\\nPreprocessing points...")
    points_processed = preprocess_points(points, downsample_factor=1)
    
    # Analyze with fixed patches (15x15)
    fixed_patches, fixed_differences = analyze_ply_with_patches(
        points_processed, cfg, n_patches_x, n_patches_y
    )
    
    if fixed_differences:
        # Visualize fixed patches results
        visualize_patch_differences(fixed_patches, fixed_differences, 
                                   n_patches_x, n_patches_y, "ply_fixed_fair_15x15")
        plt.savefig('output/ply_truly_fair_comparison/fixed_patches_15x15.png', dpi=150, bbox_inches='tight')
    
    # Analyze with sliding window (FORCED to exact 15x15 with overlap)
    sliding_patches, sliding_differences, patch_info = analyze_ply_with_sliding_window_fair(
        points_processed, cfg, n_patches_x, n_patches_y, overlap=overlap
    )
    
    if sliding_differences and patch_info:
        # Visualize sliding window results
        n_x, n_y = patch_info['grid_size']
        overlap_pct = patch_info['overlap_x'] * 100
        visualize_patch_differences(sliding_patches, sliding_differences, 
                                   n_x, n_y, f"ply_sliding_fair_{overlap_pct:.0f}pct_overlap")
        plt.savefig(f'output/ply_truly_fair_comparison/sliding_window_fair_{overlap_pct:.0f}pct_overlap.png', dpi=150, bbox_inches='tight')
    
    # Create fair comparison visualization
    if fixed_differences and sliding_differences:
        print("\\nCreating truly fair comparison visualization...")
        comparison_fig = create_fair_comparison_visualization(
            points_processed, fixed_patches, fixed_differences,
            sliding_patches, sliding_differences, patch_info,
            n_patches_x, n_patches_y
        )
        comparison_fig.savefig('output/ply_truly_fair_comparison/truly_fair_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print final comparison
        print(f"\\n{'='*60}")
        print("TRULY FAIR COMPARISON RESULTS")
        print(f"{'='*60}")
        print(f"Point cloud: {ply_filename} ({len(points)} points)")
        print(f"Both methods use EXACTLY the same grid: {n_patches_x}×{n_patches_y}")
        print()
        print(f"Fixed Patches:")
        print(f"  Valid patches: {len(fixed_patches)}")
        print(f"  Max DS magnitude: {max(fixed_differences):.6f}")
        print(f"  Mean DS magnitude: {np.mean(fixed_differences):.6f}")
        print()
        n_x, n_y = patch_info['grid_size']
        overlap_pct = patch_info['overlap_x'] * 100
        print(f"Sliding Window (forced to {n_x}×{n_y}):")
        print(f"  Valid windows: {len(sliding_patches)}")
        print(f"  Max DS magnitude: {max(sliding_differences):.6f}")
        print(f"  Mean DS magnitude: {np.mean(sliding_differences):.6f}")
        print(f"  Actual overlap achieved: {overlap_pct:.1f}%")
        print()
        
        # Fair comparison results
        improvement = (max(sliding_differences) - max(fixed_differences)) / max(fixed_differences) * 100
        mean_change = (np.mean(sliding_differences) - np.mean(fixed_differences)) / np.mean(fixed_differences) * 100
        print(f"Fair comparison results (same {n_patches_x}×{n_patches_y} grid):")
        print(f"  Peak detection change: {improvement:+.1f}%")
        print(f"  Mean magnitude change: {mean_change:+.1f}%")
        print()
        
        if improvement > 0:
            print(f"✓ Sliding window with {overlap_pct:.1f}% overlap shows {improvement:.1f}% improvement")
            print(f"  even at the same spatial resolution as fixed patches!")
        else:
            print(f"✗ Fixed patches perform {abs(improvement):.1f}% better for this dataset")
            print(f"  suggesting this data is clean enough that overlap doesn't help")

if __name__ == "__main__":
    main()