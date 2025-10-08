import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig, svd
from grid_patch_division_with_stride import divide_into_grid_patches_with_stride, visualize_grid_patches_with_overlap
from main import (generate_synthetic_floor, generate_subspace, gen_shape_difference_subspace, 
                  calculate_patch_differences_with_ds_8neighbors, visualize_patch_differences,
                  divide_into_patches)
from config import Confing
import os

def calculate_sliding_window_differences(patches, patch_info, cfg):
    """Calculate differences using sliding window approach."""
    average_magnitudes = []
    patch_subspaces = [generate_subspace(patch, cfg) for patch in patches]
    
    n_patches_x_stride, n_patches_y_stride = patch_info['grid_size']
    
    def get_neighbors_8(idx):
        row = idx // n_patches_x_stride
        col = idx % n_patches_x_stride
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
            if (0 <= new_row < n_patches_y_stride) and (0 <= new_col < n_patches_x_stride):
                neighbor_idx = new_row * n_patches_x_stride + new_col
                neighbors.append(neighbor_idx)
                weights.append(1.0)
                
        return neighbors, weights
    
    for i in range(len(patches)):
        neighbors, weights = get_neighbors_8(i)
        patch_magnitudes = []
        
        for neighbor_idx, weight in zip(neighbors, weights):
            DS = gen_shape_difference_subspace(patch_subspaces[i], patch_subspaces[neighbor_idx], cfg)
            if DS.shape[1] > 0:
                P = DS @ DS.T
                V = P @ patch_subspaces[i]
                magnitude = np.linalg.norm(V)
                patch_magnitudes.append(magnitude)
        
        if patch_magnitudes:
            avg_magnitude = sum(patch_magnitudes) / sum(weights)
            average_magnitudes.append(avg_magnitude)
        else:
            average_magnitudes.append(0)
    
    return average_magnitudes

def visualize_sliding_window_results(patches, magnitudes, patch_info, title_suffix=""):
    """Visualize results from sliding window analysis."""
    fig = plt.figure(figsize=(12, 5))
    
    n_patches_x, n_patches_y = patch_info['grid_size']
    
    # Convert magnitudes to numpy array and reshape
    heatmap = np.array(magnitudes).reshape(n_patches_y, n_patches_x)
    
    # Set consistent color scaling
    vmin = 0
    vmax = max(1e-6, np.max(heatmap))
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.PuBu
    
    # Plot heatmap
    ax = fig.add_subplot(111)
    im = ax.imshow(heatmap, cmap=cmap, norm=norm, aspect='equal')
    ax.set_title(f'Sliding Window DS Magnitude Heatmap{title_suffix}')
    plt.colorbar(im, ax=ax, label='Average DS Magnitude')
    
    # Add grid lines
    ax.set_xticks(np.arange(-.5, n_patches_x, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_patches_y, 1), minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    
    # Add text annotations for values
    for i in range(n_patches_y):
        for j in range(n_patches_x):
            color_val = heatmap[i, j]
            text_color = 'white' if color_val > (vmax + vmin) / 2 else 'black'
            ax.text(j, i, f'{heatmap[i, j]:.4f}',
                   ha='center', va='center',
                   color=text_color, fontsize=6)
    
    plt.tight_layout()
    return fig, heatmap

def compare_approaches():
    """Compare fixed patch vs sliding window approaches."""
    
    # Configuration
    cfg = Confing()
    cfg.subspace_dim = 3
    n_patches_x = 10
    n_patches_y = 10
    
    # Test cases
    test_cases = {
        "single_bump": [(25, 25, 1.0, 2.0)],
        "double_bump": [(15, 25, 0.8, 1.5), (35, 25, 0.8, 1.5)],
        "corner_bumps": [(10, 10, 0.7, 1.0), (10, 40, 0.7, 1.0), 
                         (40, 10, 0.7, 1.0), (40, 40, 0.7, 1.0)],
        "ridge": [(10, 25, 0.8, 1.0), (25, 25, 0.8, 1.0), (40, 25, 0.8, 1.0)]
    }
    
    for case_name, bumps in test_cases.items():
        print(f"\n{'='*60}")
        print(f"Testing: {case_name}")
        print(f"{'='*60}")
        
        # Generate synthetic floor
        floor_points = generate_synthetic_floor(
            width=50, length=50,
            n_points_x=100, n_points_y=100,
            bumps=bumps
        )
        
        # Create output directory if it doesn't exist
        os.makedirs('output/sliding_window_comparison', exist_ok=True)
        
        # 1. Fixed patches approach
        print("\n1. Fixed Patches Approach:")
        fixed_patches = divide_into_patches(floor_points, n_patches_x, n_patches_y)
        fixed_magnitudes = calculate_patch_differences_with_ds_8neighbors(
            fixed_patches, cfg, n_patches_x, n_patches_y
        )
        
        # Visualize fixed patches
        fig1 = visualize_patch_differences(fixed_patches, fixed_magnitudes, 
                                          n_patches_x, n_patches_y, f"{case_name}_fixed")
        plt.savefig(f'output/sliding_window_comparison/{case_name}_fixed_patches.png', dpi=150)
        plt.close()
        
        # 2. Sliding window approach with different overlaps
        overlaps = [0.0, 0.3, 0.5, 0.7]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        axes = axes.ravel()
        
        for idx, overlap in enumerate(overlaps):
            stride = 1.0 - overlap
            print(f"\n2.{idx+1} Sliding Window (Overlap: {overlap*100}%):")
            
            sliding_patches, patch_info = divide_into_grid_patches_with_stride(
                floor_points, n_patches_x=n_patches_x, n_patches_y=n_patches_y,
                stride_x=stride, stride_y=stride
            )
            
            sliding_magnitudes = calculate_sliding_window_differences(
                sliding_patches, patch_info, cfg
            )
            
            # Create heatmap
            n_x, n_y = patch_info['grid_size']
            heatmap = np.array(sliding_magnitudes).reshape(n_y, n_x)
            
            # Plot in subplot
            im = axes[idx].imshow(heatmap, cmap='PuBu', aspect='equal')
            axes[idx].set_title(f'Overlap: {overlap*100}% ({len(sliding_patches)} windows)')
            plt.colorbar(im, ax=axes[idx])
            
            # Print statistics
            print(f"  Grid size: {n_x} x {n_y}")
            print(f"  Total windows: {len(sliding_patches)}")
            print(f"  Max magnitude: {np.max(sliding_magnitudes):.6f}")
            print(f"  Mean magnitude: {np.mean(sliding_magnitudes):.6f}")
        
        plt.suptitle(f'{case_name.replace("_", " ").title()} - Sliding Window Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'output/sliding_window_comparison/{case_name}_sliding_windows.png', dpi=150)
        plt.close()
        
        # 3. Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Fixed patches
        fixed_heatmap = np.array(fixed_magnitudes).reshape(n_patches_y, n_patches_x)
        im1 = ax1.imshow(fixed_heatmap, cmap='PuBu', aspect='equal')
        ax1.set_title(f'Fixed Patches ({n_patches_x}x{n_patches_y})')
        plt.colorbar(im1, ax=ax1)
        
        # Best sliding window (50% overlap)
        stride = 0.5
        sliding_patches, patch_info = divide_into_grid_patches_with_stride(
            floor_points, n_patches_x=n_patches_x, n_patches_y=n_patches_y,
            stride_x=stride, stride_y=stride
        )
        sliding_magnitudes = calculate_sliding_window_differences(
            sliding_patches, patch_info, cfg
        )
        n_x, n_y = patch_info['grid_size']
        sliding_heatmap = np.array(sliding_magnitudes).reshape(n_y, n_x)
        
        im2 = ax2.imshow(sliding_heatmap, cmap='PuBu', aspect='equal')
        ax2.set_title(f'Sliding Window 50% overlap ({n_x}x{n_y})')
        plt.colorbar(im2, ax=ax2)
        
        plt.suptitle(f'{case_name.replace("_", " ").title()} - Fixed vs Sliding Window', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'output/sliding_window_comparison/{case_name}_comparison.png', dpi=150)
        plt.close()

if __name__ == "__main__":
    compare_approaches()