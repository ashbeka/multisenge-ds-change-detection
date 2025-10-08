import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def divide_into_grid_patches_with_stride(points, n_patches_x=10, n_patches_y=10, 
                                        stride_x=1.0, stride_y=1.0, force_exact_grid=False):
    """
    Divide point cloud into grid-based patches with stride support for overlapping patches.
    
    Args:
        points: numpy array of shape (N, 3) containing point cloud data
        n_patches_x: number of patches in x direction
        n_patches_y: number of patches in y direction
        stride_x: stride factor for x direction (0 < stride <= 1.0)
                  1.0 = no overlap, 0.5 = 50% overlap
        stride_y: stride factor for y direction (0 < stride <= 1.0)
                  1.0 = no overlap, 0.5 = 50% overlap
        force_exact_grid: if True, force output to be exactly n_patches_x × n_patches_y
                         regardless of overlap (for fair comparisons)
    
    Returns:
        patches: list of numpy arrays, each containing points in a patch
        patch_info: dictionary containing patch metadata
    """
    # Get bounding box of the point cloud
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    
    if force_exact_grid:
        # Force exactly n_patches_x × n_patches_y output regardless of overlap
        n_patches_x_stride = n_patches_x
        n_patches_y_stride = n_patches_y
        
        # Calculate patch dimensions and stride to fit exactly
        patch_width = (x_max - x_min) / n_patches_x
        patch_height = (y_max - y_min) / n_patches_y
        
        # Calculate actual stride needed for exact grid with overlap
        if n_patches_x > 1:
            stride_width = (x_max - x_min - patch_width) / (n_patches_x - 1)
        else:
            stride_width = patch_width
            
        if n_patches_y > 1:
            stride_height = (y_max - y_min - patch_height) / (n_patches_y - 1)
        else:
            stride_height = patch_height
            
        # Calculate actual overlap achieved
        actual_stride_x = stride_width / patch_width if patch_width > 0 else 1.0
        actual_stride_y = stride_height / patch_height if patch_height > 0 else 1.0
    else:
        # Original behavior - calculate grid size based on stride
        # Calculate patch dimensions
        patch_width = (x_max - x_min) / n_patches_x
        patch_height = (y_max - y_min) / n_patches_y
        
        # Calculate effective stride (distance between patch centers)
        stride_width = patch_width * stride_x
        stride_height = patch_height * stride_y
        
        # Calculate number of patches with stride
        # When stride < 1, we get more patches due to overlap
        n_patches_x_stride = int((x_max - x_min - patch_width) / stride_width) + 1
        n_patches_y_stride = int((y_max - y_min - patch_height) / stride_height) + 1
        
        actual_stride_x = stride_x
        actual_stride_y = stride_y
    
    patches = []
    patch_info = {
        'bounds': [],
        'centers': [],
        'indices': [],
        'grid_size': (n_patches_x_stride, n_patches_y_stride),
        'patch_size': (patch_width, patch_height),
        'stride': (stride_width, stride_height),
        'overlap_x': 1.0 - actual_stride_x,
        'overlap_y': 1.0 - actual_stride_y
    }
    
    # Create patches with stride
    for i in range(n_patches_y_stride):
        for j in range(n_patches_x_stride):
            # Calculate patch bounds
            patch_x_min = x_min + j * stride_width
            patch_x_max = patch_x_min + patch_width
            patch_y_min = y_min + i * stride_height
            patch_y_max = patch_y_min + patch_height
            
            # Ensure we don't exceed the original bounds
            patch_x_max = min(patch_x_max, x_max)
            patch_y_max = min(patch_y_max, y_max)
            
            # Extract points within this patch
            mask = ((points[:, 0] >= patch_x_min) & 
                   (points[:, 0] < patch_x_max) &
                   (points[:, 1] >= patch_y_min) & 
                   (points[:, 1] < patch_y_max))
            
            patch_points = points[mask]
            patches.append(patch_points)
            
            # Store patch metadata
            patch_info['bounds'].append({
                'x_min': patch_x_min, 'x_max': patch_x_max,
                'y_min': patch_y_min, 'y_max': patch_y_max
            })
            patch_info['centers'].append({
                'x': (patch_x_min + patch_x_max) / 2,
                'y': (patch_y_min + patch_y_max) / 2
            })
            patch_info['indices'].append((i, j))
    
    print(f"\nGrid-based patch division with stride:")
    print(f"  Target grid: {n_patches_x} x {n_patches_y}")
    if force_exact_grid:
        print(f"  Force exact grid: True")
        print(f"  Requested overlap: x={100*(1-stride_x):.0f}%, y={100*(1-stride_y):.0f}%")
        print(f"  Actual overlap: x={patch_info['overlap_x']*100:.1f}%, y={patch_info['overlap_y']*100:.1f}%")
    else:
        print(f"  Stride factors: x={stride_x:.2f}, y={stride_y:.2f}")
        print(f"  Overlap: x={patch_info['overlap_x']*100:.0f}%, y={patch_info['overlap_y']*100:.0f}%")
    print(f"  Resulting grid: {n_patches_x_stride} x {n_patches_y_stride}")
    print(f"  Total patches: {len(patches)}")
    print(f"  Patch size: {patch_width:.3f} x {patch_height:.3f}")
    print(f"  Stride: {stride_width:.3f} x {stride_height:.3f}")
    
    return patches, patch_info


def visualize_grid_patches_with_overlap(points, patches, patch_info, show_overlap=True):
    """
    Visualize the grid-based patches with overlap visualization.
    
    Args:
        points: original point cloud
        patches: list of patches
        patch_info: patch metadata from divide_into_grid_patches_with_stride
        show_overlap: whether to highlight overlapping regions
    """
    fig = plt.figure(figsize=(20, 8))
    
    # 1. 3D visualization of patches
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Use different colors for patches
    colors = plt.cm.rainbow(np.linspace(0, 1, len(patches)))
    
    for idx, (patch, color) in enumerate(zip(patches, colors)):
        if len(patch) > 0:
            ax1.scatter(patch[:, 0], patch[:, 1], patch[:, 2],
                       c=[color], alpha=0.3, s=1)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D View of Grid Patches')
    
    # 2. Top-down view showing patch boundaries
    ax2 = fig.add_subplot(132)
    
    # Plot all points in gray
    ax2.scatter(points[:, 0], points[:, 1], c='lightgray', s=1, alpha=0.5)
    
    # Draw patch boundaries
    for bound in patch_info['bounds']:
        rect = plt.Rectangle((bound['x_min'], bound['y_min']),
                           bound['x_max'] - bound['x_min'],
                           bound['y_max'] - bound['y_min'],
                           fill=False, edgecolor='red', linewidth=1)
        ax2.add_patch(rect)
    
    # Mark patch centers
    centers_x = [c['x'] for c in patch_info['centers']]
    centers_y = [c['y'] for c in patch_info['centers']]
    ax2.scatter(centers_x, centers_y, c='blue', marker='+', s=50)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Top-down View with Patch Boundaries')
    ax2.set_aspect('equal')
    
    # 3. Overlap visualization
    ax3 = fig.add_subplot(133)
    
    if show_overlap and (patch_info['overlap_x'] > 0 or patch_info['overlap_y'] > 0):
        # Create a density map showing overlap
        x_min = min(b['x_min'] for b in patch_info['bounds'])
        x_max = max(b['x_max'] for b in patch_info['bounds'])
        y_min = min(b['y_min'] for b in patch_info['bounds'])
        y_max = max(b['y_max'] for b in patch_info['bounds'])
        
        # Create a grid for counting overlaps
        resolution = 100
        x_grid = np.linspace(x_min, x_max, resolution)
        y_grid = np.linspace(y_min, y_max, resolution)
        overlap_count = np.zeros((resolution, resolution))
        
        # Count how many patches each grid cell belongs to
        for bound in patch_info['bounds']:
            x_mask = (x_grid >= bound['x_min']) & (x_grid <= bound['x_max'])
            y_mask = (y_grid >= bound['y_min']) & (y_grid <= bound['y_max'])
            overlap_count[np.ix_(y_mask, x_mask)] += 1
        
        im = ax3.imshow(overlap_count, extent=[x_min, x_max, y_min, y_max],
                       origin='lower', cmap='YlOrRd', aspect='equal')
        plt.colorbar(im, ax=ax3, label='Number of overlapping patches')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_title('Patch Overlap Density')
    else:
        ax3.text(0.5, 0.5, 'No overlap\n(stride = 1.0)', 
                transform=ax3.transAxes, ha='center', va='center', fontsize=16)
        ax3.set_xticks([])
        ax3.set_yticks([])
    
    plt.tight_layout()
    plt.show()


def calculate_adaptive_stride(points, n_patches_x, n_patches_y, min_points_per_patch=50):
    """
    Calculate adaptive stride based on point density to ensure minimum points per patch.
    
    Args:
        points: point cloud data
        n_patches_x, n_patches_y: desired grid size
        min_points_per_patch: minimum number of points required per patch
    
    Returns:
        stride_x, stride_y: recommended stride factors
    """
    # First, divide with no overlap to check point distribution
    patches_no_overlap, _ = divide_into_grid_patches_with_stride(
        points, n_patches_x, n_patches_y, stride_x=1.0, stride_y=1.0
    )
    
    # Find patches with too few points
    sparse_patches = [i for i, p in enumerate(patches_no_overlap) if len(p) < min_points_per_patch]
    
    if len(sparse_patches) == 0:
        print("All patches have sufficient points. No overlap needed.")
        return 1.0, 1.0
    
    # Calculate recommended overlap based on sparsity
    sparsity_ratio = len(sparse_patches) / len(patches_no_overlap)
    
    # More sparse regions need more overlap
    if sparsity_ratio > 0.5:
        stride_x = stride_y = 0.5  # 50% overlap
    elif sparsity_ratio > 0.3:
        stride_x = stride_y = 0.7  # 30% overlap
    else:
        stride_x = stride_y = 0.85  # 15% overlap
    
    print(f"Adaptive stride calculation:")
    print(f"  Sparse patches: {len(sparse_patches)}/{len(patches_no_overlap)} ({sparsity_ratio*100:.1f}%)")
    print(f"  Recommended stride: x={stride_x}, y={stride_y} ({(1-stride_x)*100:.0f}% overlap)")
    
    return stride_x, stride_y


# Example usage and testing
if __name__ == "__main__":
    # Generate sample point cloud data (you can replace this with real data)
    np.random.seed(42)
    
    # Create a surface with some bumps
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    
    # Base surface with two bumps
    Z = 0.1 * np.sin(X) + 0.1 * np.cos(Y)
    Z += 0.5 * np.exp(-((X-3)**2 + (Y-3)**2) / 0.5)  # Bump 1
    Z += 0.7 * np.exp(-((X-7)**2 + (Y-7)**2) / 0.3)  # Bump 2
    
    # Convert to point cloud
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # Add some noise
    points += np.random.normal(0, 0.02, points.shape)
    
    print("Testing grid-based patch division with different stride values...")
    print("="*60)
    
    # Test 1: No overlap (stride = 1.0)
    print("\nTest 1: No overlap")
    patches1, info1 = divide_into_grid_patches_with_stride(
        points, n_patches_x=10, n_patches_y=10, stride_x=1.0, stride_y=1.0
    )
    visualize_grid_patches_with_overlap(points, patches1, info1)
    
    # Test 2: 50% overlap (stride = 0.5)
    print("\nTest 2: 50% overlap")
    patches2, info2 = divide_into_grid_patches_with_stride(
        points, n_patches_x=10, n_patches_y=10, stride_x=0.5, stride_y=0.5
    )
    visualize_grid_patches_with_overlap(points, patches2, info2)
    
    # Test 3: Different overlap in x and y
    print("\nTest 3: Different overlap in x (30%) and y (70%)")
    patches3, info3 = divide_into_grid_patches_with_stride(
        points, n_patches_x=10, n_patches_y=10, stride_x=0.7, stride_y=0.3
    )
    visualize_grid_patches_with_overlap(points, patches3, info3)
    
    # Test 4: Adaptive stride
    print("\nTest 4: Adaptive stride calculation")
    stride_x_adaptive, stride_y_adaptive = calculate_adaptive_stride(
        points, n_patches_x=15, n_patches_y=15, min_points_per_patch=30
    )
    patches4, info4 = divide_into_grid_patches_with_stride(
        points, n_patches_x=15, n_patches_y=15, 
        stride_x=stride_x_adaptive, stride_y=stride_y_adaptive
    )
    visualize_grid_patches_with_overlap(points, patches4, info4)
    
    # Print statistics
    print("\n" + "="*60)
    print("Summary of patch statistics:")
    print(f"No overlap: {len(patches1)} patches")
    print(f"50% overlap: {len(patches2)} patches")
    print(f"Asymmetric overlap: {len(patches3)} patches")
    print(f"Adaptive overlap: {len(patches4)} patches")