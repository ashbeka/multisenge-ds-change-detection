import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy.linalg import svd
from config import Confing
import open3d as o3d
import random
import pointlib as Pointlib
from mpl_toolkits.mplot3d import Axes3D
import scipy
import os


def generate_subspace(data_segment, cfg):
    """Generate a subspace from a segment using SVD."""
    X = data_segment
    n_points = len(X)
    
    # If too few points, return a zero subspace
    if n_points < cfg.subspace_dim:
        return np.zeros((n_points, cfg.subspace_dim))
    
    # Center the data
    mv = np.mean(X, axis=0)
    X_centered = X - mv
    
    # Compute SVD
    try:
        U, s, Vt = svd(X_centered, full_matrices=False)
        # Ensure we have the right dimensions
        if U.shape[1] >= cfg.subspace_dim:
            return U[:, 0:cfg.subspace_dim]
        else:
            # Pad with zeros if needed
            result = np.zeros((n_points, cfg.subspace_dim))
            result[:, :U.shape[1]] = U
            return result
    except:
        # If SVD fails, return zeros
        return np.zeros((n_points, cfg.subspace_dim))


def gen_shape_difference_subspace(S1, S2, cfg):
    """Generate the Difference Subspace (DS) between two subspaces."""
    # Back to original implementation that was working
    # S1 and S2 are orthonormal basis matrices (n_points x n_dims)
    # But we need to ensure they have same dimensions
    
    # Get the minimum number of points
    min_points = min(S1.shape[0], S2.shape[0])
    
    # Truncate to same size if needed
    S1_truncated = S1[:min_points, :]
    S2_truncated = S2[:min_points, :]
    
    # Original formula: G = S1 @ S1.T + S2 @ S2.T
    G = S1_truncated @ S1_truncated.T + S2_truncated @ S2_truncated.T
    
    # Eigendecomposition
    eigen_val, eigen_vec = eig(G)
    eigen_val = np.real(eigen_val)
    
    # Select eigenvectors for difference subspace
    # Eigenvalues between 0 and 1 (exclusive) indicate difference
    idx = np.where((1e-6 < eigen_val) & (eigen_val < 1))[0]
    
    if len(idx) == 0:
        # Return empty subspace if no suitable eigenvalues
        return np.zeros((min_points, 0))
    
    return eigen_vec[:, idx]


def generate_synthetic_floor(width=50, length=50, n_points_x=100, n_points_y=100, bumps=None):
    """Generate synthetic floor data with controllable bumps."""
    if bumps is None:
        bumps = [(25, 25, 1, 5)]  # Default bump in the middle

    x = np.linspace(0, width, n_points_x)
    y = np.linspace(0, length, n_points_y)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for x0, y0, amplitude, spread in bumps:
        Z += amplitude * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * spread**2))

    points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
    return points


def numpy_to_pointcloud(array):
    """Convert a NumPy array to an Open3D PointCloud object."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(array)
    return pcd


def divide_into_equal_patches(floor_points, n_patches_x=5, n_patches_y=5):
    """Divide floor points into patches with approximately equal number of points."""
    n_points = len(floor_points)
    floor_points_sorted_x = floor_points[np.argsort(floor_points[:, 0])]
    points_per_strip = n_points // n_patches_x

    patches = []
    for i in range(n_patches_x):
        start_idx = i * points_per_strip
        end_idx = start_idx + points_per_strip if i < n_patches_x - 1 else n_points
        strip = floor_points_sorted_x[start_idx:end_idx]

        # Sort strip by y coordinate
        strip_sorted_y = strip[np.argsort(strip[:, 1])]
        points_per_patch = len(strip_sorted_y) // n_patches_y

        for j in range(n_patches_y):
            start_idx_y = j * points_per_patch
            end_idx_y = start_idx_y + points_per_patch if j < n_patches_y - 1 else len(strip_sorted_y)
            patch = strip_sorted_y[start_idx_y:end_idx_y]
            patches.append(patch)

    return patches

def visualize_patches(patches, n_patches_x=5, n_patches_y=5,case_type="random"):
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
    plt.title(f'Floor divided into {n_patches_x}x{n_patches_y} patches')
    plt.savefig(os.path.join('output', f'{case_type}_patches.png'), dpi=300, bbox_inches='tight')
    plt.show()

def divide_into_equal_patches(floor_points, n_patches_x=5, n_patches_y=5):
    """Divide floor points into patches with approximately equal number of points."""
    n_points = len(floor_points)
    floor_points_sorted_x = floor_points[np.argsort(floor_points[:, 0])]
    points_per_strip = n_points // n_patches_x

    patches = []
    for i in range(n_patches_x):
        start_idx = i * points_per_strip
        end_idx = start_idx + points_per_strip if i < n_patches_x - 1 else n_points
        strip = floor_points_sorted_x[start_idx:end_idx]

        # Sort strip by y coordinate
        strip_sorted_y = strip[np.argsort(strip[:, 1])]
        points_per_patch = len(strip_sorted_y) // n_patches_y

        for j in range(n_patches_y):
            start_idx_y = j * points_per_patch
            end_idx_y = start_idx_y + points_per_patch if j < n_patches_y - 1 else len(strip_sorted_y)
            patch = strip_sorted_y[start_idx_y:end_idx_y]
            patches.append(patch)

    return patches


def cal_magnitude(S1,S2):
    _, S, _ = np.linalg.svd(S1.T @ S2)
    mag = np.sum(2*(1 - S))
    return mag

def calculate_patch_differences_without_ds(patches, cfg, n_patches_x=5, n_patches_y=5):
    """Calculate patch differences without using difference subspace."""
    average_magnitudes = []
    patch_subspaces = [generate_subspace(patch, cfg) for patch in patches]
    
    def get_neighbors(idx):
        row = idx // n_patches_x
        col = idx % n_patches_x
        neighbors = []
        weights = []
        
        # Up neighbor
        if row > 0:
            neighbors.append((idx - n_patches_x, "Up"))
            weights.append(1.0)
        # Right neighbor
        if col < n_patches_x - 1:
            neighbors.append((idx + 1, "Right"))
            weights.append(1.0)
        # Down neighbor
        if row < n_patches_y - 1:
            neighbors.append((idx + n_patches_x, "Down"))
            weights.append(1.0)
        # Left neighbor
        if col > 0:
            neighbors.append((idx - 1, "Left"))
            weights.append(1.0)
            
        return neighbors, weights
    
    for i in range(len(patches)):
        neighbors, weights = get_neighbors(i)
        patch_magnitudes = []
        print(f"\nPatch {i} (Row {i//n_patches_x}, Col {i%n_patches_x}):")
        
        for (neighbor_idx, direction), weight in zip(neighbors, weights):
            magnitude = cal_magnitude(patch_subspaces[i], patch_subspaces[neighbor_idx])
            patch_magnitudes.append(magnitude)
            print(f"  {direction} neighbor (Patch {neighbor_idx}): DS magnitude = {magnitude:.4f}")
        
        if patch_magnitudes:
            avg_magnitude = sum(patch_magnitudes) / sum(weights)
            print(f"  Average magnitude: {avg_magnitude:.4f}")
            average_magnitudes.append(avg_magnitude)
        else:
            print("  No neighbors found")
            average_magnitudes.append(0)
    
    print("\nGlobal average magnitude:", sum(average_magnitudes)/len(average_magnitudes))
    return average_magnitudes


def calculate_patch_differences_with_ds(patches, cfg, n_patches_x=5, n_patches_y=5):
    """Calculate patch differences using difference subspace."""
    average_magnitudes = []
    patch_subspaces = [generate_subspace(patch, cfg) for patch in patches]
    
    def get_neighbors(idx):
        row = idx // n_patches_x
        col = idx % n_patches_x
        neighbors = []
        weights = []
        
        # Up neighbor
        if row > 0:
            neighbors.append(idx - n_patches_x)
            weights.append(1.0)
        # Right neighbor
        if col < n_patches_x - 1:
            neighbors.append(idx + 1)
            weights.append(1.0)
        # Down neighbor
        if row < n_patches_y - 1:
            neighbors.append(idx + n_patches_x)
            weights.append(1.0)
        # Left neighbor
        if col > 0:
            neighbors.append(idx - 1)
            weights.append(1.0)
            
        return neighbors, weights
    
    for i in range(len(patches)):
        neighbors, weights = get_neighbors(i)
        patch_magnitudes = []
        
        for neighbor_idx, weight in zip(neighbors, weights):
            S1 = patch_subspaces[i]
            S2 = patch_subspaces[neighbor_idx]
            
            if S1.shape[1] == cfg.subspace_dim and S2.shape[1] == cfg.subspace_dim:
                DS = gen_shape_difference_subspace(S1, S2, cfg)
                if DS.shape[1] > 0:
                    # Project S1 onto difference subspace
                    # Truncate to match dimensions
                    min_points = min(S1.shape[0], DS.shape[0])
                    S1_trunc = S1[:min_points, :]
                    DS_trunc = DS[:min_points, :]
                    
                    P = DS_trunc @ DS_trunc.T
                    V = P @ S1_trunc
                    magnitude = np.linalg.norm(V)
                    patch_magnitudes.append(magnitude)
        
        if patch_magnitudes:
            avg_magnitude = sum(patch_magnitudes) / sum(weights)
            average_magnitudes.append(avg_magnitude)
        else:
            average_magnitudes.append(0)
    
    return average_magnitudes


def calculate_patch_differences_without_ds_8neighbors(patches, cfg, n_patches_x=5, n_patches_y=5):
    """Calculate patch differences without using difference subspace, considering all 8 neighbors."""
    average_magnitudes = []
    patch_subspaces = [generate_subspace(patch, cfg) for patch in patches]
    
    def get_neighbors_8(idx):
        row = idx // n_patches_x
        col = idx % n_patches_x
        neighbors = []
        weights = []
        directions = []
        
        # Check all 8 surrounding positions
        neighbor_positions = [
            (-1, -1, "UpLeft"), (-1, 0, "Up"), (-1, 1, "UpRight"),
            (0, -1, "Left"), (0, 1, "Right"),
            (1, -1, "DownLeft"), (1, 0, "Down"), (1, 1, "DownRight")
        ]
        
        for dr, dc, direction in neighbor_positions:
            new_row = row + dr
            new_col = col + dc
            if (0 <= new_row < n_patches_y) and (0 <= new_col < n_patches_x):
                neighbor_idx = new_row * n_patches_x + new_col
                neighbors.append(neighbor_idx)
                weights.append(1.0)
                directions.append(direction)
                
        return neighbors, weights, directions
    
    for i in range(len(patches)):
        neighbors, weights, directions = get_neighbors_8(i)
        patch_magnitudes = []
        print(f"\nPatch {i} (Row {i//n_patches_x}, Col {i%n_patches_x}):")
        
        for neighbor_idx, (weight, direction) in enumerate(zip(weights, directions)):
            magnitude = cal_magnitude(patch_subspaces[i], patch_subspaces[neighbors[neighbor_idx]])
            patch_magnitudes.append(magnitude)
            print(f"  {direction} neighbor (Patch {neighbors[neighbor_idx]}): DS magnitude = {magnitude:.4f}")
        
        if patch_magnitudes:
            avg_magnitude = sum(patch_magnitudes) / sum(weights)
            print(f"  Average magnitude: {avg_magnitude:.4f}")
            average_magnitudes.append(avg_magnitude)
        else:
            print("  No neighbors found")
            average_magnitudes.append(0)
    
    print("\nGlobal average magnitude:", sum(average_magnitudes)/len(average_magnitudes))
    return average_magnitudes

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
                if neighbor_idx < len(patches):  # Check bounds
                    neighbors.append(neighbor_idx)
                    weights.append(1.0)
                
        return neighbors, weights
    
    for i in range(len(patches)):
        neighbors, weights = get_neighbors_8(i)
        patch_magnitudes = []
        
        for neighbor_idx, weight in zip(neighbors, weights):
            if neighbor_idx < len(patch_subspaces):  # Check bounds
                S1 = patch_subspaces[i]
                S2 = patch_subspaces[neighbor_idx]
                
                # Ensure both subspaces have same number of columns
                if S1.shape[1] == cfg.subspace_dim and S2.shape[1] == cfg.subspace_dim:
                    DS = gen_shape_difference_subspace(S1, S2, cfg)
                    if DS.shape[1] > 0:
                        # Project S1 onto difference subspace
                        # Truncate S1 to match DS dimensions if needed
                        min_points = min(S1.shape[0], DS.shape[0])
                        S1_trunc = S1[:min_points, :]
                        DS_trunc = DS[:min_points, :]
                        
                        # Projection matrix P = DS @ DS.T
                        P = DS_trunc @ DS_trunc.T
                        # Project S1 onto difference subspace
                        V = P @ S1_trunc
                        magnitude = np.linalg.norm(V)
                        patch_magnitudes.append(magnitude)
        
        if patch_magnitudes:
            avg_magnitude = sum(patch_magnitudes) / sum(weights)
            average_magnitudes.append(avg_magnitude)
        else:
            average_magnitudes.append(0)
    
    return average_magnitudes


def visualize_patch_differences(patches, average_magnitudes, n_patches_x=5, n_patches_y=5, case_type="random"):
    """Visualize the average DS magnitudes with a cool color scheme."""
    fig = plt.figure(figsize=(15, 6))
   
    # Convert magnitudes to numpy array and reshape
    heatmap = np.array(average_magnitudes).reshape(n_patches_y, n_patches_x)
   
    # Set consistent color scaling for both plots
    vmin = 0
    vmax = max(1e-6, np.max(heatmap))
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
   
    # Use PuBu colormap for a cool purple-to-blue gradient
    cmap = plt.cm.PuBu
   
    # 3D scatter plot
    ax_3d = fig.add_subplot(121, projection='3d')
   
    # Plot patches with consistent coloring
    for idx, patch in enumerate(patches):
        if len(patch) > 0:
            color_val = average_magnitudes[idx]
            colors = np.array([cmap(norm(color_val))] * len(patch))
            ax_3d.scatter(patch[:, 0], patch[:, 1], patch[:, 2],
                         c=colors, alpha=0.6, s=1)
   
    ax_3d.set_title('Patches colored by average DS magnitude')
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
   
    # Add colorbar for 3D plot
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax_3d, label='Average DS Magnitude')
   
    # Plot heatmap
    ax_heat = fig.add_subplot(122)
    im = ax_heat.imshow(heatmap, cmap=cmap, norm=norm)
    ax_heat.set_title('Average DS Magnitude Heatmap')
    plt.colorbar(im, ax=ax_heat, label='Average DS Magnitude')
   
    # Add grid lines
    ax_heat.set_xticks(np.arange(-.5, n_patches_x, 1), minor=True)
    ax_heat.set_yticks(np.arange(-.5, n_patches_y, 1), minor=True)
    ax_heat.grid(which="minor", color="w", linestyle='-', linewidth=2)
   
    # Add value labels to heatmap with dynamic text color
    for i in range(n_patches_y):
        for j in range(n_patches_x):
            # Calculate appropriate text color based on background intensity
            color_val = heatmap[i, j]
            text_color = 'white' if color_val > (vmax + vmin) / 2 else 'black'
            
            ax_heat.text(j, i, f'{heatmap[i, j]:.6f}',
                        ha='center', va='center',
                        color=text_color)
   
    # Print unique values for debugging
    print("\nAverage DS magnitude:", (heatmap))
    plt.savefig(os.path.join('output', f'{case_type}_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
   
    return heatmap

def visualize_patch_and_neighbors_without_ds(patches, patch_idx, cfg, n_patches_x=5, n_patches_y=5):
    """
    Visualize a specific patch and its neighbors, along with their DS values.
    The selected patch's DS value is the average of its neighbors' DS values.
    
    Args:
        patches: List of patches, where each patch is a Nx3 array of points
        patch_idx: Index of the patch to analyze
        cfg: Configuration object containing subspace_dim
        n_patches_x: Number of patches in x direction
        n_patches_y: Number of patches in y direction
    """
    # Get neighbors
    row = patch_idx // n_patches_x
    col = patch_idx % n_patches_x
    neighbors = []
    directions = []
    
    # Up neighbor
    if row > 0:
        neighbors.append(patch_idx - n_patches_x)
        directions.append('Up')
    
    # Right neighbor
    if col < n_patches_x - 1:
        neighbors.append(patch_idx + 1)
        directions.append('Right')
        
    # Down neighbor
    if row < n_patches_y - 1:
        neighbors.append(patch_idx + n_patches_x)
        directions.append('Down')
        
    # Left neighbor
    if col > 0:
        neighbors.append(patch_idx - 1)
        directions.append('Left')
    
    # Calculate DS with each neighbor
    ds_values = []
    neighbor_ds_values = {}  # Store DS values by direction
    for neighbor_idx in neighbors:
        patch_subspace = generate_subspace(patches[patch_idx], cfg)
        neighbor_subspace = generate_subspace(patches[neighbor_idx], cfg)
        DS = gen_shape_difference_subspace(patch_subspace, neighbor_subspace, cfg)
        magnitude = cal_magnitude(patches[patch_idx], patches[neighbor_idx])
        ds_values.append(magnitude)
        direction = directions[len(neighbor_ds_values)]
        neighbor_ds_values[direction] = magnitude

    # Calculate average DS for the selected patch
    selected_ds = sum(ds_values) / len(ds_values) if ds_values else 0
    
    # Create visualization
    fig = plt.figure(figsize=(15, 10))
    
    # 3D scatter plot
    ax_3d = fig.add_subplot(121, projection='3d')
    
    # Plot central patch in red
    ax_3d.scatter(patches[patch_idx][:, 0], 
                 patches[patch_idx][:, 1], 
                 patches[patch_idx][:, 2],
                 c='red', s=2, label=f'Patch {patch_idx} (DS avg: {selected_ds:.4f})')
    
    # Plot neighbors with different colors
    colors = ['blue', 'green', 'purple', 'orange']
    for neighbor_idx, direction, ds_value, color in zip(neighbors, directions, ds_values, colors):
        ax_3d.scatter(patches[neighbor_idx][:, 0],
                     patches[neighbor_idx][:, 1],
                     patches[neighbor_idx][:, 2],
                     c=color, s=2, 
                     label=f'{direction} neighbor (DS: {ds_value:.4f})')
    
    ax_3d.set_title(f'Patch {patch_idx} and its neighbors')
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.legend()
    
    # 2D grid visualization
    ax_grid = fig.add_subplot(122)
    grid = np.zeros((n_patches_y, n_patches_x))
    
    # Mark central patch
    grid[row, col] = 2
    
    # Mark neighbors
    for neighbor_idx in neighbors:
        n_row = neighbor_idx // n_patches_x
        n_col = neighbor_idx % n_patches_x
        grid[n_row, n_col] = 1
    
    # Create grid visualization
    im = ax_grid.imshow(grid, cmap='RdBu')
    
    # Add DS values as text
    for neighbor_idx, direction, ds_value in zip(neighbors, directions, ds_values):
        n_row = neighbor_idx // n_patches_x
        n_col = neighbor_idx % n_patches_x
        ax_grid.text(n_col, n_row, f'{ds_value:.4f}', 
                    ha='center', va='center', color='black')
    
    # Add selected patch DS value (average)
    ax_grid.text(col, row, f'{selected_ds:.4f}', 
                ha='center', va='center', color='white')
    
    ax_grid.set_title('Grid View with DS Values')
    
    # Add grid lines
    ax_grid.set_xticks(np.arange(-.5, n_patches_x, 1), minor=True)
    ax_grid.set_yticks(np.arange(-.5, n_patches_y, 1), minor=True)
    ax_grid.grid(which="minor", color="w", linestyle='-', linewidth=2)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed information
    print(f"\nDetailed DS analysis for Patch {patch_idx}:")
    print(f"Location: Row {row}, Column {col}")
    print(f"Average DS magnitude: {selected_ds:.4f}")
    print("\nNeighbor analysis:")
    for direction, neighbor_idx, ds_value in zip(directions, neighbors, ds_values):
        print(f"{direction} neighbor (Patch {neighbor_idx}): DS magnitude = {ds_value:.4f}")
def visualize_patch_and_neighbors_with_ds(patches, patch_idx, cfg, n_patches_x=5, n_patches_y=5):
    """
    Visualize a specific patch and its neighbors, along with their DS values.
    The selected patch's DS value is the average of its neighbors' DS values.
    
    Args:
        patches: List of patches, where each patch is a Nx3 array of points
        patch_idx: Index of the patch to analyze
        cfg: Configuration object containing subspace_dim
        n_patches_x: Number of patches in x direction
        n_patches_y: Number of patches in y direction
    """
    # Get neighbors
    row = patch_idx // n_patches_x
    col = patch_idx % n_patches_x
    neighbors = []
    directions = []
    
    # Up neighbor
    if row > 0:
        neighbors.append(patch_idx - n_patches_x)
        directions.append('Up')
    
    # Right neighbor
    if col < n_patches_x - 1:
        neighbors.append(patch_idx + 1)
        directions.append('Right')
        
    # Down neighbor
    if row < n_patches_y - 1:
        neighbors.append(patch_idx + n_patches_x)
        directions.append('Down')
        
    # Left neighbor
    if col > 0:
        neighbors.append(patch_idx - 1)
        directions.append('Left')
    
    # Calculate DS with each neighbor
    ds_values = []
    neighbor_ds_values = {}  # Store DS values by direction
    for neighbor_idx in neighbors:
        patch_subspace = generate_subspace(patches[patch_idx], cfg)
        neighbor_subspace = generate_subspace(patches[neighbor_idx], cfg)
        DS = gen_shape_difference_subspace(patch_subspace, neighbor_subspace, cfg)
        if DS.shape[1] > 0:
            # Truncate to match dimensions
            min_points = min(patch_subspace.shape[0], DS.shape[0])
            patch_sub_trunc = patch_subspace[:min_points, :]
            DS_trunc = DS[:min_points, :]
            
            P = DS_trunc @ DS_trunc.T
            V = P @ patch_sub_trunc
            magnitude = np.linalg.norm(V)
            ds_values.append(magnitude)
            direction = directions[len(neighbor_ds_values)]
            neighbor_ds_values[direction] = magnitude

    # Calculate average DS for the selected patch
    selected_ds = sum(ds_values) / len(ds_values) if ds_values else 0
    
    # Create visualization
    fig = plt.figure(figsize=(15, 10))
    
    # 3D scatter plot
    ax_3d = fig.add_subplot(121, projection='3d')
    
    # Plot central patch in red
    ax_3d.scatter(patches[patch_idx][:, 0], 
                 patches[patch_idx][:, 1], 
                 patches[patch_idx][:, 2],
                 c='red', s=2, label=f'Patch {patch_idx} (DS avg: {selected_ds:.4f})')
    
    # Plot neighbors with different colors
    colors = ['blue', 'green', 'purple', 'orange']
    for neighbor_idx, direction, ds_value, color in zip(neighbors, directions, ds_values, colors):
        ax_3d.scatter(patches[neighbor_idx][:, 0],
                     patches[neighbor_idx][:, 1],
                     patches[neighbor_idx][:, 2],
                     c=color, s=2, 
                     label=f'{direction} neighbor (DS: {ds_value:.4f})')
    
    ax_3d.set_title(f'Patch {patch_idx} and its neighbors')
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.legend()
    
    # 2D grid visualization
    ax_grid = fig.add_subplot(122)
    grid = np.zeros((n_patches_y, n_patches_x))
    
    # Mark central patch
    grid[row, col] = 2
    
    # Mark neighbors
    for neighbor_idx in neighbors:
        n_row = neighbor_idx // n_patches_x
        n_col = neighbor_idx % n_patches_x
        grid[n_row, n_col] = 1
    
    # Create grid visualization
    im = ax_grid.imshow(grid, cmap='RdBu')
    
    # Add DS values as text
    for neighbor_idx, direction, ds_value in zip(neighbors, directions, ds_values):
        n_row = neighbor_idx // n_patches_x
        n_col = neighbor_idx % n_patches_x
        ax_grid.text(n_col, n_row, f'{ds_value:.4f}', 
                    ha='center', va='center', color='black')
    
    # Add selected patch DS value (average)
    ax_grid.text(col, row, f'{selected_ds:.4f}', 
                ha='center', va='center', color='white')
    
    ax_grid.set_title('Grid View with DS Values')
    
    # Add grid lines
    ax_grid.set_xticks(np.arange(-.5, n_patches_x, 1), minor=True)
    ax_grid.set_yticks(np.arange(-.5, n_patches_y, 1), minor=True)
    ax_grid.grid(which="minor", color="w", linestyle='-', linewidth=2)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed information
    print(f"\nDetailed DS analysis for Patch {patch_idx}:")
    print(f"Location: Row {row}, Column {col}")
    print(f"Average DS magnitude: {selected_ds:.4f}")
    print("\nNeighbor analysis:")
    for direction, neighbor_idx, ds_value in zip(directions, neighbors, ds_values):
        print(f"{direction} neighbor (Patch {neighbor_idx}): DS magnitude = {ds_value:.4f}")


def divide_into_patches(points, n_patches_x=5, n_patches_y=5, n_points_x=100, n_points_y=100):
    """
    Divide floor points into patches while preserving grid structure.

    Args:
        points: Nx3 array of points from generate_synthetic_floor
        n_patches_x: Number of patches along x-axis
        n_patches_y: Number of patches along y-axis
        n_points_x: Number of points along x-axis in original grid
        n_points_y: Number of points along y-axis in original grid

    Returns:
        List of patches, where each patch preserves the original grid structure
    """
    # Calculate points per patch
    points_per_patch_x = n_points_x // n_patches_x
    points_per_patch_y = n_points_y // n_patches_y

    # Reshape points back to grid structure
    points_grid = points.reshape(n_points_y, n_points_x, 3)

    patches = []
    for i in range(n_patches_y):  # Note: y first to maintain row-major order
        for j in range(n_patches_x):
            # Extract patch while maintaining grid structure
            patch = points_grid[
                i * points_per_patch_y:(i + 1) * points_per_patch_y,
                j * points_per_patch_x:(j + 1) * points_per_patch_x,
                :
            ]
            # Flatten patch while preserving order
            patch_flat = patch.reshape(-1, 3)
            patches.append(patch_flat)

    return patches



def visualize_patch_and_neighbors_with_ds_8neighbor(patches, patch_idx, cfg, n_patches_x=5, n_patches_y=5):
    """
    Visualize a specific patch and its neighbors (including diagonals), along with their DS values.
    The selected patch's DS value is the average of its neighbors' DS values.
    
    Args:
        patches: List of patches, where each patch is a Nx3 array of points
        patch_idx: Index of the patch to analyze
        cfg: Configuration object containing subspace_dim
        n_patches_x: Number of patches in x direction
        n_patches_y: Number of patches in y direction
    """
    # Get neighbors
    row = patch_idx // n_patches_x
    col = patch_idx % n_patches_x
    neighbors = []
    directions = []
    
    # Up-Left neighbor
    if row > 0 and col > 0:
        neighbors.append(patch_idx - n_patches_x - 1)
        directions.append('Up-Left')
    
    # Up neighbor
    if row > 0:
        neighbors.append(patch_idx - n_patches_x)
        directions.append('Up')
    
    # Up-Right neighbor
    if row > 0 and col < n_patches_x - 1:
        neighbors.append(patch_idx - n_patches_x + 1)
        directions.append('Up-Right')
    
    # Left neighbor
    if col > 0:
        neighbors.append(patch_idx - 1)
        directions.append('Left')
    
    # Right neighbor
    if col < n_patches_x - 1:
        neighbors.append(patch_idx + 1)
        directions.append('Right')
    
    # Down-Left neighbor
    if row < n_patches_y - 1 and col > 0:
        neighbors.append(patch_idx + n_patches_x - 1)
        directions.append('Down-Left')
    
    # Down neighbor
    if row < n_patches_y - 1:
        neighbors.append(patch_idx + n_patches_x)
        directions.append('Down')
    
    # Down-Right neighbor
    if row < n_patches_y - 1 and col < n_patches_x - 1:
        neighbors.append(patch_idx + n_patches_x + 1)
        directions.append('Down-Right')
    
    # Calculate DS with each neighbor
    ds_values = []
    neighbor_ds_values = {}  # Store DS values by direction
    for neighbor_idx in neighbors:
        patch_subspace = generate_subspace(patches[patch_idx], cfg)
        neighbor_subspace = generate_subspace(patches[neighbor_idx], cfg)
        DS = gen_shape_difference_subspace(patch_subspace, neighbor_subspace, cfg)
        if DS.shape[1] > 0:
            # Truncate to match dimensions
            min_points = min(patch_subspace.shape[0], DS.shape[0])
            patch_sub_trunc = patch_subspace[:min_points, :]
            DS_trunc = DS[:min_points, :]
            
            P = DS_trunc @ DS_trunc.T
            V = P @ patch_sub_trunc
            magnitude = np.linalg.norm(V)
            ds_values.append(magnitude)
            direction = directions[len(neighbor_ds_values)]
            neighbor_ds_values[direction] = magnitude

    # Calculate average DS for the selected patch
    selected_ds = sum(ds_values) / len(ds_values) if ds_values else 0
    
    # Create visualization
    fig = plt.figure(figsize=(15, 10))
    
    # 3D scatter plot
    ax_3d = fig.add_subplot(121, projection='3d')
    
    # Plot central patch in red
    ax_3d.scatter(patches[patch_idx][:, 0], 
                 patches[patch_idx][:, 1], 
                 patches[patch_idx][:, 2],
                 c='red', s=2, label=f'Patch {patch_idx} (DS avg: {selected_ds:.4f})')
    
    # Plot neighbors with different colors
    colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'brown']
    for neighbor_idx, direction, ds_value, color in zip(neighbors, directions, ds_values, colors):
        ax_3d.scatter(patches[neighbor_idx][:, 0],
                     patches[neighbor_idx][:, 1],
                     patches[neighbor_idx][:, 2],
                     c=color, s=2, 
                     label=f'{direction} neighbor (DS: {ds_value:.4f})')
    
    ax_3d.set_title(f'Patch {patch_idx} and its neighbors')
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2D grid visualization
    ax_grid = fig.add_subplot(122)
    grid = np.zeros((n_patches_y, n_patches_x))
    
    # Mark central patch
    grid[row, col] = 2
    
    # Mark neighbors
    for neighbor_idx in neighbors:
        n_row = neighbor_idx // n_patches_x
        n_col = neighbor_idx % n_patches_x
        grid[n_row, n_col] = 1
    
    # Create grid visualization
    im = ax_grid.imshow(grid, cmap='RdBu')
    
    # Add DS values as text
    for neighbor_idx, direction, ds_value in zip(neighbors, directions, ds_values):
        n_row = neighbor_idx // n_patches_x
        n_col = neighbor_idx % n_patches_x
        ax_grid.text(n_col, n_row, f'{ds_value:.4f}', 
                    ha='center', va='center', color='black',
                    fontsize=8)  # Reduced font size for better fit
    
    # Add selected patch DS value (average)
    ax_grid.text(col, row, f'{selected_ds:.4f}', 
                ha='center', va='center', color='white')
    
    ax_grid.set_title('Grid View with DS Values')
    
    # Add grid lines
    ax_grid.set_xticks(np.arange(-.5, n_patches_x, 1), minor=True)
    ax_grid.set_yticks(np.arange(-.5, n_patches_y, 1), minor=True)
    ax_grid.grid(which="minor", color="w", linestyle='-', linewidth=2)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed information
    print(f"\nDetailed DS analysis for Patch {patch_idx}:")
    print(f"Location: Row {row}, Column {col}")
    print(f"Average DS magnitude: {selected_ds:.4f}")
    print("\nNeighbor analysis:")
    for direction, neighbor_idx, ds_value in zip(directions, neighbors, ds_values):
        print(f"{direction} neighbor (Patch {neighbor_idx}): DS magnitude = {ds_value:.4f}")



# Main Workflow Example
random_bumps = [
    (random.uniform(1, 49), random.uniform(1, 49), random.uniform(0.3, 0.8), random.uniform(0.5, 1.5))
    for _ in range(3)
]


def generate_test_case(case_type):
    """Generate comprehensive set of test patterns."""
    # Your existing patterns
    if case_type == "single":
        return [(25, 25, 1.0, 1.0)]
    elif case_type == "double":
        return [(20, 25, 1.0, 1.0), (30, 25, 1.0, 1.0)]
    
    elif case_type == "ridge":
        return [
            (10, 25, 0.8, 1.0),  # Left part
            (25, 25, 0.8, 1.0),  # Middle
            (40, 25, 0.8, 1.0)   # Right part
        ]
    elif case_type == "diagonal":
        return [
            (10, 10, 0.5, 1.0),
            (25, 25, 0.5, 1.0),
            (40, 40, 0.5, 1.0)
        ]
    elif case_type == "corners":
        return [
            (10, 10, 0.7, 1.0),  # Bottom left
            (10, 40, 0.7, 1.0),  # Top left
            (40, 10, 0.7, 1.0),  # Bottom right
            (40, 40, 0.7, 1.0)   # Top right
        ]
    elif case_type == "triangle":
        return [(25, 25, 1.0, 1.0), 
                (20, 30, 1.0, 1.0),
                (30, 30, 1.0, 1.0)]
    elif case_type == "circle":
        radius = 15
        center_x, center_y = 25, 25
        return [
            (center_x + radius * np.cos(angle), 
             center_y + radius * np.sin(angle), 
             0.7, 1.0)
            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False)
        ]
    elif case_type == "random":
        return [
            (random.uniform(1, 49), random.uniform(1, 49), 
            random.uniform(0.3, 0.8), random.uniform(0.5, 1.5))
            for _ in range(3)
        ]

case_type = "circle"
gen_bumps = generate_test_case(case_type)

# Generate synthetic floor
floor_points = generate_synthetic_floor(
    width=50,
    length=50,
    n_points_x=100,
    n_points_y=100,
    bumps=gen_bumps
)

# Convert to point cloud for visualization
floor_pcd = numpy_to_pointcloud(floor_points)
Pointlib.visualize([floor_pcd], point_size = 10)

cfg = Confing()
subspace_dim = 3
n_patches_x = 10
n_patches_y = 10



patches = divide_into_patches(floor_points, n_patches_x, n_patches_y,)
visualize_patches(patches, n_patches_x,n_patches_y, case_type)
#patch_0_0 = patches[0]  # First patch
#patch_2_3 = patches[2 * 5 + 3]  # Patch at position (2,3) in the grid

differences = calculate_patch_differences_with_ds_8neighbors(patches, cfg, n_patches_x,n_patches_y)

visualize_patch_differences(patches, differences,n_patches_x,n_patches_y, case_type)

visualize_patch_and_neighbors_with_ds(patches, 12, cfg,n_patches_x,n_patches_y)

#visualize_patch_and_neighbors_with_ds_8neighbor(patches, 12, cfg,n_patches_x,n_patches_y)

# Test sliding window approach
print("\n" + "="*60)
print("Testing Sliding Window Approach")
print("="*60)

from grid_patch_division_with_stride import divide_into_grid_patches_with_stride

# Test with 50% overlap only for clearer comparison
overlap = 0.5
stride = 1.0 - overlap
print(f"\nSliding Window with {overlap*100}% overlap:")

sliding_patches, patch_info = divide_into_grid_patches_with_stride(
    floor_points, n_patches_x=n_patches_x, n_patches_y=n_patches_y,
    stride_x=stride, stride_y=stride
)

# Calculate differences using sliding window
n_x, n_y = patch_info['grid_size']
sliding_differences = calculate_patch_differences_with_ds_8neighbors(
    sliding_patches, cfg, n_x, n_y
)

# Visualize results
visualize_patch_differences(sliding_patches, sliding_differences, n_x, n_y, 
                           f"{case_type}_sliding_{int(overlap*100)}pct")

print(f"Valid patches: {len(sliding_patches)}")
print(f"Max magnitude: {max(sliding_differences):.6f}")
print(f"Mean magnitude: {sum(sliding_differences)/len(sliding_differences):.6f}")

# Compare fixed vs sliding window
print("\n" + "="*60)
print("Comparison: Fixed Patches vs Sliding Window")
print("="*60)
print(f"Fixed patches: {n_patches_x}x{n_patches_y} = {len(patches)} patches")
print(f"  Max magnitude: {max(differences):.6f}")
print(f"  Mean magnitude: {sum(differences)/len(differences):.6f}")
print(f"\nSliding window (50% overlap): {n_x}x{n_y} = {len(sliding_patches)} windows")
print(f"  Max magnitude: {max(sliding_differences):.6f}")
print(f"  Mean magnitude: {sum(sliding_differences)/len(sliding_differences):.6f}")
print(f"\nIncrease in resolution: {len(sliding_patches)/len(patches):.1f}x")

