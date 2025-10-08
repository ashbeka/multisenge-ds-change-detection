import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

def visualize_patch_vs_sliding():
    """Create a visual comparison of fixed patches vs sliding window."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Common parameters
    floor_size = 10
    patch_size = 2.5
    
    # Fixed Patches (Left)
    ax1.set_title('Fixed Patches (Non-overlapping)', fontsize=16, fontweight='bold')
    ax1.set_xlim(0, floor_size)
    ax1.set_ylim(0, floor_size)
    ax1.set_aspect('equal')
    
    # Draw fixed patches
    n_patches = int(floor_size / patch_size)
    colors = plt.cm.Set3(np.linspace(0, 1, n_patches * n_patches))
    patch_idx = 0
    
    for i in range(n_patches):
        for j in range(n_patches):
            x = j * patch_size
            y = i * patch_size
            rect = Rectangle((x, y), patch_size, patch_size, 
                           facecolor=colors[patch_idx], 
                           edgecolor='black', linewidth=2,
                           alpha=0.6)
            ax1.add_patch(rect)
            ax1.text(x + patch_size/2, y + patch_size/2, f'{patch_idx}',
                    ha='center', va='center', fontsize=10, fontweight='bold')
            patch_idx += 1
    
    # Add a "bump" location
    bump_x, bump_y = 5, 5
    circle1 = plt.Circle((bump_x, bump_y), 0.5, color='red', alpha=0.8)
    ax1.add_patch(circle1)
    ax1.text(bump_x, bump_y-0.8, 'Bump', ha='center', fontweight='bold', color='red')
    
    # Grid lines
    for i in range(n_patches + 1):
        ax1.axvline(i * patch_size, color='black', linewidth=1, alpha=0.5)
        ax1.axhline(i * patch_size, color='black', linewidth=1, alpha=0.5)
    
    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('Y', fontsize=12)
    ax1.text(5, -0.8, f'Total: {n_patches}×{n_patches} = {n_patches*n_patches} patches', 
             ha='center', fontsize=12)
    
    # Sliding Window (Right)
    ax2.set_title('Sliding Window (50% Overlap)', fontsize=16, fontweight='bold')
    ax2.set_xlim(0, floor_size)
    ax2.set_ylim(0, floor_size)
    ax2.set_aspect('equal')
    
    # Draw sliding windows with overlap
    stride = patch_size * 0.5  # 50% overlap
    n_windows = int((floor_size - patch_size) / stride) + 1
    
    # Draw some example windows to show overlap
    example_positions = [(0, 0), (1.25, 0), (2.5, 0), (0, 1.25), (1.25, 1.25)]
    alphas = [0.3, 0.3, 0.3, 0.3, 0.5]
    colors_sliding = ['blue', 'green', 'purple', 'orange', 'red']
    
    for idx, (x, y) in enumerate(example_positions):
        rect = Rectangle((x, y), patch_size, patch_size,
                       facecolor=colors_sliding[idx % len(colors_sliding)],
                       edgecolor='black', linewidth=1,
                       alpha=alphas[idx])
        ax2.add_patch(rect)
    
    # Show overlap region
    overlap_rect = Rectangle((1.25, 1.25), 1.25, 1.25,
                           facecolor='yellow', edgecolor='red', 
                           linewidth=3, alpha=0.8, linestyle='--')
    ax2.add_patch(overlap_rect)
    ax2.text(1.875, 1.875, 'Overlap\nRegion', ha='center', va='center', 
             fontsize=10, fontweight='bold', color='red')
    
    # Add the same bump
    circle2 = plt.Circle((bump_x, bump_y), 0.5, color='red', alpha=0.8)
    ax2.add_patch(circle2)
    ax2.text(bump_x, bump_y-0.8, 'Bump', ha='center', fontweight='bold', color='red')
    
    # Grid lines (finer)
    for i in range(n_windows):
        ax2.axvline(i * stride, color='gray', linewidth=0.5, alpha=0.3, linestyle='--')
        ax2.axhline(i * stride, color='gray', linewidth=0.5, alpha=0.3, linestyle='--')
    
    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('Y', fontsize=12)
    ax2.text(5, -0.8, f'Total: {n_windows}×{n_windows} = {n_windows*n_windows} windows', 
             ha='center', fontsize=12)
    
    plt.suptitle('Fixed Patches vs Sliding Window Comparison', fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('output/patch_vs_sliding_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a second figure showing coverage difference
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Fixed patches - point coverage
    ax3.set_title('Point Analysis in Fixed Patches', fontsize=16, fontweight='bold')
    ax3.set_xlim(0, floor_size)
    ax3.set_ylim(0, floor_size)
    ax3.set_aspect('equal')
    
    # Show a point at patch boundary
    boundary_point = (2.5, 2.5)
    ax3.scatter(*boundary_point, s=200, c='red', marker='*', zorder=5)
    ax3.text(boundary_point[0]+0.3, boundary_point[1]+0.3, 'Boundary Point', 
             fontsize=10, fontweight='bold', color='red')
    
    # Highlight the 4 patches around the boundary
    boundary_patches = [(0, 0), (2.5, 0), (0, 2.5), (2.5, 2.5)]
    for x, y in boundary_patches:
        rect = Rectangle((x, y), patch_size, patch_size,
                       facecolor='lightblue', edgecolor='blue', 
                       linewidth=2, alpha=0.3)
        ax3.add_patch(rect)
    
    # Add arrows showing which patch analyzes the point
    ax3.annotate('', xy=(1.25, 1.25), xytext=boundary_point,
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax3.text(1.8, 1.8, 'Belongs to\nONE patch', ha='center', fontsize=10, 
             fontweight='bold', color='blue')
    
    # Grid
    for i in range(n_patches + 1):
        ax3.axvline(i * patch_size, color='black', linewidth=1)
        ax3.axhline(i * patch_size, color='black', linewidth=1)
    
    ax3.set_xlabel('X', fontsize=12)
    ax3.set_ylabel('Y', fontsize=12)
    
    # Sliding window - point coverage
    ax4.set_title('Point Analysis in Sliding Window', fontsize=16, fontweight='bold')
    ax4.set_xlim(0, floor_size)
    ax4.set_ylim(0, floor_size)
    ax4.set_aspect('equal')
    
    # Same boundary point
    ax4.scatter(*boundary_point, s=200, c='red', marker='*', zorder=5)
    ax4.text(boundary_point[0]+0.3, boundary_point[1]+0.3, 'Same Point', 
             fontsize=10, fontweight='bold', color='red')
    
    # Show overlapping windows that contain the point
    overlapping_windows = [(0, 0), (1.25, 0), (0, 1.25), (1.25, 1.25)]
    for idx, (x, y) in enumerate(overlapping_windows):
        rect = Rectangle((x, y), patch_size, patch_size,
                       facecolor=colors_sliding[idx], edgecolor='black',
                       linewidth=2, alpha=0.2)
        ax4.add_patch(rect)
        
        # Draw arrows from point to each window center
        window_center = (x + patch_size/2, y + patch_size/2)
        ax4.annotate('', xy=window_center, xytext=boundary_point,
                    arrowprops=dict(arrowstyle='->', color=colors_sliding[idx], 
                                  lw=2, alpha=0.7))
    
    ax4.text(boundary_point[0], boundary_point[1]-0.5, 
             'Analyzed by\nFOUR windows', ha='center', fontsize=10, 
             fontweight='bold', color='green')
    
    # Finer grid
    for i in range(n_windows):
        ax4.axvline(i * stride, color='gray', linewidth=0.5, alpha=0.3, linestyle='--')
        ax4.axhline(i * stride, color='gray', linewidth=0.5, alpha=0.3, linestyle='--')
    
    ax4.set_xlabel('X', fontsize=12)
    ax4.set_ylabel('Y', fontsize=12)
    
    plt.suptitle('Point Coverage: Fixed Patches vs Sliding Window', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/point_coverage_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nKey Differences Summary:")
    print("="*50)
    print("FIXED PATCHES:")
    print("- Non-overlapping regions")
    print("- Each point belongs to exactly ONE patch")
    print("- Faster computation")
    print("- May miss anomalies at boundaries")
    print(f"- Example: 4×4 grid = 16 patches")
    print("\nSLIDING WINDOW:")
    print("- Overlapping regions")
    print("- Each point can belong to MULTIPLE windows")
    print("- More computation but better coverage")
    print("- Better boundary detection")
    print(f"- Example: 7×7 grid = 49 windows (with 50% overlap)")

if __name__ == "__main__":
    visualize_patch_vs_sliding()