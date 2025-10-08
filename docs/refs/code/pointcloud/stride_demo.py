#!/usr/bin/env python3
"""
Demonstration script for stride-based overlapping patches in point cloud processing.
This script shows how to use the new divide_into_grid_patches_with_stride function.
"""

import numpy as np
import matplotlib.pyplot as plt
from main import (
    divide_into_equal_patches,
    divide_into_grid_patches_with_stride,
    visualize_patches,
    load_floor_points_from_csv,
    calculate_patch_differences_with_ds,
    visualize_patch_differences,
    generate_subspace
)
from config import Confing
import os

def generate_synthetic_floor_with_bumps(n_points=10000, floor_size=10.0, n_bumps=3):
    """Generate synthetic floor data with bumps for testing."""
    # Generate random points on a flat floor
    x = np.random.uniform(0, floor_size, n_points)
    y = np.random.uniform(0, floor_size, n_points)
    z = np.random.normal(0, 0.01, n_points)  # Slight noise for realistic floor
    
    # Add bumps
    for i in range(n_bumps):
        # Random bump center
        bump_x = np.random.uniform(1, floor_size-1)
        bump_y = np.random.uniform(1, floor_size-1)
        bump_radius = np.random.uniform(0.5, 1.0)
        bump_height = np.random.uniform(0.1, 0.3)
        
        # Add gaussian bump
        distances = np.sqrt((x - bump_x)**2 + (y - bump_y)**2)
        bump_mask = distances < bump_radius
        z[bump_mask] += bump_height * np.exp(-(distances[bump_mask] / (bump_radius/2))**2)
    
    return np.column_stack([x, y, z])

def compare_patch_methods():
    """Compare traditional equal patches vs stride-based grid patches."""
    print("=== Comparing Patch Division Methods ===")
    
    # Generate or load test data
    try:
        # Try to load existing CSV data
        floor_points = load_floor_points_from_csv('long_floor_with_many_bumps.csv')
        print("Loaded existing CSV data")
    except:
        # Generate synthetic data if CSV doesn't exist
        print("Generating synthetic floor data...")
        floor_points = generate_synthetic_floor_with_bumps(n_points=5000, floor_size=10.0, n_bumps=5)
    
    print(f"Total points: {len(floor_points)}")
    
    # Configuration
    cfg = Confing()
    n_patches_x, n_patches_y = 5, 5
    
    # Method 1: Traditional equal patches
    print("\n1. Traditional equal patches (sorting-based):")
    patches_equal = divide_into_equal_patches(floor_points, n_patches_x, n_patches_y)
    print(f"   Number of patches: {len(patches_equal)}")
    print(f"   Points per patch: {[len(p) for p in patches_equal[:5]]}...")
    
    # Method 2: Grid patches without overlap
    print("\n2. Grid patches without overlap (stride=1.0):")
    patches_grid = divide_into_grid_patches_with_stride(
        floor_points, n_patches_x, n_patches_y, stride_x=1.0, stride_y=1.0
    )
    print(f"   Number of patches: {len(patches_grid)}")
    print(f"   Points per patch: {[len(p) for p in patches_grid[:5]]}...")
    
    # Method 3: Grid patches with 50% overlap
    print("\n3. Grid patches with 50% overlap (stride=0.5):")
    patches_overlap = divide_into_grid_patches_with_stride(
        floor_points, n_patches_x, n_patches_y, stride_x=0.5, stride_y=0.5
    )
    print(f"   Number of patches: {len(patches_overlap)}")
    print(f"   Points per patch: {[len(p) for p in patches_overlap[:5]]}...")
    
    # Method 4: Grid patches with asymmetric overlap
    print("\n4. Grid patches with asymmetric overlap (stride_x=0.7, stride_y=0.3):")
    patches_asym = divide_into_grid_patches_with_stride(
        floor_points, n_patches_x, n_patches_y, stride_x=0.7, stride_y=0.3
    )
    print(f"   Number of patches: {len(patches_asym)}")
    print(f"   Points per patch: {[len(p) for p in patches_asym[:5]]}...")
    
    # Visualize all methods
    print("\n=== Visualizing Results ===")
    
    visualize_patches(patches_equal, n_patches_x, n_patches_y, "traditional_equal", 1.0, 1.0)
    visualize_patches(patches_grid, n_patches_x, n_patches_y, "grid_no_overlap", 1.0, 1.0)
    visualize_patches(patches_overlap, n_patches_x, n_patches_y, "grid_50_overlap", 0.5, 0.5)
    visualize_patches(patches_asym, n_patches_x, n_patches_y, "grid_asymmetric", 0.7, 0.3)
    
    return patches_equal, patches_grid, patches_overlap, patches_asym

def analyze_anomaly_detection_with_stride():
    """Analyze how stride affects anomaly detection performance."""
    print("\n=== Anomaly Detection with Different Stride Values ===")
    
    # Generate synthetic data with known anomalies
    print("Generating synthetic floor with anomalies...")
    floor_points = generate_synthetic_floor_with_bumps(n_points=8000, floor_size=10.0, n_bumps=4)
    
    cfg = Confing()
    n_patches_x, n_patches_y = 6, 6
    
    # Test different stride values
    stride_values = [1.0, 0.8, 0.5, 0.3]
    
    for stride in stride_values:
        print(f"\n--- Testing stride = {stride} ({(1-stride)*100:.0f}% overlap) ---")
        
        # Divide into patches
        patches = divide_into_grid_patches_with_stride(
            floor_points, n_patches_x, n_patches_y, stride_x=stride, stride_y=stride
        )
        
        # Calculate anomaly scores
        magnitudes = calculate_patch_differences_with_ds(patches, cfg, n_patches_x, n_patches_y)
        
        # Statistics
        print(f"   Number of patches: {len(patches)}")
        print(f"   Non-empty patches: {sum(1 for p in patches if len(p) > 0)}")
        print(f"   Average magnitude: {np.mean(magnitudes):.6f}")
        print(f"   Max magnitude: {np.max(magnitudes):.6f}")
        print(f"   Std magnitude: {np.std(magnitudes):.6f}")
        
        # Visualize anomaly detection results
        visualize_patch_differences(
            patches, magnitudes, n_patches_x, n_patches_y, 
            f"stride_{stride:.1f}_anomaly"
        )

def demonstrate_stride_benefits():
    """Demonstrate the benefits of using stride for overlapping patches."""
    print("\n=== Demonstrating Stride Benefits ===")
    
    # Create a simple test case with a single anomaly at patch boundary
    print("Creating test case with anomaly at patch boundary...")
    
    # Generate a simple grid of points
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X) + np.random.normal(0, 0.01, X.shape)
    
    # Add anomaly exactly at the boundary between patches
    boundary_x = 5.0  # Middle of the grid
    boundary_y = 5.0
    anomaly_mask = (np.abs(X - boundary_x) < 0.5) & (np.abs(Y - boundary_y) < 0.5)
    Z[anomaly_mask] += 0.2  # Add bump
    
    # Flatten to point cloud
    points = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
    
    cfg = Confing()
    n_patches_x, n_patches_y = 4, 4
    
    print(f"Total points: {len(points)}")
    print(f"Anomaly location: ({boundary_x}, {boundary_y})")
    
    # Test with no overlap (stride=1.0)
    print("\n1. No overlap (stride=1.0):")
    patches_no_overlap = divide_into_grid_patches_with_stride(
        points, n_patches_x, n_patches_y, stride_x=1.0, stride_y=1.0
    )
    magnitudes_no_overlap = calculate_patch_differences_with_ds(patches_no_overlap, cfg, n_patches_x, n_patches_y)
    print(f"   Max anomaly score: {np.max(magnitudes_no_overlap):.6f}")
    
    # Test with 50% overlap (stride=0.5)
    print("\n2. 50% overlap (stride=0.5):")
    patches_overlap = divide_into_grid_patches_with_stride(
        points, n_patches_x, n_patches_y, stride_x=0.5, stride_y=0.5
    )
    magnitudes_overlap = calculate_patch_differences_with_ds(patches_overlap, cfg, n_patches_x, n_patches_y)
    print(f"   Max anomaly score: {np.max(magnitudes_overlap):.6f}")
    
    # Visualize results
    visualize_patch_differences(
        patches_no_overlap, magnitudes_no_overlap, n_patches_x, n_patches_y,
        "boundary_test_no_overlap"
    )
    visualize_patch_differences(
        patches_overlap, magnitudes_overlap, n_patches_x, n_patches_y,
        "boundary_test_with_overlap"
    )
    
    print(f"\nImprovement with overlap: {(np.max(magnitudes_overlap) / np.max(magnitudes_no_overlap) - 1) * 100:.1f}%")

def main():
    """Main demonstration function."""
    print("Point Cloud Stride-Based Patch Division Demo")
    print("=" * 50)
    
    # Ensure output directory exists
    if not os.path.exists('output'):
        os.makedirs('output')
    
    # Run demonstrations
    try:
        # 1. Compare different patch division methods
        compare_patch_methods()
        
        # 2. Analyze anomaly detection with different stride values
        analyze_anomaly_detection_with_stride()
        
        # 3. Demonstrate stride benefits for boundary anomalies
        demonstrate_stride_benefits()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("Check the 'output' directory for generated visualizations.")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()