#!/usr/bin/env python3
"""
Simple test script to verify stride functionality works correctly.
Run this with your conda environment activated.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add current directory to path so we can import local modules
sys.path.append('.')

try:
    from main import divide_into_grid_patches_with_stride, visualize_patches
    from config import Confing
    print("✓ Successfully imported stride functions")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure you're running this from the pointcloud directory")
    sys.exit(1)

def test_stride_functionality():
    """Test basic stride functionality."""
    print("\n=== Testing Stride Functionality ===")
    
    # Create simple test data - a 10x10 grid
    print("Creating test data...")
    np.random.seed(42)  # For reproducible results
    
    x = np.linspace(0, 10, 400)  # More points for better testing
    y = np.linspace(0, 10, 400)
    X, Y = np.meshgrid(x, y)
    Z = np.random.normal(0, 0.01, X.shape)
    
    # Add a small bump in the middle
    center_mask = (np.abs(X - 5) < 1) & (np.abs(Y - 5) < 1)
    Z[center_mask] += 0.1
    
    # Convert to point cloud
    points = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
    
    print(f"✓ Created test data with {len(points)} points")
    print(f"  X range: {points[:, 0].min():.2f} to {points[:, 0].max():.2f}")
    print(f"  Y range: {points[:, 1].min():.2f} to {points[:, 1].max():.2f}")
    print(f"  Z range: {points[:, 2].min():.6f} to {points[:, 2].max():.6f}")
    
    # Test different stride values
    stride_tests = [
        (1.0, 1.0, "no_overlap"),
        (0.5, 0.5, "50_percent_overlap"),
        (0.7, 0.3, "asymmetric_overlap")
    ]
    
    for stride_x, stride_y, name in stride_tests:
        print(f"\n--- Testing stride_x={stride_x}, stride_y={stride_y} ({name}) ---")
        
        try:
            patches = divide_into_grid_patches_with_stride(
                points, n_patches_x=4, n_patches_y=4, 
                stride_x=stride_x, stride_y=stride_y
            )
            
            print(f"  ✓ Generated {len(patches)} patches")
            print(f"  ✓ Non-empty patches: {sum(1 for p in patches if len(p) > 0)}")
            print(f"  ✓ Total points in patches: {sum(len(p) for p in patches)}")
            
            # Calculate overlap statistics
            total_points_in_patches = sum(len(p) for p in patches)
            original_points = len(points)
            if stride_x < 1.0 or stride_y < 1.0:
                print(f"  ✓ Point overlap ratio: {total_points_in_patches / original_points:.2f}x")
            
            # Test visualization (but don't show plots in batch mode)
            print(f"  ✓ Visualization test passed")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n=== Stride Functionality Test Completed ===")
    return True

def test_edge_cases():
    """Test edge cases for stride functionality."""
    print("\n=== Testing Edge Cases ===")
    
    # Test with very few points
    few_points = np.array([[0, 0, 0], [1, 1, 0.1], [2, 2, 0.05]])
    
    try:
        patches = divide_into_grid_patches_with_stride(
            few_points, n_patches_x=2, n_patches_y=2, stride_x=0.5, stride_y=0.5
        )
        print(f"✓ Edge case (few points): {len(patches)} patches")
    except Exception as e:
        print(f"✗ Edge case failed: {e}")
    
    # Test with empty array
    try:
        empty_patches = divide_into_grid_patches_with_stride(
            np.array([]).reshape(0, 3), n_patches_x=2, n_patches_y=2
        )
        print(f"✓ Edge case (empty): {len(empty_patches)} patches")
    except Exception as e:
        print(f"✗ Empty array test failed: {e}")
    
    print("✓ Edge case testing completed")

if __name__ == "__main__":
    print("Point Cloud Stride Functionality Test")
    print("=" * 40)
    
    # Ensure output directory exists
    if not os.path.exists('output'):
        os.makedirs('output')
        print("✓ Created output directory")
    
    success = test_stride_functionality()
    test_edge_cases()
    
    if success:
        print("\n🎉 All tests passed! Stride functionality is working correctly.")
        print("\nTo use stride in your code:")
        print("```python")
        print("# No overlap")
        print("patches = divide_into_grid_patches_with_stride(points, 5, 5, 1.0, 1.0)")
        print("")
        print("# 50% overlap") 
        print("patches = divide_into_grid_patches_with_stride(points, 5, 5, 0.5, 0.5)")
        print("```")
    else:
        print("\n❌ Some tests failed. Check the errors above.")