# Re-import necessary libraries after environment reset
import numpy as np
import pandas as pd
import open3d as o3d

# Function to generate long floor with many bumps
def generate_long_floor_with_many_bumps(floor_length, floor_width, grid_size, num_bumps, max_bump_height, max_bump_spread, noise_level=0):
    # Generate grid of x, y coordinates
    x_coords = np.arange(0, floor_length, grid_size)
    y_coords = np.arange(0, floor_width, grid_size)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    # Initialize z values as a flat floor
    z_grid = np.zeros_like(x_grid)

    # Generate random positions, heights, and spreads for bumps
    bump_positions = np.random.rand(num_bumps, 2) * [floor_length, floor_width]
    bump_heights = np.random.rand(num_bumps) * max_bump_height
    bump_spreads = np.random.rand(num_bumps) * max_bump_spread

    # Add bumps at random positions
    for pos, height, spread in zip(bump_positions, bump_heights, bump_spreads):
        bump_x, bump_y = pos
        distance_from_bump = np.sqrt((x_grid - bump_x)**2 + (y_grid - bump_y)**2)
        bump = height * np.exp(-distance_from_bump**2 / (2 * spread**2))
        z_grid += bump

    # Add random noise if specified
    if noise_level > 0:
        noise = noise_level * np.random.randn(*z_grid.shape)
        z_grid += noise

    # Flatten the grid into lists of x, y, z coordinates
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    z_flat_with_bumps = z_grid.flatten()

    # Create a DataFrame
    point_cloud_data = pd.DataFrame({
        'x': x_flat,
        'y': y_flat,
        'z': z_flat_with_bumps
    })

    return point_cloud_data

# Parameters for the longer floor
floor_length = 30  # Make the floor longer
floor_width = 20
grid_size = 0.1
num_bumps = 5  # More bumps on the floor
max_bump_height = 2.0  # Maximum height for the bumps
max_bump_spread = 1.0  # Maximum spread (size) for the bumps
noise_level = 0.01  # Optional noise level

# Generate the point cloud for the longer floor with many bumps
long_floor_with_many_bumps = generate_long_floor_with_many_bumps(
    floor_length, floor_width, grid_size, num_bumps, max_bump_height, max_bump_spread, noise_level
)

# Save the new dataset with many bumps to CSV
csv_file_many_bumps_path = './/long_floor_with_many_bumps.csv'
long_floor_with_many_bumps.to_csv(csv_file_many_bumps_path, index=False)

csv_file_many_bumps_path


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(long_floor_with_many_bumps.values)
o3d.visualization.draw_geometries([pcd], window_name="Original Floor Point Cloud")


