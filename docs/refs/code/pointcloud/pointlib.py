# import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

import cv2
import copy
import cupy as cp
import plotly.graph_objs as go


def read_ply(file_path):
    return o3d.io.read_point_cloud(file_path)


def read_mesh(file_path):
    mesh = o3d.io.read_triangle_mesh(file_path)
    mesh.compute_vertex_normals()
    return mesh


def rgb_to_hsv(rgb):
    """
    Convert RGB color to HSV color.

    Parameters:
        rgb (tuple): RGB color represented as a tuple of three integers (R, G, B) in the range [0, 255].

    Returns:
        tuple: HSV color represented as a tuple of three floats (H, S, V) in the ranges [0, 360], [0, 1], [0, 1] respectively.
    """
    # Convert the RGB color to a numpy array of shape (1, 1, 3)
    rgb_np = np.uint8([[rgb]])

    # Convert the RGB color to HSV color
    hsv_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2HSV)

    # Extract the HSV values from the resulting numpy array
    h, s, v = hsv_np[0][0]

    # Scale hue value from [0, 180] to [0, 360]
    h *= 2

    # Normalize saturation and value to the range [0, 1]
    s /= 255.0
    v /= 255.0
    h = h / 360.0
    return (h, s, v)


def get_visualizer(
    point_size=0.01, bg_color=[0.95, 0.95, 0.95], rotation=[0, 0], zoom=1
):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # Set the background color of the visualization window
    vis.get_render_option().background_color = bg_color
    # Set the size of the points in the visualization
    vis.get_render_option().point_size = point_size
    # Set the view control parameters
    if rotation is not None:
        rotation_pixels = [angle * 720.0 / 180.0 for angle in rotation]
        vis.get_view_control().rotate(rotation_pixels[0], rotation_pixels[1])
    vis.get_view_control().set_zoom(zoom)

    return vis


def add_geometries(vis, geometries):
    for geometry in geometries:
        vis.add_geometry(geometry)


def visualize(
    pcd_list,
    point_size=0.01,
    bg_color=[0.95, 0.95, 0.95],
    rotation=[0, 0],
    zoom=1,
    show_normals=False,
):
    """
    Visualize a list of point clouds.

    Parameters:
        pcd_list (list): List of point clouds to visualize.
        point_size (float): Size of the points in the visualization.
        bg_color (list): Background color represented as a list of three floats [R, G, B] in the range [0, 1].
        rotation (list): Rotation represented as a list of three floats [x, y] in degrees.
    """
    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Set the background color of the visualization window
    vis.get_render_option().background_color = bg_color

    # Add the point clouds to the visualization window
    for pcd in pcd_list:
        vis.add_geometry(pcd)

    # Set the size of the points in the visualization
    vis.get_render_option().point_size = point_size
    vis.get_render_option().mesh_show_back_face = True

    # Set the view control parameters
    if rotation is not None:
        rotation_pixels = [angle * 720.0 / 180.0 for angle in rotation]
        vis.get_view_control().rotate(rotation_pixels[0], rotation_pixels[1])
    vis.get_view_control().set_zoom(zoom)

    # Run the visualization
    vis.run()

    # Close the visualization window
    vis.destroy_window()


def visualize_plotly(pcd_list, point_size=0.01, outline=False):
    """
    Visualize a list of point clouds using Plotly.

    Parameters:
        pcd_list (list): List of point clouds to visualize.
        point_size (float): Size of the points in the visualization.
        outline (bool): Whether to show point outlines. Default False.
    """
    # Create a Plotly figure
    fig = go.Figure()

    # Add the point clouds to the Plotly figure
    for pcd in pcd_list:
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]

        marker_dict = {"size": point_size, "color": np.stack([r, g, b], axis=-1)}

        if outline:
            marker_dict["line"] = dict(width=10, color="black")

        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=marker_dict,
                showlegend=False,
            )
        )

    # Update layout to completely remove background and axes
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
    )

    # Show the Plotly figure
    fig.show()


def update_visualization(vis, pcd_list):
    for pcd in pcd_list:
        vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()


def get_cube(num_points=16):
    # Define the colors for each side
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]

    # Define the vertices for each side
    vertices = [
        [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1]],  # Side 1
        [[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],  # Side 2
        [[-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1]],  # Side 3
        [[-1, 1, -1], [1, 1, -1], [1, 1, 1], [-1, 1, 1]],  # Side 4
        [[-1, -1, -1], [-1, 1, -1], [-1, 1, 1], [-1, -1, 1]],  # Side 5
        [[1, -1, -1], [1, 1, -1], [1, 1, 1], [1, -1, 1]],  # Side 6
    ]

    # Create the point cloud
    pcd = o3d.geometry.PointCloud()

    for color, vertex in zip(colors, vertices):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertex)
        mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
        side_pcd = mesh.sample_points_uniformly(number_of_points=num_points)
        side_pcd.colors = o3d.utility.Vector3dVector([color for _ in range(num_points)])
        pcd += side_pcd

    return pcd


def get_cube_grid(num_points_per_side=4):
    # Define the colors for each side

    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]

    # Define the vertices for each side
    vertices = [
        [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1]],  # Side 1
        [[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],  # Side 2
        [[-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1]],  # Side 3
        [[-1, 1, -1], [1, 1, -1], [1, 1, 1], [-1, 1, 1]],  # Side 4
        [[-1, -1, -1], [-1, 1, -1], [-1, 1, 1], [-1, -1, 1]],  # Side 5
        [[1, -1, -1], [1, 1, -1], [1, 1, 1], [1, -1, 1]],  # Side 6
    ]

    # Create the point cloud
    pcd = o3d.geometry.PointCloud()

    for color, vertex in zip(colors, vertices):
        side_points = []
        side_colors = []
        # Calculate the opposite color
        circle_color = [1, 1, 1]
        for i in np.linspace(0, 1, num_points_per_side):
            for j in np.linspace(0, 1, num_points_per_side):
                point = (1 - i) * (
                    (1 - j) * np.array(vertex[0]) + j * np.array(vertex[3])
                ) + i * ((1 - j) * np.array(vertex[1]) + j * np.array(vertex[2]))
                side_points.append(point)
                side_colors.append(color)
        side_pcd = o3d.geometry.PointCloud()
        side_pcd.points = o3d.utility.Vector3dVector(side_points)
        side_pcd.colors = o3d.utility.Vector3dVector(side_colors)
        pcd += side_pcd
    # remove duplicate points
    pcd.remove_duplicated_points()
    return pcd


def get_cube_circle(num_points_per_side=4, circle_radius=0.3, bw=False):
    # Define the colors for each side
    if bw:
        colors = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    else:
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]

    # Define the vertices for each side
    vertices = [
        [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1]],  # Side 1
        [[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],  # Side 2
        [[-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1]],  # Side 3
        [[-1, 1, -1], [1, 1, -1], [1, 1, 1], [-1, 1, 1]],  # Side 4
        [[-1, -1, -1], [-1, 1, -1], [-1, 1, 1], [-1, -1, 1]],  # Side 5
        [[1, -1, -1], [1, 1, -1], [1, 1, 1], [1, -1, 1]],  # Side 6
    ]

    # Create the point cloud
    pcd = o3d.geometry.PointCloud()

    for color, vertex in zip(colors, vertices):
        side_points = []
        side_colors = []
        # Calculate the opposite color
        circle_color = [1, 1, 1]
        for i in np.linspace(0, 1, num_points_per_side):
            for j in np.linspace(0, 1, num_points_per_side):
                point = (1 - i) * (
                    (1 - j) * np.array(vertex[0]) + j * np.array(vertex[3])
                ) + i * ((1 - j) * np.array(vertex[1]) + j * np.array(vertex[2]))
                side_points.append(point)
                # Calculate the center of the side
                center = (np.array(vertex[0]) + np.array(vertex[2])) / 2

                # Calculate the distance from the center of the side
                distance_from_center = np.linalg.norm(point - center)
                # If the distance is less than the radius of the circle, set the color to the circle color
                if distance_from_center <= circle_radius:
                    side_colors.append(circle_color)
                else:
                    side_colors.append(color)
        side_pcd = o3d.geometry.PointCloud()
        side_pcd.points = o3d.utility.Vector3dVector(side_points)
        side_pcd.colors = o3d.utility.Vector3dVector(side_colors)
        pcd += side_pcd
    # remove duplicate points
    pcd.remove_duplicated_points()
    return pcd


def get_cube_normal(num_points_per_side=4):
    # Define the normals for each side
    normals = [
        [0, 0, -1],  # Side 1
        [0, 0, 1],  # Side 2
        [0, -1, 0],  # Side 3
        [0, 1, 0],  # Side 4
        [-1, 0, 0],  # Side 5
        [1, 0, 0],  # Side 6
    ]

    # Define the colors for each side

    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]

    # Define the vertices for each side
    vertices = [
        [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1]],  # Side 1
        [[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],  # Side 2
        [[-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1]],  # Side 3
        [[-1, 1, -1], [1, 1, -1], [1, 1, 1], [-1, 1, 1]],  # Side 4
        [[-1, -1, -1], [-1, 1, -1], [-1, 1, 1], [-1, -1, 1]],  # Side 5
        [[1, -1, -1], [1, 1, -1], [1, 1, 1], [1, -1, 1]],  # Side 6
    ]

    # Create the point cloud
    pcd = o3d.geometry.PointCloud()

    for color, vertex in zip(colors, vertices):
        side_points = []
        side_colors = []
        side_normals = []
        # Calculate the opposite color
        circle_color = [1, 1, 1]
        for i in np.linspace(0, 1, num_points_per_side):
            for j in np.linspace(0, 1, num_points_per_side):
                point = (1 - i) * (
                    (1 - j) * np.array(vertex[0]) + j * np.array(vertex[3])
                ) + i * ((1 - j) * np.array(vertex[1]) + j * np.array(vertex[2]))
                side_points.append(point)
                side_colors.append(color)
                side_normals.append(normals[colors.index(color)])
        side_pcd = o3d.geometry.PointCloud()
        side_pcd.points = o3d.utility.Vector3dVector(side_points)
        side_pcd.colors = o3d.utility.Vector3dVector(side_colors)
        side_pcd.normals = o3d.utility.Vector3dVector(side_normals)
        pcd += side_pcd

    # remove duplicate points
    pcd.remove_duplicated_points()
    return pcd


def get_sphere(radius=1.0, resolution=100, color=[1, 0, 0]):
    # Create the point cloud
    pcd = o3d.geometry.PointCloud()

    # Generate points on the sphere using spherical coordinates
    for theta in np.linspace(0, 2 * np.pi, resolution):
        for phi in np.linspace(0, np.pi, resolution):
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)
            pcd.points.append([x, y, z])
            pcd.colors.append(color)

    return pcd


def get_two_planes(angle_degrees, width=1, height=1, num_points_per_axis=10):
    # Generate points on a plane (grid)
    y_values = np.linspace(0, width, num_points_per_axis)
    z_values = np.linspace(0, height, num_points_per_axis)
    x_values = np.ones(num_points_per_axis) * 0.5

    points = []
    for y in y_values:
        for z in z_values:
            points.append([0.5, y, z])
    points = np.array(points)

    # Create two separate point clouds
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points)

    rotation1 = o3d.geometry.get_rotation_matrix_from_xyz(
        (0, 0, np.deg2rad(angle_degrees / 2))
    )

    rotation2 = o3d.geometry.get_rotation_matrix_from_xyz(
        (0, 0, np.deg2rad(-angle_degrees / 2))
    )

    pcd1.rotate(rotation1)
    pcd2.rotate(rotation2)

    pts = np.asarray(pcd1.points)
    b = pts.min(axis=0)
    pcd1.translate([0.5 - b[0], 0, 0])

    pts = np.asarray(pcd2.points)
    b = pts.min(axis=0)
    pcd2.translate([b[0] - 0.5, 0, 0])

    pts2 = np.asarray(pcd2.points)
    pts2 = pts2[:90, :]
    pcd2.points = o3d.utility.Vector3dVector(pts2)

    pcd1.paint_uniform_color([1, 0, 0])
    pcd2.paint_uniform_color([0, 1, 0])

    # Combine the point clouds
    combined_pcd = pcd1 + pcd2

    return combined_pcd


def clip_points(pcd, center, rect_size, rotation):
    idxs, bbox = get_idx_from_bbox(pcd, center, rect_size, rotation)
    return pcd.select_by_index(idxs)


def get_idx_from_bbox(pcd, center, rect_size, rotation=None, rot_matrix=None):
    center = np.array(center)
    rect_size = np.array(rect_size)
    mode = ""
    if rotation is not None:
        rotation = np.array(rotation)
        mode = "rotation"

    if rot_matrix is not None:
        rot_matrix = np.array(rot_matrix)
        mode = "rot_matrix"

    # Define the width, height, and depth
    width = rect_size[0]
    height = rect_size[1]
    depth = rect_size[2]

    # Calculate the half of the dimensions
    half_width = width / 2
    half_height = height / 2
    half_depth = depth / 2

    # Create the bounding box
    bounding_box = np.array(
        [
            [center[0] - half_width, center[1] - half_height, center[2] - half_depth],
            [center[0] + half_width, center[1] - half_height, center[2] - half_depth],
            [center[0] - half_width, center[1] + half_height, center[2] - half_depth],
            [center[0] + half_width, center[1] + half_height, center[2] - half_depth],
            [center[0] - half_width, center[1] - half_height, center[2] + half_depth],
            [center[0] + half_width, center[1] - half_height, center[2] + half_depth],
            [center[0] - half_width, center[1] + half_height, center[2] + half_depth],
            [center[0] + half_width, center[1] + half_height, center[2] + half_depth],
        ]
    )

    crop_points = o3d.utility.Vector3dVector(bounding_box)
    bbox = o3d.geometry.OrientedBoundingBox.create_from_points(crop_points)

    if mode == "rotation":
        R = o3d.geometry.get_rotation_matrix_from_xyz(rotation)
        bbox.R = R
    elif mode == "rot_matrix":
        bbox.R = rot_matrix

    idxs = bbox.get_point_indices_within_bounding_box(pcd.points)
    # print(f"Number of points in the bounding box: {len(idxs)}")

    return idxs, bbox


def var_points(pcd):
    # Compute variance of the color in grayscale
    if len(pcd.points) == 0:
        return 0

    colors = np.asarray(pcd.colors)
    gray = colors[:, 0]
    res = np.var(gray)
    # print(f"Variance: {res}")
    return res


def mean_points(pcd):
    # Compute mean of the color in grayscale
    # just average of the colors
    if len(pcd.points) == 0:
        return 0

    colors = np.asarray(pcd.colors)
    gray = colors[:, 0]
    res = np.mean(gray)
    # print(f"Mean: {res}")
    return res


def merge_pcds(pcds):
    pcd = pcds[0]
    for i in range(1, len(pcds)):
        pcd += pcds[i]
    return pcd


def add_fake_points(pcd, size):
    fake_pcd = o3d.geometry.PointCloud()
    # Generate random points within the bounding box of pcd_inner
    fake_points = np.random.uniform(0, 1, (int(size), 3))

    # Create a point cloud from the fake points
    fake_pcd.points = o3d.utility.Vector3dVector(fake_points)
    fake_pcd.paint_uniform_color([0, 0, 0])
    # Add the fake points to pcd_inner
    pcd += fake_pcd
    return pcd


def calc_separability(region1, region2):

    num_pixels_region1 = len(region1.points)
    num_pixels_region2 = len(region2.points)

    mean_region1 = mean_points(region1)
    mean_region2 = mean_points(region2)

    two_region = merge_pcds([region1, region2])
    num_pixels_two_region = len(two_region.points)

    mean_two_region = mean_points(two_region)
    var_two_region = var_points(two_region)

    o2b = (
        num_pixels_region1 * (mean_region1 - mean_two_region) ** 2
        + num_pixels_region2 * (mean_region2 - mean_two_region) ** 2
    )

    o2t = num_pixels_two_region * var_two_region

    o2b = round(o2b, 6)
    o2t = round(o2t, 6)

    if o2t == 0:
        sep_val = 0
    else:
        sep_val = o2b / o2t

    return sep_val


def calc_ultrafast_separability_with_bg_points_raw(
    region1_size: int, region2_size: int, bg_size1: int, bg_size2: int
) -> float:
    """
    Calculate the statistical separability between two regions including their background points.
    This function implements a variance-based separability measure similar to Fisher's criterion,
    where higher values indicate better separation between regions.

    The calculation assumes binary data where:
    - Region points have value 1
    - Background points have value 0

    The separability is calculated as the ratio of between-class variance to total variance (o2b/o2t),
    where values closer to 1 indicate better separation between regions.

    Parameters:
    -----------
    region1_size : int
        Number of points in the first region (foreground points)
    region2_size : int
        Number of points in the second region (foreground points)
    bg_size1 : int
        Number of background points associated with region 1
    bg_size2 : int
        Number of background points associated with region 2

    Returns:
    --------
    float
        Separability value between 0 and 1, where:
        - 0 indicates no separation (regions are identical)
        - 1 indicates perfect separation
        - 0 is returned if total variance (o2t) is 0 to avoid division by zero

    Mathematical Components:
    ----------------------
    - Mean calculations:
        * Individual region means (proportion of foreground points)
        * Combined mean across both regions
    - Variance calculations:
        * Individual region variances using binary (0/1) values
        * Combined variance incorporating both within-class and between-class variance
    - Final separability:
        * Ratio of between-class variance (o2b) to total variance (o2t)
    """

    # calculate separability assumed that region values is 1, and bg values is 0
    bg_size1 = int(bg_size1)
    bg_size2 = int(bg_size2)
    object_size1 = int(region1_size)
    object_size2 = int(region2_size)

    num_pixels_region1 = object_size1 + bg_size1
    num_pixels_region2 = object_size2 + bg_size2
    num_pixels_two_region = num_pixels_region1 + num_pixels_region2

    # Calculate the means of each region
    mean_region1 = object_size1 / num_pixels_region1
    mean_region2 = object_size2 / num_pixels_region2

    # Calculate the combined mean
    mean_two_region = (object_size1 + object_size2) / num_pixels_two_region

    # # Calculate the variances of each region
    # var_region1 = mean_region1 * (1 - mean_region1)
    # var_region2 = mean_region2 * (1 - mean_region2)

    # Calculate the combined variance
    # var_two_region = (
    #     (num_pixels_region1 - 1) * var_region1
    #     + (num_pixels_region2 - 1) * var_region2
    #     + num_pixels_region1 * (mean_region1 - mean_two_region) ** 2
    #     + num_pixels_region2 * (mean_region2 - mean_two_region) ** 2
    # ) / (num_pixels_two_region - 1)

    o2b = (
        num_pixels_region1 * (mean_region1 - mean_two_region) ** 2
        + num_pixels_region2 * (mean_region2 - mean_two_region) ** 2
    )

    o2t = num_pixels_two_region * mean_two_region * (1 - mean_two_region)

    o2b = round(o2b, 24)
    o2t = round(o2t, 24)

    if o2t == 0:
        sep_val = 0
    else:
        sep_val = o2b / o2t

    return sep_val


def calc_ultrafast_separability_with_bg_points_o3d(
    region1, region2, bg_size1: int, bg_size2: int
) -> float:
    """
    Calculate the statistical separability between two regions including their background points.
    This function works with region objects that contain points attributes, making it suitable
    for direct use with geometric or image processing region representations.

    The calculation assumes binary data where:
    - Region points have value 1
    - Background points have value 0

    The separability is calculated as the ratio of between-class variance to total variance (o2b/o2t),
    where values closer to 1 indicate better separation between regions.

    Parameters:
    -----------
    region1 : Region
        First region object containing a 'points' attribute that defines the region's points
    region2 : Region
        Second region object containing a 'points' attribute that defines the region's points
    bg_size1 : int
        Number of background points associated with region 1
    bg_size2 : int
        Number of background points associated with region 2

    Returns:
    --------
    float
        Separability value between 0 and 1, where:
        - 0 indicates no separation (regions are identical)
        - 1 indicates perfect separation
        - 0 is returned if total variance (o2t) is 0 to avoid division by zero

    Requirements:
    ------------
    - Input region objects must have a 'points' attribute that can be measured with len()
    - The points attribute should contain the foreground points of the region


    """

    # calculate separability assumed that region values is 1, and bg values is 0
    bg_size1 = int(bg_size1)
    bg_size2 = int(bg_size2)
    object_size1 = len(region1.points)
    object_size2 = len(region2.points)

    num_pixels_region1 = object_size1 + bg_size1
    num_pixels_region2 = object_size2 + bg_size2
    num_pixels_two_region = num_pixels_region1 + num_pixels_region2

    # Calculate the means of each region
    mean_region1 = object_size1 / num_pixels_region1
    mean_region2 = object_size2 / num_pixels_region2

    # Calculate the combined mean
    mean_two_region = (object_size1 + object_size2) / num_pixels_two_region

    # # Calculate the variances of each region
    # var_region1 = mean_region1 * (1 - mean_region1)
    # var_region2 = mean_region2 * (1 - mean_region2)

    # # Calculate the combined variance
    # var_two_region = (
    #     (num_pixels_region1 - 1) * var_region1
    #     + (num_pixels_region2 - 1) * var_region2
    #     + num_pixels_region1 * (mean_region1 - mean_two_region) ** 2
    #     + num_pixels_region2 * (mean_region2 - mean_two_region) ** 2
    # ) / (num_pixels_two_region - 1)

    o2b = (
        num_pixels_region1 * (mean_region1 - mean_two_region) ** 2
        + num_pixels_region2 * (mean_region2 - mean_two_region) ** 2
    )

    o2t = num_pixels_two_region * mean_two_region * (1 - mean_two_region)
    # print(o2b, o2t, round(o2b, 6), round(o2t, 6))
    o2b = round(o2b, 6)
    o2t = round(o2t, 6)

    if o2t == 0:
        sep_val = 0
    else:
        sep_val = o2b / o2t
    # print(sep_val)
    return sep_val

    # calculate separability assumed that region values is 1, and bg values is 0
    bg_size1 = int(bg_size1)
    bg_size2 = int(bg_size2)
    object_size1 = len(region1)
    object_size2 = len(region2)

    num_pixels_region1 = object_size1 + bg_size1
    num_pixels_region2 = object_size2 + bg_size2
    num_pixels_two_region = num_pixels_region1 + num_pixels_region2

    # Calculate the means of each region
    mean_region1 = object_size1 / num_pixels_region1
    mean_region2 = object_size2 / num_pixels_region2

    # Calculate the combined mean
    mean_two_region = (object_size1 + object_size2) / num_pixels_two_region

    # Calculate the variances of each region
    var_region1 = mean_region1 * (1 - mean_region1)
    var_region2 = mean_region2 * (1 - mean_region2)

    # Calculate the combined variance
    var_two_region = (
        (num_pixels_region1 - 1) * var_region1
        + (num_pixels_region2 - 1) * var_region2
        + num_pixels_region1 * (mean_region1 - mean_two_region) ** 2
        + num_pixels_region2 * (mean_region2 - mean_two_region) ** 2
    ) / (num_pixels_two_region - 1)

    o2b = (
        num_pixels_region1 * (mean_region1 - mean_two_region) ** 2
        + num_pixels_region2 * (mean_region2 - mean_two_region) ** 2
    )

    o2t = num_pixels_two_region * var_two_region

    o2b = round(o2b, 6)
    o2t = round(o2t, 6)

    if o2t == 0:
        sep_val = 0
    else:
        sep_val = o2b / o2t

    return sep_val


def gpu_calc_fast_separability_with_bg_points(region1, region2, bg_size1, bg_size2):
    # calculate separability assuming bg values is x1 and x2, but region values is dynamic
    bg_size1 = int(bg_size1)
    bg_size2 = int(bg_size2)

    x1 = 0  # Example value for background points in region 1
    x2 = 0  # Example value for background points in region 2

    object_size1 = len(region1.points)
    object_size2 = len(region2.points)

    if object_size1 == 0:
        mean_object1 = 0
        var_object1 = 1
    else:
        object_points1 = cp.mean(cp.asarray(region1.colors), axis=1)
        mean_object1 = cp.mean(object_points1)
        var_object1 = cp.var(object_points1)

    if object_size2 == 0:
        mean_object2 = 0
        var_object2 = 1
    else:
        object_points2 = cp.mean(cp.asarray(region2.colors), axis=1)
        mean_object2 = cp.mean(object_points2)
        var_object2 = cp.var(object_points2)

    num_pixels_region1 = object_size1 + bg_size1
    num_pixels_region2 = object_size2 + bg_size2
    num_pixels_two_region = num_pixels_region1 + num_pixels_region2

    # Calculate the means of each region
    mean_region1 = (object_size1 * mean_object1 + bg_size1 * x1) / num_pixels_region1
    mean_region2 = (object_size2 * mean_object2 + bg_size2 * x2) / num_pixels_region2

    # Calculate the variances of each region
    var_region1 = (
        (object_size1 * var_object1)
        + (bg_size1 * 0)
        + (bg_size1 * (x1 - mean_region1) ** 2)
        + (object_size1 * (mean_object1 - mean_region1) ** 2)
    ) / num_pixels_region1

    var_region2 = (
        (object_size2 * var_object2)
        + (bg_size2 * 0)
        + (bg_size2 * (x2 - mean_region2) ** 2)
        + (object_size2 * (mean_object2 - mean_region2) ** 2)
    ) / num_pixels_region2

    # Calculate the combined mean
    mean_two_region = (
        num_pixels_region1 * mean_region1 + num_pixels_region2 * mean_region2
    ) / num_pixels_two_region

    # Calculate the combined variance
    var_two_region = (
        (num_pixels_region1 - 1) * var_region1
        + (num_pixels_region2 - 1) * var_region2
        + num_pixels_region1 * (mean_region1 - mean_two_region) ** 2
        + num_pixels_region2 * (mean_region2 - mean_two_region) ** 2
    ) / (num_pixels_two_region - 1)

    o2b = (
        num_pixels_region1 * (mean_region1 - mean_two_region) ** 2
        + num_pixels_region2 * (mean_region2 - mean_two_region) ** 2
    )

    o2t = num_pixels_two_region * var_two_region

    o2b = cp.round(o2b, 12)
    o2t = cp.round(o2t, 12)

    if o2t == 0:
        sep_val = 0
    else:
        sep_val = o2b / o2t

    return sep_val.get()  # Convert CuPy array back to a standard Python float


def calc_fast_separability_with_bg_points(region1, region2, bg_size1, bg_size2):
    # calculate separability assuming bg values is x1 and x2, but region values is dynamic
    bg_size1 = int(bg_size1)
    bg_size2 = int(bg_size2)

    x1 = 0  # Example value for background points in region 1
    x2 = 0  # Example value for background points in region 2

    object_size1 = len(region1.points)
    object_size2 = len(region2.points)

    if object_size1 == 0:
        mean_object1 = 0
        var_object1 = 1
    else:
        object_points1 = np.mean(np.asarray(region1.colors), axis=1)
        mean_object1 = np.mean(object_points1)
        var_object1 = np.var(object_points1)

    if object_size2 == 0:
        mean_object2 = 0
        var_object2 = 1
    else:
        object_points2 = np.mean(np.asarray(region2.colors), axis=1)
        mean_object2 = np.mean(object_points2)
        var_object2 = np.var(object_points2)

    num_pixels_region1 = object_size1 + bg_size1
    num_pixels_region2 = object_size2 + bg_size2
    num_pixels_two_region = num_pixels_region1 + num_pixels_region2

    # Calculate the means of each region
    mean_region1 = (object_size1 * mean_object1 + bg_size1 * x1) / num_pixels_region1
    mean_region2 = (object_size2 * mean_object2 + bg_size2 * x2) / num_pixels_region2

    # Calculate the variances of each region
    var_region1 = (
        (object_size1 * var_object1)
        + (bg_size1 * 0)
        + (bg_size1 * (x1 - mean_region1) ** 2)
        + (object_size1 * (mean_object1 - mean_region1) ** 2)
    ) / num_pixels_region1

    var_region2 = (
        (object_size2 * var_object2)
        + (bg_size2 * 0)
        + (bg_size2 * (x2 - mean_region2) ** 2)
        + (object_size2 * (mean_object2 - mean_region2) ** 2)
    ) / num_pixels_region2

    # Calculate the combined mean
    mean_two_region = (
        num_pixels_region1 * mean_region1 + num_pixels_region2 * mean_region2
    ) / num_pixels_two_region

    # Calculate the combined variance
    var_two_region = (
        (num_pixels_region1 - 1) * var_region1
        + (num_pixels_region2 - 1) * var_region2
        + num_pixels_region1 * (mean_region1 - mean_two_region) ** 2
        + num_pixels_region2 * (mean_region2 - mean_two_region) ** 2
    ) / (num_pixels_two_region - 1)

    o2b = (
        num_pixels_region1 * (mean_region1 - mean_two_region) ** 2
        + num_pixels_region2 * (mean_region2 - mean_two_region) ** 2
    )

    o2t = num_pixels_two_region * var_two_region

    o2b = round(o2b, 12)
    o2t = round(o2t, 12)

    if o2t == 0:
        sep_val = 0
    else:
        sep_val = o2b / o2t

    return sep_val


def calc_dense_separability_with_bg_points(region1, region2, bg_size1, bg_size2):
    # calculate separability assuming bg values is x1 and x2, but region values is dynamic
    bg_size1 = int(bg_size1)
    bg_size2 = int(bg_size2)

    x1 = 0  # Example value for background points in region 1
    x2 = 0  # Example value for background points in region 2

    object_size1 = len(region1)
    object_size2 = len(region2)

    if object_size1 == 0:
        mean_object1 = 0
        var_object1 = 1
    else:
        object_points1 = np.mean(np.asarray(region1))
        mean_object1 = np.mean(object_points1)
        var_object1 = np.var(object_points1)

    if object_size2 == 0:
        mean_object2 = 0
        var_object2 = 1
    else:
        object_points2 = np.mean(np.asarray(region2))
        mean_object2 = np.mean(object_points2)
        var_object2 = np.var(object_points2)

    num_pixels_region1 = object_size1 + bg_size1
    num_pixels_region2 = object_size2 + bg_size2
    num_pixels_two_region = num_pixels_region1 + num_pixels_region2

    # Calculate the means of each region
    mean_region1 = (object_size1 * mean_object1 + bg_size1 * x1) / num_pixels_region1
    mean_region2 = (object_size2 * mean_object2 + bg_size2 * x2) / num_pixels_region2

    # Calculate the variances of each region
    var_region1 = (
        (object_size1 * var_object1)
        + (bg_size1 * 0)
        + (bg_size1 * (x1 - mean_region1) ** 2)
        + (object_size1 * (mean_object1 - mean_region1) ** 2)
    ) / num_pixels_region1

    var_region2 = (
        (object_size2 * var_object2)
        + (bg_size2 * 0)
        + (bg_size2 * (x2 - mean_region2) ** 2)
        + (object_size2 * (mean_object2 - mean_region2) ** 2)
    ) / num_pixels_region2

    # Calculate the combined mean
    mean_two_region = (
        num_pixels_region1 * mean_region1 + num_pixels_region2 * mean_region2
    ) / num_pixels_two_region

    # Calculate the combined variance
    var_two_region = (
        (num_pixels_region1 - 1) * var_region1
        + (num_pixels_region2 - 1) * var_region2
        + num_pixels_region1 * (mean_region1 - mean_two_region) ** 2
        + num_pixels_region2 * (mean_region2 - mean_two_region) ** 2
    ) / (num_pixels_two_region - 1)

    o2b = (
        num_pixels_region1 * (mean_region1 - mean_two_region) ** 2
        + num_pixels_region2 * (mean_region2 - mean_two_region) ** 2
    )

    o2t = num_pixels_two_region * var_two_region

    o2b = round(o2b, 12)
    o2t = round(o2t, 12)

    if o2t == 0:
        sep_val = 0
    else:
        sep_val = o2b / o2t

    return sep_val


def calc_separability_with_bg_points(region1, region2, bg_size1, bg_size2):
    bg_size1 = int(bg_size1)
    bg_size2 = int(bg_size2)
    object_size1 = len(region1.points)
    object_size2 = len(region2.points)
    num_pixels_region1 = object_size1 + bg_size1
    num_pixels_region2 = object_size2 + bg_size2
    num_pixels_two_region = num_pixels_region1 + num_pixels_region2

    # Generate an array of zeros for the bg points and ones for the object points
    bg_points1 = np.zeros(bg_size1)
    bg_points2 = np.zeros(bg_size2)

    ob_points1 = np.mean(np.asarray(region1.colors), axis=1)
    ob_points2 = np.mean(np.asarray(region2.colors), axis=1)

    # Concatenate the real points and the bg points
    all_points_region1 = np.concatenate((ob_points1, bg_points1))
    all_points_region2 = np.concatenate((ob_points2, bg_points2))
    all_points_two_region = np.concatenate((all_points_region1, all_points_region2))

    # Calculate the mean considering the bg points
    mean_region1 = np.mean(all_points_region1)
    mean_region2 = np.mean(all_points_region2)
    mean_two_region = np.mean(all_points_two_region)

    # Calculate the variance using numpy
    var_two_region = np.var(all_points_two_region)

    o2b = (
        num_pixels_region1 * (mean_region1 - mean_two_region) ** 2
        + num_pixels_region2 * (mean_region2 - mean_two_region) ** 2
    )

    o2t = num_pixels_two_region * var_two_region
    o2b = round(o2b, 12)
    o2t = round(o2t, 12)

    if o2t == 0:
        sep_val = 0
    else:
        sep_val = o2b / o2t

    return sep_val


def calc_ultrafast_separability3_with_bg_points_o3d(
    region1, region2, region3, bg_size1: int, bg_size2: int, bg_size3: int
) -> float:
    """
    Calculate the statistical separability between three regions including their background points.
    This function implements the same direct calculation approach as the two-region version,
    extended to handle three regions.

    The calculation assumes binary data where:
    - Region points have value 1
    - Background points have value 0

    Parameters:
    -----------
    region1, region2, region3 : Region
        Region objects containing 'points' attributes that define each region's points
    bg_size1, bg_size2, bg_size3 : int
        Number of background points associated with each region

    Returns:
    --------
    float
        Separability value between 0 and 1, where:
        - 0 indicates no separation (regions are identical)
        - 1 indicates perfect separation
        - 0 is returned if total variance (o2t) is 0 to avoid division by zero
    """
    # Convert background sizes to integers
    bg_size1 = int(bg_size1)
    bg_size2 = int(bg_size2)
    bg_size3 = int(bg_size3)

    # Get the size of each region from their points attributes
    object_size1 = len(region1.points)
    object_size2 = len(region2.points)
    object_size3 = len(region3.points)

    # Calculate total pixels in each region (foreground + background)
    num_pixels_region1 = object_size1 + bg_size1
    num_pixels_region2 = object_size2 + bg_size2
    num_pixels_region3 = object_size3 + bg_size3
    num_pixels_three_region = (
        num_pixels_region1 + num_pixels_region2 + num_pixels_region3
    )

    # Calculate mean proportion of foreground points in each region
    mean_region1 = object_size1 / num_pixels_region1
    mean_region2 = object_size2 / num_pixels_region2
    mean_region3 = object_size3 / num_pixels_region3

    # Calculate combined mean across all three regions
    mean_three_region = (
        object_size1 + object_size2 + object_size3
    ) / num_pixels_three_region

    # Calculate variance within each region (using binary 0/1 values)
    var_region1 = mean_region1 * (1 - mean_region1)
    var_region2 = mean_region2 * (1 - mean_region2)
    var_region3 = mean_region3 * (1 - mean_region3)

    # Calculate combined variance (pooled variance)
    var_three_region = (
        (num_pixels_region1 - 1) * var_region1
        + (num_pixels_region2 - 1) * var_region2
        + (num_pixels_region3 - 1) * var_region3
        + num_pixels_region1 * (mean_region1 - mean_three_region) ** 2
        + num_pixels_region2 * (mean_region2 - mean_three_region) ** 2
        + num_pixels_region3 * (mean_region3 - mean_three_region) ** 2
    ) / (num_pixels_three_region - 1)

    # Calculate between-class variance (o2b)
    o2b = (
        num_pixels_region1 * (mean_region1 - mean_three_region) ** 2
        + num_pixels_region2 * (mean_region2 - mean_three_region) ** 2
        + num_pixels_region3 * (mean_region3 - mean_three_region) ** 2
    )

    # Calculate total variance (o2t)
    o2t = num_pixels_three_region * var_three_region

    # Round values to avoid floating point precision issues
    o2b = round(o2b, 6)
    o2t = round(o2t, 6)

    # Calculate final separability value
    if o2t == 0:
        sep_val = 0
    else:
        sep_val = o2b / o2t

    return sep_val


def calc_ultrafast_separability3_with_bg_points_raw(
    region1_size: int,
    region2_size: int,
    region3_size: int,
    bg_size1: int,
    bg_size2: int,
    bg_size3: int,
) -> float:
    """
    Calculate the statistical separability between three regions using raw size values.
    This is the generalized version that works with integer inputs directly instead of region objects.

    Parameters:
    -----------
    region1_size, region2_size, region3_size : int
        Number of points in each region (foreground points)
    bg_size1, bg_size2, bg_size3 : int
        Number of background points associated with each region

    Returns:
    --------
    float
        Separability value between 0 and 1, where:
        - 0 indicates no separation (regions are identical)
        - 1 indicates perfect separation
        - 0 is returned if total variance (o2t) is 0 to avoid division by zero
    """
    # Convert all inputs to integers
    bg_size1 = int(bg_size1)
    bg_size2 = int(bg_size2)
    bg_size3 = int(bg_size3)
    object_size1 = int(region1_size)
    object_size2 = int(region2_size)
    object_size3 = int(region3_size)

    # Calculate total pixels in each region (foreground + background)
    num_pixels_region1 = object_size1 + bg_size1
    num_pixels_region2 = object_size2 + bg_size2
    num_pixels_region3 = object_size3 + bg_size3
    num_pixels_three_region = (
        num_pixels_region1 + num_pixels_region2 + num_pixels_region3
    )

    # Calculate mean proportion of foreground points in each region
    mean_region1 = object_size1 / num_pixels_region1
    mean_region2 = object_size2 / num_pixels_region2
    mean_region3 = object_size3 / num_pixels_region3

    # Calculate combined mean across all three regions
    mean_three_region = (
        object_size1 + object_size2 + object_size3
    ) / num_pixels_three_region

    # Calculate variance within each region (using binary 0/1 values)
    var_region1 = mean_region1 * (1 - mean_region1)
    var_region2 = mean_region2 * (1 - mean_region2)
    var_region3 = mean_region3 * (1 - mean_region3)

    # Calculate combined variance (pooled variance)
    var_three_region = (
        (num_pixels_region1 - 1) * var_region1
        + (num_pixels_region2 - 1) * var_region2
        + (num_pixels_region3 - 1) * var_region3
        + num_pixels_region1 * (mean_region1 - mean_three_region) ** 2
        + num_pixels_region2 * (mean_region2 - mean_three_region) ** 2
        + num_pixels_region3 * (mean_region3 - mean_three_region) ** 2
    ) / (num_pixels_three_region - 1)

    # Calculate between-class variance (o2b)
    o2b = (
        num_pixels_region1 * (mean_region1 - mean_three_region) ** 2
        + num_pixels_region2 * (mean_region2 - mean_three_region) ** 2
        + num_pixels_region3 * (mean_region3 - mean_three_region) ** 2
    )

    # Calculate total variance (o2t)
    o2t = num_pixels_three_region * var_three_region

    # Round values to avoid floating point precision issues
    o2b = round(o2b, 24)
    o2t = round(o2t, 24)

    # Calculate final separability value
    if o2t == 0:
        sep_val = 0
    else:
        sep_val = o2b / o2t

    return sep_val


def calc_separability3_with_bg_points(
    region1, region2, region3, bg_size1, bg_size2, bg_size3
):
    bg_size1 = int(bg_size1)
    bg_size2 = int(bg_size2)
    bg_size3 = int(bg_size3)
    object_size1 = len(region1.points)
    object_size2 = len(region2.points)
    object_size3 = len(region3.points)
    num_pixels_region1 = object_size1 + bg_size1
    num_pixels_region2 = object_size2 + bg_size2
    num_pixels_region3 = object_size3 + bg_size3
    num_pixels_three_region = (
        num_pixels_region1 + num_pixels_region2 + num_pixels_region3
    )

    # Generate an array of zeros for the bg points and ones for the object points
    bg_points1 = np.zeros(bg_size1)
    bg_points2 = np.zeros(bg_size2)
    bg_points3 = np.zeros(bg_size3)

    ob_points1 = np.ones(object_size1)
    ob_points2 = np.ones(object_size2)
    ob_points3 = np.ones(object_size3)

    # Concatenate the real points and the bg points
    all_points_region1 = np.concatenate((ob_points1, bg_points1))
    all_points_region2 = np.concatenate((ob_points2, bg_points2))
    all_points_region3 = np.concatenate((ob_points3, bg_points3))

    all_points_three_region = np.concatenate(
        (all_points_region1, all_points_region2, all_points_region3)
    )

    # Calculate the mean considering the bg points
    mean_region1 = np.mean(all_points_region1)
    mean_region2 = np.mean(all_points_region2)
    mean_region3 = np.mean(all_points_region3)
    mean_three_region = np.mean(all_points_three_region)

    # Calculate the variance using numpy
    var_three_region = np.var(all_points_three_region)

    o2b = (
        num_pixels_region1 * (mean_region1 - mean_three_region) ** 2
        + num_pixels_region2 * (mean_region2 - mean_three_region) ** 2
        + num_pixels_region3 * (mean_region3 - mean_three_region) ** 2
    )

    o2t = num_pixels_three_region * var_three_region
    o2b = round(o2b, 6)
    o2t = round(o2t, 6)

    if o2t == 0:
        sep_val = 0
    else:
        sep_val = o2b / o2t

    return sep_val


def calc_separability3(region1, region2, region3):
    num_pixels_region1 = len(region1.points)
    num_pixels_region2 = len(region2.points)
    num_pixels_region3 = len(region3.points)

    mean_region1 = mean_points(region1)
    mean_region2 = mean_points(region2)
    mean_region3 = mean_points(region3)

    three_region = merge_pcds([region1, region2, region3])
    num_pixels_three_region = len(three_region.points)

    mean_three_region = mean_points(three_region)
    var_three_region = var_points(three_region)

    o2b = (
        num_pixels_region1 * (mean_region1 - mean_three_region) ** 2
        + num_pixels_region2 * (mean_region2 - mean_three_region) ** 2
        + num_pixels_region3 * (mean_region3 - mean_three_region) ** 2
    )

    o2t = num_pixels_three_region * var_three_region

    if o2t == 0:
        sep_val = 0
    else:
        sep_val = o2b / o2t

    return sep_val


def normalize(vector):
    """Normalize a vector to unit length."""
    return vector / np.linalg.norm(vector)


def normal_to_rotation_angles(normal):
    """Compute rotation angles around x, y, and z axes from a normal vector."""
    # Step 1: Normalize the normal vector
    n = normalize(normal)

    # Step 2: Compute rotation angles
    # Rotation around x-axis (pitch)
    x_angle = np.arcsin(n[1])
    # Rotation around y-axis (yaw)
    y_angle = np.arctan2(-n[0], n[2])
    # Rotation around z-axis (roll)
    z_angle = 0  # No roll angle for a normal vector

    return np.degrees(x_angle), np.degrees(y_angle), np.degrees(z_angle)


def rotation_matrix_from_normal(normal):
    """Compute a rotation matrix from a normal vector."""
    # Step 1: Normalize the normal vector
    n = normalize(normal)

    # Step 2: Find two orthogonal vectors
    # First, find a vector not parallel to the normal
    v = np.array([1, 0, 0])
    if np.allclose(v, n):
        v = np.array([0, 1, 0])
    # Calculate the first orthogonal vector
    u = np.cross(n, v)
    u = normalize(u)
    # Calculate the second orthogonal vector
    v = np.cross(u, n)

    # Step 3: Form the rotation matrix
    rotation_matrix = np.array([u, v, n])
    new_rot = o3d.geometry.get_rotation_matrix_from_axis_angle([0, np.radians(90), 0])
    rotation_matrix = np.dot(rotation_matrix.T, new_rot)

    return rotation_matrix


def get_rotation_matrix_from_angles(angles):
    """Compute a rotation matrix from rotation angles."""
    # Step 1: Convert angles from degrees to radians
    x, y, z = np.radians(angles)

    # Step 2: Compute the rotation matrix
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(x), -np.sin(x)],
            [0, np.sin(x), np.cos(x)],
        ]
    )
    Ry = np.array(
        [
            [np.cos(y), 0, np.sin(y)],
            [0, 1, 0],
            [-np.sin(y), 0, np.cos(y)],
        ]
    )
    Rz = np.array(
        [
            [np.cos(z), -np.sin(z), 0],
            [np.sin(z), np.cos(z), 0],
            [0, 0, 1],
        ]
    )

    return np.dot(Rz, np.dot(Ry, Rx))


def get_separability(pcd, center, rect_size, rotation=None, rot_matrix=None, vis=False):
    visualization = vis
    step = 10

    if rotation is not None:
        rotation = np.array(rotation)
        rot_center = clipped.get_rotation_matrix_from_xyz(rotation)
    elif rot_matrix is not None:
        rot_center = rot_matrix

    clipped_idx, bbox = get_idx_from_bbox(pcd, center, rect_size, rot_matrix=rot_center)
    clipped = pcd.select_by_index(clipped_idx)
    clipped = clipped.rotate(rot_center.T, (0, 0, 0))
    bbox = bbox.rotate(rot_center.T, (0, 0, 0))
    seps = []
    for i in range(1, step):

        rect_size1 = [rect_size[0] * i / step, rect_size[1], rect_size[2]]
        idxs1, bbox1 = get_idx_from_bbox(
            clipped, [-rect_size[0] / 2 + rect_size1[0] / 2, 0, 0], rect_size1, None
        )

        rect_size2 = [
            rect_size[0] * (1 - i / step),
            rect_size[1],
            rect_size[2],
        ]
        idxs2, bbox2 = get_idx_from_bbox(
            clipped, [rect_size[0] / 2 - rect_size2[0] / 2, 0, 0], rect_size2, None
        )

        bbox1.color = (1, 0, 0)
        bbox2.color = (0, 1, 0)
        bbox.color = (0, 0, 1)

        bbox1 = bbox1.translate(bbox.get_center())
        bbox2 = bbox2.translate(bbox.get_center())

        idxs1 = bbox1.get_point_indices_within_bounding_box(clipped.points)
        idxs2 = bbox2.get_point_indices_within_bounding_box(clipped.points)

        region1 = clipped.select_by_index(idxs1)
        region2 = clipped.select_by_index(idxs2)
        # bbox1.R = bbox.R
        # bbox2.R = bbox.R

        if visualization:
            o3d.visualization.draw_geometries([clipped, bbox1, bbox2, bbox])

        # split to two regions
        sep = calc_separability(region1, region2)
        seps.append(sep)
    return seps


def get_edge_region(pcd, center, rect_size, step=11, rot_matrix=None, vis=False):
    x, y, s = center
    idxs_edge, bbox_edge = get_idx_from_bbox(
        pcd,
        [x, y, s],
        [rect_size[0] / step, rect_size[1], rect_size[2]],
        rot_matrix=rot_matrix,
    )

    edge_rot = get_rotation_matrix_from_angles([0, 90, 0])
    bbox_edge.R = bbox_edge.R @ edge_rot
    # visualise the bounding box
    bbox_edge.color = (0, 1, 0)

    return pcd.select_by_index(idxs_edge), bbox_edge

    edge_rect = [rect_size[0] / step, rect_size[1], rect_size[2]]
    color = (0, 1, 0)

    pcd_edge, bbox_edge = get_custom_region(
        pcd,
        center,
        edge_rect,
        rot_matrix=rot_matrix,
        bbox_color=color,
    )

    return pcd_edge, bbox_edge


def get_inner_region(
    pcd, center, rect_size, step=11, rot_matrix=None, normal=None, vis=False
):
    inner_rect = [rect_size[0] / step * (step - 1) / 2, rect_size[1], rect_size[2]]
    translate = (normal * rect_size[0] / step) * (0.5 + (step - 1) / 4)
    color = (0, 0, 1)

    pcd_inner, bbox_inner = get_custom_region(
        pcd,
        center,
        inner_rect,
        rot_matrix=rot_matrix,
        translate=translate,
        bbox_color=color,
    )

    return pcd_inner, bbox_inner


def get_outter_region(
    pcd, center, rect_size, step=11, rot_matrix=None, normal=None, vis=False
):
    outter_rect = [rect_size[0] / step * (step - 1) / 2, rect_size[1], rect_size[2]]
    translate = (normal * rect_size[0] / step) * -(0.5 + (step - 1) / 4)
    color = (1, 0, 0)

    pcd_outter, bbox_outter = get_custom_region(
        pcd,
        center,
        outter_rect,
        rot_matrix=rot_matrix,
        translate=translate,
        bbox_color=color,
    )

    return pcd_outter, bbox_outter


def get_custom_region(
    pcd,
    center,
    rect_size,
    rot_matrix=None,
    translate=[0, 0, 0],
    bbox_color=(0, 0, 0),
):
    x, y, s = center
    idxs_custom, bbox_custom = get_idx_from_bbox(
        pcd,
        [x, y, s],
        [rect_size[0], rect_size[1], rect_size[2]],
        rot_matrix=rot_matrix,
    )
    bbox_custom.color = bbox_color
    bbox_custom.translate(translate)

    return pcd.select_by_index(idxs_custom), bbox_custom


def get_cuboid_search(
    idx,
    steps,
    center,
    rect_size,
    rot_matrix=None,
):
    search_range = rect_size[0]
    move_step = search_range / steps

    w_left = idx * move_step
    w_right = search_range - w_left

    t_left = w_right / 2
    t_right = w_left / 2

    box_left = o3d.geometry.TriangleMesh.create_box(
        width=w_left, height=rect_size[1], depth=rect_size[2]
    )
    # box_left.translate(-box_left.get_center() + center)
    box_left.translate(-box_left.get_center())
    box_left.translate([-t_left, 0, 0])
    box_left.paint_uniform_color([1, 0, 0])

    box_right = o3d.geometry.TriangleMesh.create_box(
        width=w_right, height=rect_size[1], depth=rect_size[2]
    )
    # box_right.translate(-box_right.get_center() + center)
    box_right.translate(-box_right.get_center())
    box_right.translate([t_right, 0, 0])
    box_right.paint_uniform_color([0, 1, 0])

    merged = o3d.geometry.PointCloud()

    for p in np.asarray(box_left.vertices):
        merged.points.append(p)
    for p in np.asarray(box_right.vertices):
        merged.points.append(p)

    # we dont translate or rotate here,
    # we translate on the object
    # merged.translate(-merged.get_center() + center)
    # if rot_matrix is not None:
    #     merged.rotate(
    #         rot_matrix,
    #         center=center,
    #     )

    # return 8 points of the cuboid
    left_pts = np.asarray(merged.points)[:8]
    right_pts = np.asarray(merged.points)[8:]
    return left_pts, right_pts


def get_cuboid_search_3(
    idx,
    steps,
    center,
    rect_size,
    rot_matrix=None,
):
    search_range = rect_size[0]
    move_step = search_range / steps

    w_left = idx * move_step
    w_right = search_range - w_left

    t_left = (w_right / 2) + (move_step / 2)
    t_right = (w_left / 2) + (move_step / 2)
    t_mid = t_left - t_right

    box_left = o3d.geometry.TriangleMesh.create_box(
        width=w_left, height=rect_size[1], depth=rect_size[2]
    )
    # box_left.translate(-box_left.get_center() + center)
    box_left.translate(-box_left.get_center())
    box_left.translate([-t_left, 0, 0])
    box_left.paint_uniform_color([1, 0, 0])

    box_mid = o3d.geometry.TriangleMesh.create_box(
        width=move_step, height=rect_size[1], depth=rect_size[2]
    )
    box_mid.translate(-box_mid.get_center())
    box_mid.translate([-t_mid, 0, 0])
    box_mid.paint_uniform_color([1, 1, 0])

    box_right = o3d.geometry.TriangleMesh.create_box(
        width=w_right, height=rect_size[1], depth=rect_size[2]
    )
    # box_right.translate(-box_right.get_center() + center)
    box_right.translate(-box_right.get_center())
    box_right.translate([t_right, 0, 0])
    box_right.paint_uniform_color([0, 1, 0])

    merged = o3d.geometry.PointCloud()

    for p in np.asarray(box_left.vertices):
        merged.points.append(p)
    for p in np.asarray(box_mid.vertices):
        merged.points.append(p)
    for p in np.asarray(box_right.vertices):
        merged.points.append(p)

    # we dont translate or rotate here,
    # we translate on the object
    # merged.translate(-merged.get_center() + center)
    # if rot_matrix is not None:
    #     merged.rotate(
    #         rot_matrix,
    #         center=center,
    #     )

    # return 8 points of the cuboid
    left_pts = np.asarray(merged.points)[:8]
    mid_pts = np.asarray(merged.points)[8:16]
    right_pts = np.asarray(merged.points)[16:]
    return left_pts, mid_pts, right_pts


def get_bbox(points, color=[1, 0, 0]):
    # points = fill_points_on_bbox_edges(points)
    pts_pcd = o3d.geometry.PointCloud()
    pts_pcd.points = o3d.utility.Vector3dVector(points)
    # bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
    #     o3d.utility.Vector3dVector(points)
    # )
    bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(points)
    )
    # bbox = pts_pcd.get_minimal_oriented_bounding_box()
    bbox.color = color
    return bbox, pts_pcd


def zero_rot_matrix():
    return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def fill_points_on_bbox_edges(points, num_points_per_edge=5):
    """Fill more points along the edges of a bounding box.
    :param points: An array of 8 points defining the bounding box.
    :param num_points_per_edge: The number of points to add to each edge.
    :return: An array of points including the original points and the new points on the edges.
    """
    # Define the pairs of points that form the edges
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    # Initialize the array of all points
    all_points = list(points)

    # Loop over the edges
    for i, j in edges:
        # Interpolate between the two points that form the edge
        for t in np.linspace(0, 1, num_points_per_edge):
            point_on_edge = (1 - t) * points[i] + t * points[j]
            all_points.append(point_on_edge)

    return np.array(all_points)


def get_rotation_matrix_from_vectors(original, target):
    original = normalize(original)
    target = normalize(target)

    dot_product = np.dot(original, target)

    if np.allclose(dot_product, 1):
        # The vectors are parallel
        return np.eye(3)
    elif np.allclose(dot_product, -1):
        # The vectors are anti-parallel
        return -np.eye(3)
    else:
        v = np.cross(original, target)
        c = dot_product
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + np.dot(kmat, kmat) * (1 - c) / (s**2)
        return rotation_matrix


def get_two_planes(angle_degrees, width=1, height=1, num_points_per_axis=10):
    # Generate points on a plane (grid)
    y_values = np.linspace(0, width, num_points_per_axis)
    z_values = np.linspace(0, height, num_points_per_axis)
    x_values = np.ones(num_points_per_axis) * 0.5

    points = []
    for y in y_values:
        for z in z_values:
            points.append([0.5, y, z])
    points = np.array(points)

    # Create two separate point clouds
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points)

    rotation1 = o3d.geometry.get_rotation_matrix_from_xyz(
        (0, 0, np.deg2rad(angle_degrees / 2))
    )

    rotation2 = o3d.geometry.get_rotation_matrix_from_xyz(
        (0, 0, np.deg2rad(-angle_degrees / 2))
    )

    pcd1.rotate(rotation1)
    pcd2.rotate(rotation2)

    pts = np.asarray(pcd1.points)
    b = pts.min(axis=0)
    pcd1.translate([0.5 - b[0], 0, 0])

    pts = np.asarray(pcd2.points)
    b = pts.min(axis=0)
    pcd2.translate([b[0] - 0.5, 0, 0])

    pts2 = np.asarray(pcd2.points)
    pts2 = pts2[: num_points_per_axis * (num_points_per_axis - 1), :]
    # reverse pts2
    pts2 = pts2[::-1, :]
    pcd2.points = o3d.utility.Vector3dVector(pts2)

    pcd1.paint_uniform_color([1, 0, 0])
    pcd2.paint_uniform_color([0, 1, 0])

    # Combine the point clouds
    combined_pcd = pcd1 + pcd2
    combined_pcd.rotate(
        o3d.geometry.get_rotation_matrix_from_xyz((0, np.deg2rad(-90), 0))
    )
    return combined_pcd


def apply_noise(pcd, mu=0, sigma=0.1):
    noisy_pcd = copy.deepcopy(pcd)
    points = np.asarray(noisy_pcd.points)
    points += np.random.normal(mu, sigma, size=points.shape)
    noisy_pcd.points = o3d.utility.Vector3dVector(points)
    return noisy_pcd


def add_noise(pcd, mu=0, sigma=0.1, downsample=1, color=None, color_bias=0):
    noisy_pcd = copy.deepcopy(pcd)
    noisy_pcd = noisy_pcd.uniform_down_sample(downsample)
    points = np.asarray(noisy_pcd.points)
    points += np.random.normal(mu, sigma, size=points.shape)
    noisy_pcd.points = o3d.utility.Vector3dVector(points)
    # color with random color
    # Generate a random color for each point
    if color is None:
        colors = np.random.rand(len(points), 3)
    else:
        colors = np.ones((len(points), 3)) * color

    # Add a bias to the color
    if color_bias != 0:
        colors -= color_bias
        colors = np.clip(colors, 0, 1)

    noisy_pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd + noisy_pcd


def add_outlier(pcd, mu=0, sigma=0.1, percentage=0.5):
    outlier_pcd = copy.deepcopy(pcd)
    num_outliers = int(len(outlier_pcd.points) * percentage)
    outlier_indices = np.random.choice(
        len(outlier_pcd.points), num_outliers, replace=False
    )
    points = np.asarray(outlier_pcd.points)
    points[outlier_indices] += np.random.normal(
        mu, sigma, size=points[outlier_indices].shape
    )
    outlier_pcd.points = o3d.utility.Vector3dVector(points)

    # color with random color
    # Generate a random color for each point
    colors = np.random.rand(
        len(points), 3
    )  # Generates an array of shape (N, 3) with random colors
    outlier_pcd.colors = o3d.utility.Vector3dVector(colors)
    return outlier_pcd


def get_cube_pcd(num_points):
    cube = o3d.geometry.TriangleMesh.create_box(width=3, height=3, depth=3)
    cube_pcd = cube.sample_points_poisson_disk(
        number_of_points=num_points, init_factor=5
    )
    cube_pcd.paint_uniform_color([0, 0, 0])
    # Translate the cube point cloud to the center
    cube_center = np.mean(np.asarray(cube_pcd.points), axis=0)
    cube_pcd.points = o3d.utility.Vector3dVector(
        np.asarray(cube_pcd.points) - cube_center
    )
    return cube_pcd


def get_sphere_pcd(num_points, radius=3):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere_pcd = sphere.sample_points_poisson_disk(
        number_of_points=num_points, init_factor=5
    )
    sphere_pcd.paint_uniform_color([0, 0, 0])
    # Translate the sphere point cloud to the center
    sphere_center = np.mean(np.asarray(sphere_pcd.points), axis=0)
    sphere_pcd.points = o3d.utility.Vector3dVector(
        np.asarray(sphere_pcd.points) - sphere_center
    )
    return sphere_pcd


def get_cone_pcd(num_points):
    cone = o3d.geometry.TriangleMesh.create_cone(radius=3, height=5)
    cone_pcd = cone.sample_points_poisson_disk(
        number_of_points=num_points, init_factor=5
    )
    cone_pcd.paint_uniform_color([0, 0, 0])
    # Translate the cone point cloud to the center
    cone_center = np.mean(np.asarray(cone_pcd.points), axis=0)
    cone_pcd.points = o3d.utility.Vector3dVector(
        np.asarray(cone_pcd.points) - cone_center
    )
    return cone_pcd


def get_rect_pcd(num_points):
    rect = o3d.geometry.TriangleMesh.create_box(width=5, height=3, depth=1)
    rect_pcd = rect.sample_points_poisson_disk(
        number_of_points=num_points, init_factor=5
    )
    rect_pcd.paint_uniform_color([0, 0, 0])
    # Translate the rectangle point cloud to the center
    rect_center = np.mean(np.asarray(rect_pcd.points), axis=0)
    rect_pcd.points = o3d.utility.Vector3dVector(
        np.asarray(rect_pcd.points) - rect_center
    )
    return rect_pcd


def get_mesh_from_surface_pcd(dot, div_u, div_v):
    # Create a grid of indices
    indices = np.arange(div_u * div_v).reshape(div_u, div_v)

    # Create a list of lines
    lines = []
    for i in range(div_u):
        for j in range(div_v - 1):
            lines.append([indices[i, j], indices[i, j + 1]])
    for j in range(div_v):
        for i in range(div_u - 1):
            lines.append([indices[i, j], indices[i + 1, j]])

    # Create a line set from the point cloud and the lines
    line_set = o3d.geometry.LineSet(
        points=dot.points,
        lines=o3d.utility.Vector2iVector(lines),
    )

    # Visualize the line set
    # ptl.visualize([dot, line_set], point_size=10)

    # Create a list of triangles
    triangles = []
    for i in range(div_u - 1):
        for j in range(div_v - 1):
            # triangles.append([indices[i, j], indices[i, j + 1], indices[i + 1, j]])
            # triangles.append([indices[i + 1, j], indices[i, j + 1], indices[i + 1, j + 1]])

            triangles.append([indices[i, j], indices[i + 1, j], indices[i + 1, j + 1]])
            triangles.append([indices[i, j], indices[i + 1, j + 1], indices[i, j + 1]])

        # Connect the first and last triangles along the div_v direction
        # triangles.append([indices[i, 0], indices[i + 1, 0], indices[i + 1, div_v - 1]])
        # triangles.append([indices[i, 0], indices[i + 1, div_v - 1], indices[i, div_v - 1]])

    # Create a triangle mesh from the point cloud and the triangles
    mesh = o3d.geometry.TriangleMesh(
        vertices=dot.points,
        triangles=o3d.utility.Vector3iVector(triangles),
    )

    return mesh


def get_lineset_from_surface_pcd(dot, div_u, div_v):
    # Create a grid of indices
    indices = np.arange(div_u * div_v).reshape(div_u, div_v)

    # Create a list of lines
    lines = []
    for i in range(div_u):
        for j in range(div_v - 1):
            lines.append([indices[i, j], indices[i, j + 1]])
    for j in range(div_v):
        for i in range(div_u - 1):
            lines.append([indices[i, j], indices[i + 1, j]])

    # Create a line set from the point cloud and the lines
    line_set = o3d.geometry.LineSet(
        points=dot.points,
        lines=o3d.utility.Vector2iVector(lines),
    )

    return line_set


def get_wirepoints_from_pcd(pcd, div_u, div_v):
    points = np.asarray(pcd.points)
    ex_points = get_extreme_points_from_pcd(pcd)
    top = ex_points[0]
    bottom = ex_points[1]
    height = np.linalg.norm(top - bottom)
    segment_height = height / div_u

    wirepoints = []
    for i in range(div_u):
        segment_bottom = bottom[1] + segment_height * i
        segment_top = bottom[1] + segment_height * (i + 1)
        cur_idx = np.where(
            (points[:, 1] >= segment_bottom) & (points[:, 1] < segment_top)
        )[0]
        if len(cur_idx) > 0:
            cur_pcd = pcd.select_by_index(cur_idx)
            ex_points = get_extreme_points_from_pcd(cur_pcd)
            for e in ex_points[2:]:
                wirepoints.append(e)
        # o3d.visualization.draw_geometries([cur_pcd])
    return wirepoints


def get_extreme_points_from_pcd(pcd):
    # get the most top point from point cloud
    points = np.asarray(pcd.points)
    top_point = points[np.argmax(points[:, 1])]
    # get the most bottom point from point cloud
    bottom_point = points[np.argmin(points[:, 1])]
    # get the most left point from point cloud
    left_point = points[np.argmin(points[:, 0])]
    # get the most right point from point cloud
    right_point = points[np.argmax(points[:, 0])]
    # get the most front point from point cloud
    front_point = points[np.argmax(points[:, 2])]
    # get the most back point from point cloud
    back_point = points[np.argmin(points[:, 2])]
    return top_point, bottom_point, left_point, front_point, right_point, back_point


def get_scaled_ellipsoid_from_pcd(pcd, scaling_factor=1, vis=False):
    points = np.asarray(pcd.points)
    scale = scaling_factor - 1
    # Find the six extreme points
    min_x, min_y, min_z = np.min(points, axis=0)
    max_x, max_y, max_z = np.max(points, axis=0)

    extreme_points = np.array(
        [
            [min_x, (min_y + max_y) / 2, (min_z + max_z) / 2],  # Left
            [max_x, (min_y + max_y) / 2, (min_z + max_z) / 2],  # Right
            [(min_x + max_x) / 2, min_y, (min_z + max_z) / 2],  # Bottom
            [(min_x + max_x) / 2, max_y, (min_z + max_z) / 2],  # Top
            [(min_x + max_x) / 2, (min_y + max_y) / 2, min_z],  # Back
            [(min_x + max_x) / 2, (min_y + max_y) / 2, max_z],  # Front
        ]
    )
    # Define the center and radii of the ellipsoid based on extreme points
    center = np.mean(extreme_points, axis=0)
    radius = (max_x - min_x) / 2, (max_y - min_y) / 2, (max_z - min_z) / 2

    radius_mid = np.median(np.array(radius))

    bonus_radius = radius_mid * scale

    # Apply scaling factors
    scaled_radius = [
        radius[0] + bonus_radius,
        radius[1] + bonus_radius,
        radius[2] + bonus_radius,
    ]
    print(scaled_radius)
    return center, scaled_radius


def get_ellipsoid_from_pcd(pcd, vis=False):
    points = np.asarray(pcd.points)
    # Find the six extreme points
    min_x, min_y, min_z = np.min(points, axis=0)
    max_x, max_y, max_z = np.max(points, axis=0)

    extreme_points = np.array(
        [
            [min_x, (min_y + max_y) / 2, (min_z + max_z) / 2],  # Left
            [max_x, (min_y + max_y) / 2, (min_z + max_z) / 2],  # Right
            [(min_x + max_x) / 2, min_y, (min_z + max_z) / 2],  # Bottom
            [(min_x + max_x) / 2, max_y, (min_z + max_z) / 2],  # Top
            [(min_x + max_x) / 2, (min_y + max_y) / 2, min_z],  # Back
            [(min_x + max_x) / 2, (min_y + max_y) / 2, max_z],  # Front
        ]
    )

    # Define the center and radii of the ellipsoid based on extreme points
    center = np.mean(extreme_points, axis=0)
    radii = (max_x - min_x) / 2, (max_y - min_y) / 2, (max_z - min_z) / 2

    if vis:
        # Create a mesh grid for the ellipsoid
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        x = center[0] + radii[0] * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radii[1] * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radii[2] * np.outer(np.ones_like(u), np.cos(v))

        # Create vertices for the wireframe
        vertices = np.vstack((x.ravel(), y.ravel(), z.ravel())).T

        # Create lines for the wireframe
        lines = []
        for i in range(len(u)):
            for j in range(len(v) - 1):
                lines.append([i * len(v) + j, i * len(v) + j + 1])
        for i in range(len(u) - 1):
            for j in range(len(v)):
                lines.append([i * len(v) + j, (i + 1) * len(v) + j])

        # Create Open3D LineSet for the wireframe ellipsoid
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(vertices)
        line_set.lines = o3d.utility.Vector2iVector(lines)

        # Define colors for the lines
        colors = [[0, 0, 1] for _ in lines]  # Blue color
        line_set.colors = o3d.utility.Vector3dVector(colors)

        # Visualize the point cloud and the wireframe ellipsoid
        # o3d.visualization.draw_geometries([pcd, line_set])

    return center, radii


def add_outlier_in_bbox(pcd, scale=2, percentage=0.1):
    pcd2 = copy.deepcopy(pcd)
    pcd2.scale(scale, center=pcd2.get_center())
    bbox = pcd2.get_axis_aligned_bounding_box()

    # generate random points in bbox
    num_points = int(len(pcd2.points) * percentage)
    min_bound = bbox.get_min_bound()
    max_bound = bbox.get_max_bound()
    outlier_points = np.random.uniform(min_bound, max_bound, (num_points, 3))

    # add outlier points to pcd
    outlier = o3d.geometry.PointCloud()
    outlier.points = o3d.utility.Vector3dVector(outlier_points)
    outlier.paint_uniform_color([1, 0, 0])

    return pcd + outlier


def add_outlier_in_center_cube(pcd, cube_scale=0.1, percentage=0.1):
    # Copy the original point cloud to avoid modifying it directly
    pcd2 = copy.deepcopy(pcd)

    # Calculate the center of the point cloud
    center = pcd2.get_center()

    # Define the cube's bounds based on the cube_scale
    cube_min_bound = center - cube_scale / 2
    cube_max_bound = center + cube_scale / 2

    # Generate random points within the cube
    num_points = int(len(pcd2.points) * percentage)
    outlier_points = np.random.uniform(cube_min_bound, cube_max_bound, (num_points, 3))

    # Add outlier points to the point cloud
    outlier = o3d.geometry.PointCloud()
    outlier.points = o3d.utility.Vector3dVector(outlier_points)

    # color with random color
    # Generate a random color for each point
    colors = np.random.rand(
        len(outlier.points), 3
    )  # Generates an array of shape (N, 3) with random colors
    outlier.colors = o3d.utility.Vector3dVector(colors)
    # outlier.paint_uniform_color([0.5, 0, 0])

    # Combine the original point cloud with the outliers
    combined_pcd = pcd + outlier

    return combined_pcd


def calc_density(pcd, radius=0.5):
    # Compute the density of the point cloud
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    densities = []
    for i in range(len(pcd.points)):
        [_, indices, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], radius)
        density = len(indices)
        densities.append(density)
    return np.array(densities)


def visualize_surface(surf, sample_size=50):

    # Set sample size
    surf.sample_size_u = sample_size
    surf.sample_size_v = sample_size

    # Evaluate surface
    surf.evaluate()

    # Get evaluated points
    surface_points = np.array(surf.evalpts)

    # Correctly reshape the surface points
    surface_points = surface_points.reshape((sample_size, sample_size, 3))

    # Convert control points to numpy array
    ctrl_points = np.array(surf.ctrlpts)
    num_u = surf.ctrlpts_size_u
    num_v = surf.ctrlpts_size_v
    ctrl_points = ctrl_points.reshape((num_u, num_v, 3))
    # Create the 3D plot
    fig = go.Figure()

    # Add the B-spline surface
    fig.add_trace(
        go.Surface(
            x=surface_points[:, :, 0],
            y=surface_points[:, :, 1],
            z=surface_points[:, :, 2],
            name="B-spline surface",
            showscale=False,
            opacity=0.9,
            colorscale="viridis",
            contours={
                "x": {"show": True, "width": 1, "color": "white"},
                "y": {"show": True, "width": 1, "color": "white"},
            },
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        )
    )
    fig.show()
