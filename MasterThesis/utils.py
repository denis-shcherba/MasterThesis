import numpy as np
import robotic as ry
import cmath 
from collections import deque

def point_in_box_filtering(points, box_params):
    """
    Filters points that are within a given 3D bounding box.

    Parameters:
    points (np.ndarray): Nx3 array of 3D points.
    box_params (tuple): (center, size), where:
        - center (array-like): (cx, cy, cz) center of the box.
        - size (array-like): (sx, sy, sz) size of the box along x, y, z.

    Returns:
    np.ndarray: Filtered array containing only points inside the bounding box.
    """
    center, size = box_params
    center = np.array(center)
    size = np.array(size)
    
    min_bound = center - size / 2
    max_bound = center + size / 2
    
    mask = np.all((points >= min_bound) & (points <= max_bound), axis=1)
    
    return points[mask]

def cuboid_corners_to_size_com(corner_points):
    """
    Transforms cuboid corners to its size and COM

    Parameters:
    points (np.ndarray): 8x3 array of cuboid corner.
    """

    # Transform corner points to sizes in x, y, z
    z = np.linalg.norm(corner_points[0]-corner_points[1], 2)
    y = np.linalg.norm(corner_points[1]-corner_points[3], 2)
    x = np.linalg.norm(corner_points[0]-corner_points[4], 2)
    center=np.sum(corner_points, axis=0)/len(corner_points)

    return center, [x, y, z]

def sample_cuboid_edges(C, box_frame, yaw, sides_to_sample=[True, True, True, True], sides_rel=False, samples=10):
    """
    Samples points from the edges of a cuboid except for specified sides.
    
    Parameters:
    C (ry.Config): Configuration
    box_frame (string): Frame name of the box
    yaw (float): Yaw rotation of the cuboid in radians
    sides_to_sample (list): List of four booleans indicating which sides to sample:
                           [right, top, left, bottom] - True to sample that side
    samples (int): Total number of samples to be drawn
    
    Returns:
    List of 3D points on the selected cuboid edges
    """
    x_size = C.getFrame(box_frame).getSize()[0]
    y_size = C.getFrame(box_frame).getSize()[1]
    box_com = C.getFrame(box_frame).getPosition()
    
    if sides_rel:
        index = int((yaw + np.pi / 4) // (np.pi / 2)) % 4
        sides_to_sample = shift_list(sides_to_sample, index)

    # Check if we have at least one side to sample
    if not any(sides_to_sample):
        raise ValueError("At least one side must be selected for sampling")
    
    # Count how many sides we're sampling
    sides_count = sum(sides_to_sample)
    
    # Calculate samples per side (distribute evenly)
    samples_per_side = [0, 0, 0, 0]
    base_samples = samples // sides_count
    remainder = samples % sides_count
    
    side_index = 0
    for i in range(4):
        if sides_to_sample[i]:
            samples_per_side[i] = base_samples
            if side_index < remainder:
                samples_per_side[i] += 1
            side_index += 1
    
    edge_points = []
    
    # Sample the right side (x = x_size/2, y ranges from -y_size/2 to y_size/2)
    if sides_to_sample[0] and samples_per_side[0] > 0:
        y_values = np.linspace(-y_size/2, y_size/2, samples_per_side[0])
        for y in y_values:
            local_point = np.array([x_size/2, y, 0])
            rotated_point = rotate_point(local_point, yaw)
            edge_points.append(box_com + rotated_point)
    
    # Sample the top side (y = y_size/2, x ranges from x_size/2 to -x_size/2)
    if sides_to_sample[1] and samples_per_side[1] > 0:
        x_values = np.linspace(x_size/2, -x_size/2, samples_per_side[1])
        for x in x_values:
            local_point = np.array([x, y_size/2, 0])
            rotated_point = rotate_point(local_point, yaw)
            edge_points.append(box_com + rotated_point)
    
    # Sample the left side (x = -x_size/2, y ranges from y_size/2 to -y_size/2)
    if sides_to_sample[2] and samples_per_side[2] > 0:
        y_values = np.linspace(y_size/2, -y_size/2, samples_per_side[2])
        for y in y_values:
            local_point = np.array([-x_size/2, y, 0])
            rotated_point = rotate_point(local_point, yaw)
            edge_points.append(box_com + rotated_point)
    
    # Sample the bottom side (y = -y_size/2, x ranges from -x_size/2 to x_size/2)
    if sides_to_sample[3] and samples_per_side[3] > 0:
        x_values = np.linspace(-x_size/2, x_size/2, samples_per_side[3])
        for x in x_values:
            local_point = np.array([x, -y_size/2, 0])
            rotated_point = rotate_point(local_point, yaw)
            edge_points.append(box_com + rotated_point)
    
    return edge_points

def shift_list(lst, shift):
    d = deque(lst)
    d.rotate(-shift)  # Negative rotates left, positive rotates right
    return list(d)

def rotate_point(point, yaw):
    """
    Rotates a 2D point by the given yaw angle
    
    Parameters:
    point (np.array): Local coordinates [x, y, z]
    yaw (float): Rotation angle in radians
    
    Returns:
    np.array: Rotated coordinates
    """
    x, y, z = point
    rotated_x = x * np.cos(yaw) - y * np.sin(yaw)
    rotated_y = x * np.sin(yaw) + y * np.cos(yaw)
    return np.array([rotated_x, rotated_y, z])

def find_nearest_cuboid_edge_center(C, box_frame, yaw):
    """
    Given a cuboid and its yaw rotation, find the center point of the edge closest to 0 radiants observer
    """
    x_size = C.getFrame(box_frame).getSize()[0]
    y_size = C.getFrame(box_frame).getSize()[1]
    book_com = C.getFrame(box_frame).getPosition()

    # dir to push push dir waypoint from com
    directions = [
        (-x_size / 2, 0),
        (0, y_size / 2),
        (x_size / 2, 0),
        (0, -y_size / 2)
    ]

    index = int((yaw + np.pi / 4) // (np.pi / 2)) % 4
    delta = complex(*directions[index]) * cmath.exp(1j * yaw)

    return book_com+np.array([delta.real, delta.imag, 0])