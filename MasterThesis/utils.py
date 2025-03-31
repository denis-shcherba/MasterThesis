import numpy as np
import robotic as ry
import cmath 

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

def sample_equiangular_cuboid_edges(C, box_frame, yaw, angle_interval=[0, 2 * np.pi], samples=10):
    """
    Samples equiangular points from edges of a cuboid considering its yaw rotation.
    
    Parameters:
    C (ry.Config): Configuration
    box_frame (string): Frame name of the box
    yaw (float): Yaw rotation of the cuboid in radians
    angle_interval (list): Angle interval [start, end] in radians
    samples (int): Number of samples to be drawn
    
    Returns:
    List of 3D points on the cuboid edges
    """
    x_size = C.getFrame(box_frame).getSize()[0]
    y_size = C.getFrame(box_frame).getSize()[1]
    box_com = C.getFrame(box_frame).getPosition()
    
    # Generate equiangular samples
    angles = np.linspace(angle_interval[0], angle_interval[1], samples, endpoint=False)
    edge_points = []
    
    for angle in angles:
        # Adjust the angle to account for the yaw rotation of the box
        adjusted_angle = (angle - yaw) % (2 * np.pi)
        
        # Determine which edge to sample based on the angle
        if 0 <= adjusted_angle < np.pi/2:  # First quadrant (0° to 90°)
            # Interpolate between right edge and top edge
            if adjusted_angle < np.pi/4:
                # Closer to right edge (0°)
                t = adjusted_angle / (np.pi/4)
                x = x_size/2
                y = -y_size/2 + t * y_size
            else:
                # Closer to top edge (90°)
                t = (adjusted_angle - np.pi/4) / (np.pi/4)
                x = x_size/2 - t * x_size
                y = y_size/2
                
        elif np.pi/2 <= adjusted_angle < np.pi:  # Second quadrant (90° to 180°)
            # Interpolate between top edge and left edge
            if adjusted_angle < 3*np.pi/4:
                # Closer to top edge (90°)
                t = (adjusted_angle - np.pi/2) / (np.pi/4)
                x = -x_size/2 + t * x_size
                y = y_size/2
            else:
                # Closer to left edge (180°)
                t = (adjusted_angle - 3*np.pi/4) / (np.pi/4)
                x = -x_size/2
                y = y_size/2 - t * y_size
                
        elif np.pi <= adjusted_angle < 3*np.pi/2:  # Third quadrant (180° to 270°)
            # Interpolate between left edge and bottom edge
            if adjusted_angle < 5*np.pi/4:
                # Closer to left edge (180°)
                t = (adjusted_angle - np.pi) / (np.pi/4)
                x = -x_size/2
                y = -y_size/2 + t * y_size
            else:
                # Closer to bottom edge (270°)
                t = (adjusted_angle - 5*np.pi/4) / (np.pi/4)
                x = -x_size/2 + t * x_size
                y = -y_size/2
                
        else:  # Fourth quadrant (270° to 360°)
            # Interpolate between bottom edge and right edge
            if adjusted_angle < 7*np.pi/4:
                # Closer to bottom edge (270°)
                t = (adjusted_angle - 3*np.pi/2) / (np.pi/4)
                x = x_size/2 - t * x_size
                y = -y_size/2
            else:
                # Closer to right edge (360°/0°)
                t = (adjusted_angle - 7*np.pi/4) / (np.pi/4)
                x = x_size/2
                y = -y_size/2 + t * y_size
        
        # Rotate the local coordinates by yaw before adding to box_com
        rotated_x = x * np.cos(yaw) - y * np.sin(yaw)
        rotated_y = x * np.sin(yaw) + y * np.cos(yaw)
        
        # Add point to the list
        edge_points.append(box_com + np.array([rotated_x, rotated_y, 0]))
    
    return edge_points

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