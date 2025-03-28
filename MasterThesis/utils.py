import numpy as np
import robotic as ry

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

def plot_box(C, box_params):
    """
    Plots a box into current configuration for debug purposes
    
    C (ry.Config): 
        - Configuration
    box_params (tuple): (center, size), where:
        - center (array-like): (cx, cy, cz) center of the box.
        - size (array-like): (sx, sy, sz) size of the box along x, y, z.

    """
    C.addFrame("_temp_box") \
        .setPosition(box_params[0]) \
        .setShape(ry.ST.box, size=box_params[1]) \
        .setColor([0., 1., 0.]) \
        .setContact(0)

    C.view(True)     
    C.view(False)
