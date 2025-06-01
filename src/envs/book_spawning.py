import numpy as np

def get_corners(box):
    """Returns the 4 corners of the box in the x-y plane, given (X, Y, Z, x, y, z, yaw)."""
    X, Y, _, x_c, y_c, _, yaw = box
    # Box local corners
    dx = X / 2
    dy = Y / 2
    corners = np.array([
        [ dx,  dy],
        [-dx,  dy],
        [-dx, -dy],
        [ dx, -dy]
    ])
    # Rotation matrix
    R = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw),  np.cos(yaw)]
    ])
    # Rotate and translate
    return np.dot(corners, R.T) + np.array([x_c, y_c])

def project_onto_axis(corners, axis):
    """Projects corners onto the given axis and returns min/max."""
    projections = np.dot(corners, axis)
    return projections.min(), projections.max()

def obb_2d_overlap(box1, box2):
    """Checks 2D OBB overlap using SAT."""
    corners1 = get_corners(box1)
    corners2 = get_corners(box2)
    # Axes: box1 edges and box2 edges (normals)
    axes = []
    for corners in [corners1, corners2]:
        for i in range(4):
            edge = corners[(i+1)%4] - corners[i]
            axis = np.array([-edge[1], edge[0]])  # Perpendicular
            axis /= np.linalg.norm(axis)
            axes.append(axis)
    # SAT test
    for axis in axes:
        min1, max1 = project_onto_axis(corners1, axis)
        min2, max2 = project_onto_axis(corners2, axis)
        if max1 < min2 or max2 < min1:
            return False  # Separating axis found
    return True

def boxes_collide(box1, box2):
    """
    Checks if two boxes (with possible yaw) collide.
    """
    # Unpack box parameters
    X1, Y1, Z1, x1, y1, z1, yaw1 = box1
    X2, Y2, Z2, x2, y2, z2, yaw2 = box2

    # Check for overlap in z (axis-aligned)
    overlap_z = abs(z1 - z2) < (Z1 + Z2) / 2
    if not overlap_z:
        return False

    # If both yaws are 0, use fast AABB check
    if yaw1 == 0 and yaw2 == 0:
        overlap_x = abs(x1 - x2) < (X1 + X2) / 2
        overlap_y = abs(y1 - y2) < (Y1 + Y2) / 2
        return overlap_x and overlap_y and overlap_z

    # Otherwise, use OBB check in x-y
    return obb_2d_overlap(box1, box2) and overlap_z

def generate_random_box_params(shelf_size, box_size_ranges, num_samples=1000, num_boxes=1, allow_yaw=False, max_attempts=100):
    """
    Generates random sizes and positions x on a fixed-size shelf, allowing generation of an arbitrary yaw rotation as well.
    Ensures that boxes do not collide with each other.
    """
    all_samples = []

    for _ in range(num_samples):
        boxes = []
        for box_idx in range(num_boxes):
            for attempt in range(max_attempts):
                # Randomly select box dimensions within the given range
                X_b = np.random.uniform(*box_size_ranges['x'])
                Y_b = np.random.uniform(*box_size_ranges['y'])
                Z_b = np.random.uniform(*box_size_ranges['z'])

                yaw = 0  # Default yaw angle
                if allow_yaw:
                    yaw = np.random.uniform(0, 2 * np.pi)
                    X_b_rot = abs(X_b * np.cos(yaw)) + abs(Y_b * np.sin(yaw))
                    Y_b_rot = abs(X_b * np.sin(yaw)) + abs(Y_b * np.cos(yaw))
                else:
                    X_b_rot, Y_b_rot = X_b, Y_b

                x = np.random.uniform(X_b_rot / 2, shelf_size[0] - X_b_rot / 2)
                y = np.random.uniform(Y_b_rot / 2, shelf_size[1] - Y_b_rot / 2)
                z = np.random.uniform(Z_b / 2, shelf_size[2] - Z_b / 2)

                new_box = (X_b, Y_b, Z_b, x, y, z, yaw)

                # Check for collision with existing boxes
                collision = False
                for prev_box in boxes:
                    try:
                        if boxes_collide(new_box, prev_box):
                            collision = True
                            break
                    except NotImplementedError:
                        # If yaw != 0, skip collision check for now
                        collision = False

                if not collision:
                    boxes.append(new_box)
                    break
            else:
                # Could not place this box without collision after max_attempts
                break

        if len(boxes) == num_boxes:
            all_samples.append(boxes)

    return all_samples