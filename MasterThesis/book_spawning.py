import numpy as np

def generate_random_box_params(shelf_size, box_size_ranges, num_samples=1000, allow_yaw=False, allow_ss_sampling=False):
    """
    Generates random sizes and positions x on a fixed-size shelf, allowing generation of an arbitrary yaw rotation as well.
    
    :param shelf_size: Tuple (X_s, Y_s, Z_s) - dimensions of the shelf.
    :param box_size_ranges: Dict { 'x': (min, max), 'y': (min, max), 'z': (min, max) } - size ranges of the box.
    :param num_samples: Number of samples to generate.
    :param allow_yaw: Whether to allow arbitrary yaw rotation.
    :return: List of tuples [(X_b, Y_b, Z_b, x, y, z, yaw), ...]
    """
    params = []
    
    for _ in range(num_samples):
        # Randomly select box dimensions within the given range
        X_b = np.random.uniform(*box_size_ranges['x'])
        Y_b = np.random.uniform(*box_size_ranges['y'])
        Z_b = np.random.uniform(*box_size_ranges['z'])

        if allow_ss_sampling:        
            ss_b = np.random.uniform(*box_size_ranges['ss'])

        yaw = 0  # Default yaw angle
        if allow_yaw:
            yaw = np.random.uniform(0, 2 * np.pi)  # Random yaw angle in radians
            
            # Compute bounding box after rotation
            X_b_rot = abs(X_b * np.cos(yaw)) + abs(Y_b * np.sin(yaw))
            Y_b_rot = abs(X_b * np.sin(yaw)) + abs(Y_b * np.cos(yaw))
        else:
            X_b_rot, Y_b_rot = X_b, Y_b
        
        # Ensure box fits within the shelf and randomly place it
        x = np.random.uniform(X_b_rot / 2, shelf_size[0] - X_b_rot / 2)
        y = np.random.uniform(Y_b_rot / 2, shelf_size[1] - Y_b_rot / 2)
        z = np.random.uniform(Z_b / 2, shelf_size[2] - Z_b / 2)
        
        params.append((X_b, Y_b, Z_b, x, y, z, yaw))
    
    return params