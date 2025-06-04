
import numpy as np
import torch
import robotic as ry

def pose_9d_to_7d(pose: np.ndarray) -> np.ndarray:
    """
    Convert a 9D pose (position + 6D rotation from first two columns) to a 7D pose (position + quaternion).
    The rotation columns are assumed to be predicted by a network and may not be unit norm or orthogonal.
    SVD is used to find the closest valid rotation matrix.
    
    Args:
        pose: A NumPy array of shape (9,) representing the 9D pose.
              Assumed order: [px, py, pz, R00, R10, R20, R01, R11, R21]
              where R_ij are elements of the first two columns of the rotation matrix.
        
    Returns:
        A NumPy array of shape (7,) representing the 7D pose [px, py, pz, qx, qy, qz, qw].
    """
    
    # Extract position (first 3 elements)
    position = pose[:3]
    
    # Extract the predicted first two columns of the rotation matrix
    pred_col1 = pose[3:6] # R00, R10, R20
    pred_col2 = pose[6:9] # R01, R11, R21
    
    # --- Step 1: Normalize the predicted columns (optional but good practice) ---
    # This helps ensure stability if the network predictions are wildly off-scale
    # If the network already tries to predict unit vectors, this step might be less critical
    # but it doesn't hurt.
    
    norm_col1 = np.linalg.norm(pred_col1)
    if norm_col1 > 1e-6: # Avoid division by zero
        col1 = pred_col1 / norm_col1
    else:
        col1 = np.array([1.0, 0.0, 0.0]) # Default to X-axis if column is zero vector
        print("Warning: First predicted rotation column is near zero. Defaulting to X-axis.")

    norm_col2 = np.linalg.norm(pred_col2)
    if norm_col2 > 1e-6: # Avoid division by zero
        col2 = pred_col2 / norm_col2
    else:
        col2 = np.array([0.0, 1.0, 0.0]) # Default to Y-axis if column is zero vector
        print("Warning: Second predicted rotation column is near zero. Defaulting to Y-axis.")
    
    # --- Step 2: Reconstruct an initial 3x3 matrix ---
    # Derive the third column as the cross product of the first two.
    # This matrix M might still not be perfectly orthogonal or have det=1
    
    col3 = np.cross(col1, col2)
    
    # If the first two columns are nearly parallel, the cross product will be near zero.
    # In such cases, we need a fallback for col3 to ensure a non-degenerate matrix.
    norm_col3 = np.linalg.norm(col3)
    if norm_col3 < 1e-6:
        # If col1 and col2 are parallel, pick an arbitrary orthogonal third axis.
        # This is a bit of a heuristic. A more robust solution might involve PCA
        # or a different network output strategy if this happens frequently.
        # For now, let's try to find a vector orthogonal to col1.
        if np.isclose(np.linalg.norm(np.cross(col1, [0, 1, 0])), 0):
            col3 = np.cross(col1, [0, 0, 1])
        else:
            col3 = np.cross(col1, [0, 1, 0])
        # Ensure it's unit norm
        col3 = col3 / np.linalg.norm(col3)
        print("Warning: First two predicted rotation columns are nearly parallel. Approximating third column.")



    M = np.vstack((col1, col2, col3)).T 

    U, S, Vt = np.linalg.svd(M)
    
    R = U @ Vt
    if np.linalg.det(R) < 0:
        # Flip the sign of the last column of U or V_transpose.
        # A common approach is to flip the last column of U
        U[:, 2] *= -1
        R = U @ Vt
        # Now det(R) should be +1

    # Convert rotation matrix to quaternion using ry
    # Ensure R is float32 if ry expects it, or if you need consistency
    quat_arr = ry.Quaternion().setMatrix(R.astype(np.float32)).asArr()
    
    pose_7d = np.concatenate((position, quat_arr))
    
    return pose_7d.astype(np.float32)

def pose_7d_to_9d(pose: np.ndarray) -> np.ndarray:
    """
    Convert a 7D pose (position + quaternion) to a 9D pose (position + 6D rotation from first two columns).
    
    Args:
        pose: A NumPy array of shape (7,) representing the 7D pose [px, py, pz, qx, qy, qz, qw].
        
    Returns:
        A NumPy array of shape (9,) representing the 9D pose 
        [px, py, pz, R00, R10, R20, R01, R11, R21].
    """
    
    # Extract position (first 3 elements)
    position = pose[:3]
    
    quat_values = pose[3:] 
    
    q = ry.Quaternion().set(quat_values) 
    R = q.getMatrix() 
    
    col1 = R[:, 0] 
    col2 = R[:, 1] 
    
    pose_9d = np.concatenate((position, col1, col2))
    
    return pose_9d.astype(np.float32)


def normalize_point_cloud_to_unit_sphere(points: np.ndarray) -> np.ndarray:
    """
    Normalize a single point cloud to be centered and fit within a unit sphere.
    
    Args:
        points: Point cloud array of shape (num_points, 3)
        
    Returns:
        Normalized point cloud as a float32 NumPy array.
    """
    if points.shape[0] == 0: # Handle empty point cloud
        return points.astype(np.float32)
        
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    
    max_dist = np.max(np.linalg.norm(points_centered, axis=1))
    if max_dist > 1e-8:  # Avoid division by zero or near-zero
        points_normalized = points_centered / max_dist
    else:
        points_normalized = points_centered # Or return points_centered if points are already very close
        
    return points_normalized.astype(np.float32)


def normalize_point_cloud_to_unit_sphere_torch(points_tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalize a single point cloud tensor to be centered and fit within a unit sphere.
    
    Args:
        points_tensor: Point cloud tensor of shape (num_points, 3)
        
    Returns:
        Normalized point cloud tensor.
    """
    if points_tensor.shape[0] == 0:
        return points_tensor

    centroid = torch.mean(points_tensor, dim=0)
    points_centered = points_tensor - centroid
    
    max_dist = torch.max(torch.norm(points_centered, dim=1))
    if max_dist > 1e-8:
        points_normalized = points_centered / max_dist
    else:
        points_normalized = points_centered
        
    return points_normalized
