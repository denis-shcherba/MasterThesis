import numpy as np
import torch
import torch.nn.functional as F

def pose_9d_loss(pred_pose, target_pose, position_weight=1.0, orientation_weight=1.0):
    """
    Loss function for 9D pose representation (3D position + 6D orientation).
    
    Args:
        pred_pose: Predicted pose [batch_size, 9] - [px, py, pz, r11, r12, r13, r21, r22, r23]
        target_pose: Target pose [batch_size, 9] - same format
        position_weight: Weight for position loss component
        orientation_weight: Weight for orientation loss component
    
    Returns:
        Combined loss value
    """
    
    # Split pose into position and orientation components
    pred_pos = pred_pose[:, :3]          # [batch_size, 3]
    pred_rot_cols = pred_pose[:, 3:]     # [batch_size, 6]
    
    target_pos = target_pose[:, :3]      # [batch_size, 3]
    target_rot_cols = target_pose[:, 3:] # [batch_size, 6]
    
    # Position loss (simple L2)
    pos_loss = F.mse_loss(pred_pos, target_pos)
    
    # Orientation loss - several options:
    
    # Option 1: Direct L2 loss on matrix columns (simple but not geometrically meaningful)
    # rot_loss = F.mse_loss(pred_rot_cols, target_rot_cols)
    
    # Option 2: Cosine similarity loss (better for rotation matrices)
    rot_loss = rotation_matrix_columns_loss(pred_rot_cols, target_rot_cols)
    
    # Option 3: Geodesic loss (most geometrically meaningful)
    # rot_loss = geodesic_rotation_loss(pred_rot_cols, target_rot_cols)
    
    return position_weight * pos_loss + orientation_weight * rot_loss


def rotation_matrix_columns_loss(pred_rot_cols, target_rot_cols):
    """
    Loss based on cosine similarity of rotation matrix columns.
    This ensures the predicted columns have the same direction as target columns.
    """
    batch_size = pred_rot_cols.shape[0]
    
    # Reshape to [batch_size, 2, 3] - 2 columns, each 3D
    pred_cols = pred_rot_cols.view(batch_size, 2, 3)
    target_cols = target_rot_cols.view(batch_size, 2, 3)
    
    # Normalize columns to unit vectors
    pred_cols_norm = F.normalize(pred_cols, p=2, dim=2)
    target_cols_norm = F.normalize(target_cols, p=2, dim=2)
    
    # Cosine similarity loss for each column
    cos_sim_col1 = F.cosine_similarity(pred_cols_norm[:, 0], target_cols_norm[:, 0], dim=1)
    cos_sim_col2 = F.cosine_similarity(pred_cols_norm[:, 1], target_cols_norm[:, 1], dim=1)
    
    # Convert to loss (1 - cosine_similarity)
    loss_col1 = 1 - cos_sim_col1
    loss_col2 = 1 - cos_sim_col2
    
    return (loss_col1 + loss_col2).mean()


def geodesic_rotation_loss(pred_rot_cols, target_rot_cols):
    """
    Geodesic distance on SO(3) - most geometrically meaningful for rotations.
    Reconstructs full rotation matrices and computes the geodesic distance.
    """
    batch_size = pred_rot_cols.shape[0]
    
    # Reconstruct full rotation matrices from 2 columns
    pred_R = reconstruct_rotation_matrix(pred_rot_cols)  # [batch_size, 3, 3]
    target_R = reconstruct_rotation_matrix(target_rot_cols)  # [batch_size, 3, 3]
    
    # Compute relative rotation: R_rel = R_pred^T @ R_target
    R_rel = torch.bmm(pred_R.transpose(-2, -1), target_R)
    
    # Geodesic distance: ||log(R_rel)||_F where log is matrix logarithm
    # For rotation matrices, this simplifies to: arccos((trace(R_rel) - 1) / 2)
    trace_R_rel = torch.diagonal(R_rel, dim1=-2, dim2=-1).sum(dim=-1)
    
    # Clamp to avoid numerical issues with arccos
    cos_angle = (trace_R_rel - 1) / 2
    cos_angle = torch.clamp(cos_angle, -1 + 1e-7, 1 - 1e-7)
    
    geodesic_dist = torch.acos(torch.abs(cos_angle))
    
    return geodesic_dist.mean()


def reconstruct_rotation_matrix(rot_cols):
    """
    Reconstruct full 3x3 rotation matrix from 2 columns using Gram-Schmidt.
    
    Args:
        rot_cols: [batch_size, 6] - two 3D columns of rotation matrix
    
    Returns:
        R: [batch_size, 3, 3] - full rotation matrices
    """
    batch_size = rot_cols.shape[0]
    
    # Reshape to [batch_size, 2, 3]
    cols = rot_cols.view(batch_size, 2, 3)
    
    # Extract first two columns
    col1 = cols[:, 0]  # [batch_size, 3]
    col2 = cols[:, 1]  # [batch_size, 3]
    
    # Gram-Schmidt orthogonalization
    u1 = F.normalize(col1, p=2, dim=1)
    
    # Remove component of col2 in direction of u1
    u2 = col2 - torch.sum(col2 * u1, dim=1, keepdim=True) * u1
    u2 = F.normalize(u2, p=2, dim=1)
    
    # Third column is cross product
    u3 = torch.cross(u1, u2, dim=1)
    
    # Stack to form rotation matrix
    R = torch.stack([u1, u2, u3], dim=2)  # [batch_size, 3, 3]
    
    return R


def weighted_pose_loss(pred_pose, target_pose, pos_weight=1.0, rot_weight=1.0):
    """
    Simple weighted combination of position and rotation losses.
    Good starting point for most applications.
    """
    pred_pos = pred_pose[:, :3]
    pred_rot = pred_pose[:, 3:]
    target_pos = target_pose[:, :3]
    target_rot = target_pose[:, 3:]
    
    pos_loss = F.mse_loss(pred_pos, target_pos)
    rot_loss = F.mse_loss(pred_rot, target_rot)
    
    return pos_weight * pos_loss + rot_weight * rot_loss


def robust_pose_loss(pred_pose, target_pose, pos_weight=1.0, rot_weight=1.0, use_huber=True):
    """
    Robust loss function using Huber loss to handle outliers.
    """
    pred_pos = pred_pose[:, :3]
    pred_rot = pred_pose[:, 3:]
    target_pos = target_pose[:, :3]
    target_rot = target_pose[:, 3:]
    
    if use_huber:
        pos_loss = F.huber_loss(pred_pos, target_pos)
        rot_loss = F.huber_loss(pred_rot, target_rot)
    else:
        pos_loss = F.l1_loss(pred_pos, target_pos)
        rot_loss = F.l1_loss(pred_rot, target_rot)
    
    return pos_weight * pos_loss + rot_weight * rot_loss


# Usage examples
if __name__ == "__main__":
    # Example usage
    batch_size = 32
    
    # Random example poses (in practice, these would be your actual data)
    pred_poses = torch.randn(batch_size, 9)
    target_poses = torch.randn(batch_size, 9)
    
    # Normalize rotation columns to make them more realistic
    pred_poses[:, 3:6] = F.normalize(pred_poses[:, 3:6], p=2, dim=1)
    pred_poses[:, 6:9] = F.normalize(pred_poses[:, 6:9], p=2, dim=1)
    target_poses[:, 3:6] = F.normalize(target_poses[:, 3:6], p=2, dim=1)
    target_poses[:, 6:9] = F.normalize(target_poses[:, 6:9], p=2, dim=1)
    
    # Compare different loss functions
    print("Loss function comparison:")
    print(f"Simple weighted loss: {weighted_pose_loss(pred_poses, target_poses):.4f}")
    print(f"Cosine similarity loss: {pose_9d_loss(pred_poses, target_poses):.4f}")
    print(f"Robust Huber loss: {robust_pose_loss(pred_poses, target_poses):.4f}")
    
    # You can also compute geodesic loss (more expensive but most accurate)
    # print(f"Geodesic loss: {geodesic_rotation_loss(pred_poses[:, 3:], target_poses[:, 3:]):.4f}")


# RECOMMENDATIONS:
"""
For your 9D pose representation, I recommend:

1. START SIMPLE: Use weighted_pose_loss() with pos_weight=1.0, rot_weight=0.1
   (rotation errors are typically smaller in magnitude than position errors)

2. FOR BETTER GEOMETRY: Use pose_9d_loss() with cosine similarity for rotations

3. FOR ROBUSTNESS: Use robust_pose_loss() if you have outliers in your data

4. FOR BEST ACCURACY: Use geodesic_rotation_loss() if computational cost is not critical

The key insight is that position (meters) and rotation (unitless) have different
scales, so you need to weight them appropriately!
"""