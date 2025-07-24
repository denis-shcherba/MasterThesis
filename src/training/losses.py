import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

def create_individual_loss_function(loss_cfg):
    """
    Create a standard individual loss function based on configuration.
    
    Args:
        loss_cfg: Configuration for the individual loss (e.g., type, delta, beta).
        
    Returns:
        criterion: An nn.Module loss function.
    """
    loss_type = loss_cfg.get('type', 'mse').lower()
    
    if loss_type == 'mse':
        criterion = nn.MSELoss()
    elif loss_type == 'l1':
        criterion = nn.L1Loss()
    else:
        raise ValueError(f"Unknown individual loss type: {loss_type}")
    
    return criterion

# --- Rotation Matrix Reconstruction ---
def reconstruct_rotation_matrix_from_6d(vectors_6d):
    """
    Reconstructs a batch of 3x3 rotation matrices from a batch of 6D representations.
    The 6D representation consists of the first two columns of the rotation matrix.
    Uses Gram-Schmidt orthogonalization.

    Args:
        vectors_6d (torch.Tensor): Tensor of shape (..., 6) representing
                                   the first two columns (concatenated).
                                   Ellipsis (...) indicates optional batch dimensions.
    
    Returns:
        torch.Tensor: Tensor of shape (..., 3, 3) representing rotation matrices.
    """
    col1_pred = vectors_6d[..., 0:3]
    col2_pred = vectors_6d[..., 3:6]

    # Normalize the first column
    r1 = F.normalize(col1_pred, dim=-1, eps=1e-8)

    # Make the second column orthogonal to the first and normalize it
    # r2_proj = col2_pred - torch.sum(r1 * col2_pred, dim=-1, keepdim=True) * r1
    # r2 = F.normalize(r2_proj, dim=-1, eps=1e-8)
    
    # More robust method (often used for 6D to SO(3)):
    # Use cross product to ensure right-handed orthogonal basis
    # r1 is x-axis
    # r3 is z-axis (orthogonal to r1 and col2_pred plane)
    # r2 is y-axis (orthogonal to r1 and r3)
    r3_unnormalized = torch.cross(r1, col2_pred, dim=-1)
    r3 = F.normalize(r3_unnormalized, dim=-1, eps=1e-8)
    
    r2 = torch.cross(r3, r1, dim=-1) # r2 is already normalized if r1 and r3 are orthonormal

    # Stack to form the rotation matrix
    # Columns of R are r1, r2, r3
    R = torch.stack([r1, r2, r3], dim=-1) # (..., 3, 3)
    return R

# --- Geodesic Loss for Rotations ---
class GeodesicLoss(nn.Module):
    """
    Computes the mean geodesic distance between batches of rotation matrices.
    The geodesic distance is the angle of the minimal rotation transforming one
    orientation to another.
    """
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, R_pred, R_gt):
        """
        Args:
            R_pred (torch.Tensor): Predicted rotation matrices, shape (..., 3, 3).
            R_gt (torch.Tensor): Ground truth rotation matrices, shape (..., 3, 3).
        
        Returns:
            torch.Tensor: Scalar tensor representing the mean geodesic distance in radians.
        """
        # Ensure inputs are rotation matrices (or at least try to make them so)
        # This should ideally be handled by reconstruction if inputs are 6D

        # Relative rotation: R_rel = R_pred^T * R_gt
        # Transpose R_pred: (..., 3, 3) -> (..., 3, 3)
        R_pred_T = R_pred.transpose(-2, -1)
        R_rel = torch.matmul(R_pred_T, R_gt)

        # Angle from trace of R_rel
        # trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
        trace = torch.diagonal(R_rel, offset=0, dim1=-2, dim2=-1).sum(dim=-1)
        
        # Clamp to avoid acos(x) with |x| > 1 due to numerical errors
        # (trace - 1) / 2 = cos(angle)
        cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0 + self.eps, 1.0 - self.eps)
        angle_rad = torch.acos(cos_angle)
        
        return angle_rad.mean()

class PoseLoss9D(nn.Module):
    """
    Computes a combined loss for 9D poses (3D position + 6D rotation).
    The 6D rotation is represented by the first two columns of the rotation matrix.
    """
    def __init__(self, loss_cfg):
        super().__init__()
        self.lambda_pos = loss_cfg.get('lambda_pos', 1.0)
        self.lambda_rot = loss_cfg.get('lambda_rot', 1.0)

        # Position Loss
        self.pos_loss_config = loss_cfg.get('position_loss', {'type': 'mse'})
        self.pos_loss_fn = create_individual_loss_function(self.pos_loss_config)

        # Rotation Loss
        self.rot_loss_config = loss_cfg.get('rotation_loss', {'type': 'geodesic'})
        self.rot_loss_type = self.rot_loss_config.get('type', 'geodesic').lower()
        
        if self.rot_loss_type == 'mse_6d':
            self.rot_loss_fn = nn.MSELoss()
        elif self.rot_loss_type == 'geodesic':
            geodesic_eps = self.rot_loss_config.get('eps', 1e-7)
            self.rot_loss_fn = GeodesicLoss(eps=geodesic_eps)
        else:
            raise ValueError(f"Unknown rotation loss type in PoseLoss9D: {self.rot_loss_type}")

    def forward(self, prediction_9d, target_9d):
        """
        Args:
            prediction_9d (torch.Tensor): Predicted 9D poses, shape (..., 9).
                                          First 3 elements are position, next 6 are 6D rotation.
            target_9d (torch.Tensor): Ground truth 9D poses, shape (..., 9).
        
        Returns:
            torch.Tensor: Scalar tensor representing the total combined loss.
        """
        # Split into position and rotation components
        pred_pos = prediction_9d[..., 0:3]
        target_pos = target_9d[..., 0:3]

        pred_rot_6d = prediction_9d[..., 3:9]
        target_rot_6d = target_9d[..., 3:9]

        # Calculate Position Loss
        loss_pos = self.pos_loss_fn(pred_pos, target_pos)

        # Calculate Rotation Loss
        if self.rot_loss_type == 'mse_6d':
            # Direct MSE on the 6D vectors
            loss_rot = self.rot_loss_fn(pred_rot_6d, target_rot_6d)
        elif self.rot_loss_type == 'geodesic':
            # Reconstruct full rotation matrices from 6D representations
            R_pred = reconstruct_rotation_matrix_from_6d(pred_rot_6d)
            R_target = reconstruct_rotation_matrix_from_6d(target_rot_6d)
            loss_rot = self.rot_loss_fn(R_pred, R_target)
        else: 
            # This case should ideally not be reached due to __init__ checks
            loss_rot = torch.tensor(0.0, device=prediction_9d.device, dtype=prediction_9d.dtype)
            
        # Combine losses
        total_loss = self.lambda_pos * loss_pos + self.lambda_rot * loss_rot
        
        # Optionally, return individual components for logging
        # return total_loss, {"pos_loss": loss_pos.item(), "rot_loss": loss_rot.item()}
        return total_loss

class MSELoss(nn.Module):
    """
    Computes a simple Mean Squared Error (MSE) loss.
    This class can be used for any tensor input where MSE is desired.
    """
    def __init__(self):
        super().__init__()
        # Initialize the Mean Squared Error loss function
        self.mse_loss = nn.MSELoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prediction (torch.Tensor): The predicted tensor.
            target (torch.Tensor): The ground truth tensor.
        
        Returns:
            torch.Tensor: Scalar tensor representing the MSE loss.
        """
        loss = self.mse_loss(prediction, target)
        
        return loss



# --- Main Loss Creation Function (Factory) ---
log = logging.getLogger(__name__)

def create_loss_function(loss_cfg_global):
    """
    Create the main loss function based on the global loss configuration.
    This function acts as a factory.

    Args:
        loss_cfg_global: The main loss configuration object (e.g., from Hydra).
                         Expected to have a 'name' field to dispatch to the correct loss.
                         If 'name' is 'PoseLoss9D', it uses the specialized 9D pose loss.
                         Otherwise, it attempts to create a simple individual loss.
        
    Returns:
        criterion: An nn.Module loss function.
    
    Raises:
        ValueError: If the loss configuration is invalid or the specified loss name is unknown.
    """
    loss_name = loss_cfg_global.get("name", None)

    if loss_name == 'PoseLoss9D':
        # PoseLoss9D expects the full configuration, as it processes nested configs.
        log.info("Creating PoseLoss9D.")
        return PoseLoss9D(loss_cfg_global)
    
    elif loss_name == "MSELoss":
        # If the name is 'MSELoss', we create a simple MSE loss function.
        log.info("Creating MSELoss.")
        return MSELoss()

    elif loss_name is not None:
        # If a name is given and it's not PoseLoss9D, assume it's an individual loss type.
        # The 'name' field here corresponds to the 'type' field expected by create_individual_loss_function.
        # We construct a new config dict to match the expected input for create_individual_loss_function.
        # This assumes create_individual_loss_function expects a config like {'type': 'mse', 'param1': val1, ...}
        
        # Make a copy to avoid modifying the original config if it's shared
        individual_loss_cfg = dict(loss_cfg_global) 
        
        # Use the 'name' from global config as 'type' for individual loss
        individual_loss_cfg['type'] = loss_name 
        
        # Remove the 'name' field if it's not expected by individual_loss_cfg
        # (This is optional, depends on how create_individual_loss_function handles extra keys)
        del individual_loss_cfg['name'] 

        log.info(f"Creating individual loss function of type '{loss_name}'.")
        return create_individual_loss_function(individual_loss_cfg)
    else:
        # If 'name' is not specified at all, it's ambiguous.
        # It's better to explicitly require 'name' for clarity or define a clear default.
        # For now, let's raise an error, forcing explicit configuration.
        # Alternatively, if you *must* have a default, decide what it is (e.g., 'mse').
        raise ValueError(
            "Loss configuration is missing a 'name' field. "
            "Please specify 'name: PoseLoss9D' or 'name: <individual_loss_type>'."
            f"Provided config: {loss_cfg_global}"
        )