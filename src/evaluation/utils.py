# evaluation/utils.py
import torch
import numpy as np
import logging
from omegaconf import DictConfig

from data_handling.processing import normalize_point_cloud_to_unit_sphere_torch, pose_9d_to_7d

log = logging.getLogger(__name__)

class InferencePreprocessor:
    """Prepares raw observation data into a batch of tensors for the model."""
    def __init__(self, model_cfg: DictConfig, data_cfg: DictConfig, device: torch.device):
        self.device = device
        self.model_type = model_cfg.type
        self.num_points = model_cfg.num_points
        self.state_dim = model_cfg.get('state_dim', 0)
        self.normalize_points = data_cfg.get('normalize_points', False)
        log.info(f"Preprocessor initialized for '{self.model_type}' model.")

    def process(self, raw_obs: dict) -> dict:
        """Processes a single raw observation dictionary."""
        processed_input = {}

        # --- 1. Process Point Cloud ---
        pc_processed = self._process_point_cloud(raw_obs['point_cloud'])
        processed_input['point_cloud'] = pc_processed.unsqueeze(0).to(self.device)

        # --- 2. Process State (if applicable) ---
        if self.model_type == 'multimodal' and self.state_dim > 0:
            state_processed = self._process_state(raw_obs['robot_state'])
            processed_input['state'] = state_processed.unsqueeze(0).to(self.device)

        return processed_input

    def _process_point_cloud(self, pc_raw: np.ndarray) -> torch.Tensor:
        """Subsamples/pads, normalizes, and converts a point cloud to a tensor."""
        if pc_raw.shape[0] > self.num_points:
            indices = np.random.choice(pc_raw.shape[0], self.num_points, replace=False)
            pc_np = pc_raw[indices, :]
        elif pc_raw.shape[0] < self.num_points:
            padding = np.zeros((self.num_points - pc_raw.shape[0], 3), dtype=pc_raw.dtype)
            pc_np = np.concatenate([pc_raw, padding], axis=0)
        else:
            pc_np = pc_raw

        pc_tensor = torch.from_numpy(pc_np).float()

        if self.normalize_points:
            pc_tensor = normalize_point_cloud_to_unit_sphere_torch(pc_tensor)
            
        return pc_tensor

    def _process_state(self, state_raw: np.ndarray) -> torch.Tensor:
        """Validates and converts a robot state array to a tensor."""
        if state_raw.shape[0] != self.state_dim:
            raise ValueError(f"Robot state dim mismatch. Expected {self.state_dim}, got {state_raw.shape[0]}")
        return torch.from_numpy(state_raw).float()


class ActionPostprocessor:
    """Converts a model's output tensor into an executable environment action."""
    def __init__(self, model_cfg: DictConfig, env_cfg: DictConfig):
        self.action_dim = model_cfg.action_dim
        self.path_mode = env_cfg.get("path_mode", "ABSOLUTE")
        log.info(f"Postprocessor initialized for action_dim={self.action_dim}, path_mode='{self.path_mode}'.")

    def process(self, model_output: torch.Tensor, current_pose_7d: np.ndarray) -> np.ndarray:
        """Processes the model output tensor into a 7D robot pose."""
        action_np = model_output.squeeze().cpu().numpy()

        if self.action_dim == 9:
            return pose_9d_to_7d(action_np)
        
        if self.action_dim == 3:
            if self.path_mode == "DELTA3D":
                # Create a delta for the 7D pose (3D position change, 0 rotation change)
                delta_pose = np.zeros(7)
                delta_pose[:3] = action_np
                return current_pose_7d + delta_pose
            else: # Absolute position
                # Return new pose with new position and original orientation
                new_pose = current_pose_7d.copy()
                new_pose[:3] = action_np
                return new_pose
        
        # Handle the direct regression case
        if self.action_dim == 7 or self.action_dim == 3: # Assuming regression outputs absolute pose/position
             new_pose = np.zeros(7)
             new_pose[:action_np.shape[0]] = action_np
             # Fill orientation with default if only position is predicted
             if action_np.shape[0] == 3:
                 new_pose[3:] = [1, 0, 0, 0] # Default quaternion
             return new_pose

        raise NotImplementedError(f"Post-processing not implemented for action_dim={self.action_dim}")