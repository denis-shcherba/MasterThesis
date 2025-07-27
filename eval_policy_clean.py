#TODO cleanup, use gym env

import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import logging
import os
from envs.create_env import ShelfPullDataCollector 
from models.policy_head.policy_network import create_model
from data_handling.processing import normalize_point_cloud_to_unit_sphere_torch, pose_9d_to_7d, pose_7d_to_9d
import time 

log = logging.getLogger(__name__)

def get_observation_data(collector, cfg):
    """Get observation data based on configured observation mode."""
    obs_mode = cfg.get("observation_mode", "POINTCLOUD")
    num_points = cfg.get("data", {}).get("num_points", 4096)
    
    if obs_mode == "pointcloud":
        return collector.render(observation_mode="POINTCLOUD", n_samples=num_points)
    elif obs_mode in ["depth", "rgb_depth"]:
        return collector.render(observation_mode=obs_mode, n_samples=num_points)
    else:
        log.warning(f"Unknown observation mode: {obs_mode}, defaulting to POINTCLOUD")
        return collector.render(observation_mode="POINTCLOUD", n_samples=num_points)

def get_robot_state(collector, cfg):
    """Get robot state based on action dimension configuration."""
    action_dim = cfg["model"]["action_dim"]
    
    if action_dim == 9:
        return pose_7d_to_9d(collector.C.getJointState())
    elif action_dim == 3:
        return collector.C.getJointState()[:3]
    else:
        log.warning(f"Unsupported action_dim: {action_dim}")
        return collector.C.getJointState()

def preprocess_inference_input(raw_input_data: dict, cfg: DictConfig, device: torch.device) -> dict:
    """
    Preprocesses a single raw input data point for inference.
    This function replicates the transformations your Dataset applies.
    """
    log.info("Preprocessing inference input...")
    processed_input = {}
    model_cfg = cfg.model
    data_cfg = cfg.get('data', {})

    # Process observation data (point cloud or depth)
    obs_mode = cfg.get("observation_mode", "POINTCLOUD")
    
    if obs_mode == "pointcloud":
        if 'point_cloud' not in raw_input_data:
            raise ValueError("Raw input data must contain 'point_cloud' for POINTCLOUD mode.")
        
        point_cloud_raw = raw_input_data['point_cloud']
        if not isinstance(point_cloud_raw, np.ndarray):
            raise TypeError("Point cloud must be a NumPy array.")
        if point_cloud_raw.ndim != 2 or point_cloud_raw.shape[1] != 3:
            raise ValueError(f"Point cloud must have shape (N, 3), got {point_cloud_raw.shape}")

        pc_tensor = torch.from_numpy(point_cloud_raw).float()
        
        # Apply normalization if configured
        if data_cfg.get('normalize_points', False):
            log.info("Normalizing point cloud to unit sphere...")
            pc_tensor = normalize_point_cloud_to_unit_sphere_torch(pc_tensor)

        processed_input['point_cloud'] = pc_tensor.unsqueeze(0).to(device)
        
    elif obs_mode in ["depth", "rgb_depth"]:
        if 'depth' not in raw_input_data:
            raise ValueError(f"Raw input data must contain 'depth' for {obs_mode} mode.")
        
        depth_raw = raw_input_data['depth']
        if isinstance(depth_raw, np.ndarray):
            depth_tensor = torch.from_numpy(depth_raw).float()
        else:
            depth_tensor = depth_raw.float()
        
        processed_input['depth'] = depth_tensor.unsqueeze(0).to(device)

    # Process robot state for multimodal models
    if model_cfg.type == 'multimodal':
        model_state_dim = model_cfg.get('state_dim', 0)
        if model_state_dim > 0:
            if 'robot_state' not in raw_input_data:
                raise ValueError("'robot_state' missing for multimodal policy.")

            robot_state_raw = raw_input_data['robot_state']
            if not isinstance(robot_state_raw, np.ndarray):
                raise TypeError("'robot_state' must be a NumPy array.")
            if robot_state_raw.ndim != 1 or robot_state_raw.shape[0] != model_state_dim:
                raise ValueError(f"Robot state dimension should be {model_state_dim}, got {robot_state_raw.shape}")

            state_tensor = torch.from_numpy(robot_state_raw).float()
            processed_input['state'] = state_tensor.unsqueeze(0).to(device)

    # Verify required inputs are present
    required_keys = set()
    if obs_mode == "pointcloud":
        required_keys.add('point_cloud')
    elif obs_mode in ["depth", "rgb_depth"]:
        required_keys.add('depth')
    
    if model_cfg.type == 'multimodal' and model_cfg.get('state_dim', 0) > 0:
        required_keys.add('state')

    missing_keys = required_keys - set(processed_input.keys())
    if missing_keys:
        raise RuntimeError(f"Missing required model inputs: {missing_keys}")

    log.info(f"Preprocessing complete. Input keys: {list(processed_input.keys())}")
    for key, value in processed_input.items():
        log.info(f"  {key} shape: {value.shape}, device: {value.device}")

    return processed_input

def run_model_inference(model, input_data, cfg, step_idx, hidden_state=None):
    """Run model inference based on model type and configuration."""
    policy_type = cfg.model.type
    policy_head_type = cfg.model.get("policy_head_type", "mlp")
    obs_mode = cfg.get("observation_mode", "POINTCLOUD")
    
    with torch.no_grad():
        if policy_type == "multimodal":
            if policy_head_type == "gru":
                # For GRU-based policies
                if obs_mode == "POINTCLOUD":
                    output, hidden_state = model(
                        input_data["point_cloud"], 
                        input_data["state"], 
                        torch.tensor(step_idx).reshape(1).to(input_data["point_cloud"].device),
                        hidden_state=hidden_state
                    )
                else:  # DEPTH mode
                    output, hidden_state = model(
                        input_data["depth"], 
                        input_data["state"], 
                        torch.tensor(step_idx).reshape(1).to(input_data["depth"].device),
                        hidden_state=hidden_state
                    )
            else:  # MLP-based policies
                if obs_mode == "POINTCLOUD":
                    output = model(
                        input_data["point_cloud"], 
                        input_data["state"], 
                        torch.tensor(step_idx).reshape(1).to(input_data["point_cloud"].device)
                    )
                else:  # DEPTH mode
                    output = model(
                        input_data["depth"], 
                        input_data["state"], 
                        torch.tensor(step_idx).reshape(1).to(input_data["depth"].device)
                    )
                    
        elif policy_type == "regression":
            if obs_mode == "POINTCLOUD":
                output = model(input_data["point_cloud"])
            else:  # DEPTH mode
                output = model(input_data["depth"])
        else:
            raise ValueError(f"Unsupported policy type: {policy_type}")
    
    return output, hidden_state

def execute_action(collector, output, cfg, q0):
    """Execute the predicted action in the environment."""
    action_dim = cfg["model"]["action_dim"]
    model_type = cfg["model"]["type"]
    
    # Convert output to numpy
    if isinstance(output, torch.Tensor):
        action = output.squeeze().cpu().numpy()
    else:
        action = output
    
    log.info(f"Predicted action: {action}")
    
    if model_type == "regression":
        # For regression models, set absolute position
        collector.C.setJointState([action[0], action[1], action[2], 1, 0, 0, 0])
        collector.C.view(True)
        
        # Reset for next iteration
        collector.C.setJointState(q0)
        collector.C.delFrame("target_book_0")
        collector.C.view(False)
        collector.spawn_books_scene()
        collector.C.view(True)
        
    elif action_dim == 9:
        # Convert 9D pose to 7D
        pose7d = pose_9d_to_7d(action)
        log.info(f"Converted to 7D pose: {pose7d}")
        collector.C.setJointState(pose7d)
        
    elif action_dim == 3:
        path_mode = cfg.get("env", {}).get("path_mode", "ABSOLUTE3D")
        
        if path_mode == "DELTA3D":
            # Apply delta position
            current_state = collector.C.getJointState()
            delta_state = np.array([action[0], action[1], action[2], 0, 0, 0, 0])
            new_state = current_state + delta_state
            collector.C.setJointState(new_state)
        else:  # ABSOLUTE3D
            # Set absolute position with fixed orientation
            collector.C.view(False)

            collector.C.setJointState(np.array([action[0], action[1], action[2], 1, 0, 0, 0]))

def load_model_checkpoint(model, checkpoint_path, device):
    """Load model checkpoint with proper error handling."""
    log.info(f"Loading model checkpoint from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        
        # Handle potential 'module.' prefix from DataParallel training
        if any(key.startswith('module.') for key in state_dict.keys()):
            log.info("Removing 'module.' prefix from state_dict keys")
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        log.info("Model weights loaded successfully")
        
    except Exception as e:
        log.error(f"Error loading checkpoint: {e}")
        raise

@hydra.main(config_path="configs", config_name="inference", version_base=None)
def eval_policy(cfg: DictConfig) -> None:
    """Main evaluation/inference function for the manipulation policy."""
    log.info("Starting policy evaluation/inference...")
    log.info(f"Using experiment config: {cfg.experiment_name}")

    # Setup device
    default_device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device_str = cfg.get("inference", {}).get("device", default_device_str)
    device = torch.device(device_str)
    log.info(f"Using device: {device}")

    # Initialize environment
    collector = ShelfPullDataCollector(**cfg.env)
    collector.spawn_books_scene()
    collector.C.view(False) 
    q0 = collector.C.getJointState()

    # Initialize model
    log.info("Initializing model...")
    model = create_model(cfg.model).to(device)
    log.info(f"Model created: {type(model).__name__}")

    # Load checkpoint
    checkpoint_path_cfg = cfg.get("inference", {}).get("checkpoint_path", None)
    if checkpoint_path_cfg is None:
        log.error("Checkpoint path not found. Please specify `inference.checkpoint_path` in your config.")
        return

    # Handle relative paths with Hydra
    if not os.path.isabs(checkpoint_path_cfg) and hydra.utils.get_original_cwd() != os.getcwd():
        checkpoint_path = os.path.join(hydra.utils.get_original_cwd(), checkpoint_path_cfg)
    else:
        checkpoint_path = checkpoint_path_cfg

    load_model_checkpoint(model, checkpoint_path, device)
    model.eval()
    log.info("Model set to evaluation mode")

    # Initialize variables for sequential policies
    hidden_state = None
    num_episodes = cfg.get("inference", {}).get("num_episodes", 64)
    
    # Main evaluation loop
    for episode in range(num_episodes):
        log.info(f"Running episode {episode + 1}/{num_episodes}")
        
        try:
            # Get observations
            obs_data = get_observation_data(collector, cfg)
            robot_state = get_robot_state(collector, cfg)
            
            # Prepare input data
            obs_mode = cfg.get("observation_mode", "POINTCLOUD")
            if obs_mode == "POINTCLOUD":
                raw_input_data = {"point_cloud": obs_data, "robot_state": robot_state}
            else:  # DEPTH mode
                raw_input_data = {"depth": obs_data, "robot_state": robot_state}
            
            # Preprocess input
            input_for_model = preprocess_inference_input(raw_input_data, cfg, device)
            
            # Run inference
            output, hidden_state = run_model_inference(
                model, input_for_model, cfg, episode, hidden_state
            )
            
            # Execute action
            execute_action(collector, output, cfg, q0)
            
            # Brief pause for visualization
            time.sleep(0.01)
            
        except Exception as e:
            log.error(f"Error in episode {episode}: {e}")
            if cfg.get("inference", {}).get("stop_on_error", True):
                raise
            else:
                continue

    log.info("Policy evaluation/inference finished")

if __name__ == "__main__":
    eval_policy()