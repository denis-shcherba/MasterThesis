import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import logging
import os # For path joining
from envs.create_env import ShelfPullDataCollector 
from models.policy_head.policy_network import create_model # From your project
from data_handling.processing import normalize_point_cloud_to_unit_sphere_torch, pose_9d_to_7d, pose_7d_to_9d
import time 

log = logging.getLogger(__name__)

def preprocess_inference_input(raw_input_data: dict, cfg: DictConfig, device: torch.device) -> dict:
    """
    Preprocesses a single raw input data point for inference.
    This function MUST replicate the transformations your Dataset applies.

    Args:
        raw_input_data: Dict with raw data. Expected keys depend on the model type.
                        e.g., {'point_cloud': np.array (N,3), 'robot_state': np.array (D,)}
        cfg: The Hydra configuration object.
        device: The torch device to move tensors to.

    Returns:
        A dictionary of processed tensors, ready for the model, with a batch dim of 1.
    """
    log.info("Preprocessing inference input...")
    processed_input = {}
    model_cfg = cfg.model
    data_cfg = cfg.get('data', {}) # Safely get data config

    # --- 1. Process Point Cloud (Common to both policy types) ---
    if 'point_cloud' not in raw_input_data:
        log.error("Missing 'point_cloud' in raw_input_data.")
        raise ValueError("Raw input data must contain 'point_cloud'.")

    point_cloud_raw = raw_input_data['point_cloud']
    if not isinstance(point_cloud_raw, np.ndarray):
        log.error(f"raw_input_data['point_cloud'] must be a NumPy array, got {type(point_cloud_raw)}")
        raise TypeError("Point cloud must be a NumPy array.")
    if point_cloud_raw.ndim != 2 or point_cloud_raw.shape[1] != 3:
        log.error(f"Point cloud shape invalid. Expected (N, 3), got {point_cloud_raw.shape}")
        raise ValueError("Point cloud must have shape (N, 3).")

    # Adjust number of points (padding/subsampling) - CRITICAL
    # This logic should mirror your training dataset's point adjustment
    num_model_points = model_cfg.num_points
    if point_cloud_raw.shape[0] > num_model_points:
        log.info(f"Subsampling point cloud from {point_cloud_raw.shape[0]} to {num_model_points} points.")
        # Example: random subsampling, or just take the first N
        indices = np.random.choice(point_cloud_raw.shape[0], num_model_points, replace=False)
        pc_processed_np = point_cloud_raw[indices, :]
    elif point_cloud_raw.shape[0] < num_model_points:
        log.info(f"Padding point cloud from {point_cloud_raw.shape[0]} to {num_model_points} points.")
        padding_needed = num_model_points - point_cloud_raw.shape[0]
        # Example: pad with zeros, or repeat last point (check training dataset's method)
        padding = np.zeros((padding_needed, 3), dtype=point_cloud_raw.dtype)
        pc_processed_np = np.concatenate([point_cloud_raw, padding], axis=0)
    else:
        pc_processed_np = point_cloud_raw

    pc_tensor = torch.from_numpy(pc_processed_np).float()

    # Apply normalization if configured (use the same function as in training)
    if data_cfg.get('normalize_points', False): # Check actual path in your config for this flag
        log.info("Normalizing point cloud to unit sphere...")
        # Ensure normalize_point_cloud_to_unit_sphere_torch is available and works on a single PC
        pc_tensor = normalize_point_cloud_to_unit_sphere_torch(pc_tensor)

    processed_input['point_cloud'] = pc_tensor.unsqueeze(0).to(device) # Shape: (1, num_model_points, 3)

    # --- 2. Process State (Only for multimodal policy if state_dim > 0) ---
    if model_cfg.type == 'multimodal':
        model_state_dim = model_cfg.get('state_dim', 0)
        if model_state_dim > 0:
            # Assuming 'robot_state' in raw_input_data maps to 'state' for the model
            if 'robot_state' not in raw_input_data:
                log.error("Multimodal policy expects 'robot_state' in raw_input_data but it's missing.")
                raise ValueError("'robot_state' missing for multimodal policy.")

            robot_state_raw = raw_input_data['robot_state']
            if not isinstance(robot_state_raw, np.ndarray):
                 raise TypeError("'robot_state' must be a NumPy array.")
            if robot_state_raw.ndim != 1 or robot_state_raw.shape[0] != model_state_dim:
                log.error(f"Dimension mismatch for 'robot_state'. Expected ({model_state_dim},), got {robot_state_raw.shape}")
                raise ValueError(f"Robot state dimension should be {model_state_dim}.")

            state_tensor = torch.from_numpy(robot_state_raw).float()
            processed_input['state'] = state_tensor.unsqueeze(0).to(device) # Shape: (1, model_state_dim)
        else:
            log.info("Multimodal policy, but model.state_dim is 0. No 'state' input will be prepared.")
    elif model_cfg.type == 'pointcloud' and model_cfg.get('state_dim', 0) > 0:
        log.warning(f"Policy type is '{model_cfg.type}' but model.state_dim ({model_cfg.get('state_dim')}) is > 0. "
                    "State input from 'robot_state' (if provided) will be ignored by PointCloudPolicy.")


    # --- 3. The DUMMY data generation block (reconsider its use here) ---
    # For actual inference, this block is problematic as it overwrites real data.
    # It's better to ensure the above steps correctly prepare all inputs.
    # If you want to keep it for some debugging or specific fallback, make it conditional and non-destructive.
    # For now, let's assume it should NOT run if we are processing real data.

    # Example: Only run if a special debug flag is set and no real data was processed
    # create_dummy_if_needed = cfg.get("debug_create_dummy_input", False)
    # if create_dummy_if_needed and not processed_input: # or some other condition
    if hasattr(cfg.model, 'input_features_eval'):
        log.warning("`cfg.model.input_features_eval` is defined. "
                    "If this block runs, IT MAY OVERWRITE your REAL processed data with DUMMY data. "
                    "This is generally UNDESIRABLE for actual inference.")
        # If you absolutely need this for some reason, ensure it only fills MISSING keys,
        # or is used in a completely separate mode.
        # for key, shape_without_batch in cfg.model.input_features_eval.items():
        #     if key not in processed_input: # Only if missing
        #         batch_shape = [1] + list(shape_without_batch)
        #         log.info(f"Creating DUMMY input for missing key '{key}'.")
        #         processed_input[key] = torch.randn(batch_shape, device=device)
        #     else:
        #         log.info(f"Key '{key}' from input_features_eval already processed from real data. Skipping dummy generation.")

    # --- 4. Final Checks & Logging ---
    # Verify that all necessary inputs for the selected model type are present.
    required_keys = set(['point_cloud'])
    if model_cfg.type == 'multimodal' and model_cfg.get('state_dim', 0) > 0:
        required_keys.add('state')

    missing_keys = required_keys - set(processed_input.keys())
    if missing_keys:
        log.error(f"Missing required model inputs after preprocessing: {missing_keys}")
        raise RuntimeError(f"Preprocessing failed to produce all required inputs for model type '{model_cfg.type}'.")

    extra_keys = set(processed_input.keys()) - required_keys
    if extra_keys:
        log.warning(f"Extra keys found in processed_input not strictly required by model's forward args: {extra_keys}")


    log.info(f"Preprocessing complete. Final processed input keys: {list(processed_input.keys())}")
    for key, value in processed_input.items():
        log.info(f"  Input '{key}' shape: {value.shape}, device: {value.device}")

    return processed_input


@hydra.main(config_path="configs", config_name="inference", version_base=None)
def eval_policy(cfg: DictConfig) -> None:
    """
    Main evaluation/inference function for the manipulation policy.
    Args:
        cfg: Hydra configuration object.
    """

    log.info("Starting policy evaluation/inference...")
    log.info(f"Using experiment config: {cfg.experiment_name}")
    # log.info(f"Full config: {OmegaConf.to_yaml(cfg)}") # For debugging

    # --- 1. Setup Device and Environment---
    default_device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device_str = cfg.get("inference", {}).get("device", default_device_str)
    device = torch.device(device_str)
    log.info(f"Using device: {device}")

    collector = ShelfPullDataCollector(**cfg.env)
    collector.spawn_books_scene()
    collector.C.view(True)
    collector.C.view(False) 

    # --- 2. Initialize Model ---
    log.info("Initializing model...")
    model = create_model(cfg.model).to(device)
    log.info(f"Model created: {type(model).__name__}")

    # --- 3. Load Trained Weights ---
    # Determine checkpoint path.
    # Option 1: Directly in config (e.g., cfg.inference.checkpoint_path)
    # Option 2: Construct from training output directory (more robust if following a pattern)
    checkpoint_path_cfg = cfg.get("inference", {}).get("checkpoint_path", None)
    if checkpoint_path_cfg is None:
        log.error("Checkpoint path not found. Please specify `inference.checkpoint_path` in your config or command line.")
        log.error("Example: python eval_policy.py inference.checkpoint_path=outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/model_epoch_N.pth")
        return

    # Ensure the path is absolute or correctly relative if hydra changes CWD
    if not os.path.isabs(checkpoint_path_cfg) and hydra.utils.get_original_cwd() != os.getcwd():
        checkpoint_path = os.path.join(hydra.utils.get_original_cwd(), checkpoint_path_cfg)
    else:
        checkpoint_path = checkpoint_path_cfg

    log.info(f"Loading model checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        log.error(f"Checkpoint file not found at {checkpoint_path}")
        log.error(f"Original CWD: {hydra.utils.get_original_cwd()}, Current CWD: {os.getcwd()}")
        return

    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)['model_state_dict']
        
        # Handle potential 'module.' prefix if DataParallel was used during training
        if any(key.startswith('module.') for key in state_dict.keys()):
            log.info("Removing 'module.' prefix from state_dict keys (model likely trained with DataParallel).")
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        log.info("Model weights loaded successfully.")
    except Exception as e:
        log.error(f"Error loading checkpoint: {e}")
        raise

    hidden_state = None 
    policy_type = cfg.get("model").get("type")

    model.eval() # IMPORTANT: Set the model to evaluation mode

    log.info("Model set to evaluation mode.")

    for i in range (64):
        # --- 4. Prepare Input Data for Inference ---
        pc = collector.render()
        robot_state = None
        #if cfg.get() 
        if cfg["model"]["action_dim"] == 9:
            robot_state = pose_7d_to_9d(collector.C.getJointState())
        elif cfg["model"]["action_dim"] == 3:
            # if floatin
            robot_state = collector.C.getJointState()[:3] 
        raw_inference_data = {"point_cloud": pc, "robot_state": robot_state}

        input_for_model = preprocess_inference_input(raw_inference_data, cfg, device)

        if not input_for_model:
            log.error("Preprocessing did not return any data. Aborting.")
            return

        # --- 5. Perform Inference ---
        log.info("Running model inference...")
        with torch.no_grad(): # Disable gradient calculations
            try:

                if policy_type == "multimodal":
                    if cfg.get("model").get("policy_head_type") == "gru":
                        output, hidden_state = model(
                            input_for_model["point_cloud"], 
                            input_for_model["state"], 
                            torch.tensor(i).reshape(1),
                            hidden_state=hidden_state
                        )
                    elif cfg.get("model").get("policy_head_type") == "mlp":
                        output = model(input_for_model["point_cloud"], input_for_model["state"], torch.tensor(i).reshape(1))
                elif policy_type == "regression":
                    output = model(input_for_model["point_cloud"])

            except Exception as e:
                log.error(f"Error during model forward pass: {e}")
                log.error("Ensure the `input_for_model` structure and tensor shapes/types match your model's `forward` method.")
                log.error(f"Input keys: {input_for_model.keys()}")
                for k, v in input_for_model.items():
                    if isinstance(v, torch.Tensor):
                        log.error(f"  {k}: shape {v.shape}, dtype {v.dtype}, device {v.device}")
                raise

        log.info(f"Inference output raw: {output}")

        if cfg["model"]["type"] == "regression":
            path_dataset = output.squeeze().cpu().numpy()

            collector.C.setJointState([path_dataset[0], path_dataset[1], path_dataset[2], 1, 0, 0, 0])  # Assuming the first 7 values are joint angles
            collector.C.view(True)
        if cfg["model"]["action_dim"] == 9:
            pose7d = pose_9d_to_7d(output.squeeze().cpu().numpy())
            log.info("pose7d:", pose7d)
            
            collector.C.setJointState(pose7d)
        elif cfg["model"]["action_dim"] == 3:
            if cfg["env"]["path_mode"] == "DELTA3D":
                delta_pos = output.squeeze().cpu().numpy()
                collector.C.setJointState(collector.C.getJointState() + np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0, 0, 0, 0]))

            else:   
                pos = output.squeeze().cpu().numpy()
                collector.C.setJointState(np.array([pos[0], pos[1], pos[2], 1, 0, 0, 0])) # Assuming a fixed orientation for the gripper


        
        collector.C.view(False)
        time.sleep(.3)
        if isinstance(output, torch.Tensor):
            log.info(f"Output tensor shape: {output.shape}")
        elif isinstance(output, dict):
            log.info(f"Output dictionary keys: {output.keys()}")
            for k,v in output.items():
                if isinstance(v, torch.Tensor):
                    log.info(f"  {k} shape: {v.shape}")


    log.info("Policy evaluation/inference finished.")


if __name__ == "__main__":
    eval_policy()