import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import logging
import os # For path joining
from envs.create_env import ShelfPullDataCollector 
from models.policy_head.policy_network import create_model # From your project
from data_handling.processing import normalize_point_cloud_to_unit_sphere_torch, pose_9d_to_7d, pose_7d_to_9d
from utils.data_utils import normalize_depth, normalize_state, denormalize_actions
import yaml
import robotic as ry

log = logging.getLogger(__name__)
SIMULATE = True  # Set to False if you don't want to simulate the environment

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

    pc_tensor = torch.from_numpy(point_cloud_raw).float()

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


    # --- 3. Final Checks & Logging ---
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
    #collector.C.view(True)
    collector.C.view(False) 
    q0 = collector.C.getJointState()

    # --- 2. Initialize Model ---
    log.info("Initializing model...")
    model = create_model(cfg.model).to(device)
    log.info(f"Model created: {type(model).__name__}")

    # --- 3. Load Trained Weights ---
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

    normalization_stats_path = cfg.get("inference", {}).get("normalization_stats_path", None)
    if normalization_stats_path is None:
        log.error("normalization_stats_path path not found. Please specify `inference.normalization_stats_path` in your config or command line.")
        return

    with open(normalization_stats_path, 'r') as file:
        normalization_stats = yaml.safe_load(file)

    print("\nAction Stats:")
    print(normalization_stats['action_stats'])
    
    print("\nDepth Stats:")
    print(normalization_stats['depth_stats'])

    hidden_state = None 
    policy_type = cfg.get("model").get("type")

    model.eval() # IMPORTANT: Set the model to evaluation mode

    log.info("Model set to evaluation mode.")

    depth_sequence = []
    state_sequence = []
    sequence_length = 10

    if SIMULATE:
        sim = ry.Simulation(collector.C, ry.SimulationEngine.physx, verbose=0)

    for i in range(64):
        # --- 4. Prepare Input Data for Inference ---
        pc = collector.render(observation_mode="POINTCLOUD", n_samples=cfg.get("data", {}).get("num_points", 4096))
        depth_ = collector.render(observation_mode=cfg.get("observation_mode"), n_samples=cfg.get("data", {}).get("num_points", 0))
        depth = torch.from_numpy(depth_).float().to(device)  # [H, W]
        depth = depth.unsqueeze(0)  # [1, H, W]

        # ✅ Normalize depth only for Transformer
        if cfg["model"]["policy_head_type"] == "transformer":
            depth = normalize_depth(depth, normalization_stats["depth_stats"])

        robot_state = None
        if cfg["model"]["action_dim"] == 9:
            robot_state = pose_7d_to_9d(collector.C.getJointState())
        elif cfg["model"]["action_dim"] == 3:
            robot_state = collector.C.getJointState()[:3]

        raw_inference_data = {"point_cloud": pc, "robot_state": robot_state}
        input_for_model = preprocess_inference_input(raw_inference_data, cfg, device)

        if not input_for_model:
            log.error("Preprocessing did not return any data. Aborting.")
            return

        # --- Maintain history buffers ---
        depth_sequence.append(depth)  # [1, H, W]

        state_tensor = torch.tensor(input_for_model["state"], dtype=torch.float32, device=device).unsqueeze(0)  # [1, 3]
        
        # ✅ Normalize state only for Transformer
        if cfg["model"]["policy_head_type"] == "transformer":
            state_tensor = normalize_state(state_tensor, normalization_stats["action_stats"])
        
        state_sequence.append(state_tensor)

        # Trim and pad (same as before)
        if len(depth_sequence) > sequence_length:
            depth_sequence = depth_sequence[-sequence_length:]
            state_sequence = state_sequence[-sequence_length:]

        num_pad = sequence_length - len(depth_sequence)
        if num_pad > 0:
            dummy_depth = torch.zeros_like(depth)
            dummy_state = torch.zeros_like(state_tensor)
            depth_sequence = [dummy_depth] * num_pad + depth_sequence
            state_sequence = [dummy_state] * num_pad + state_sequence

        depth_seq = torch.stack(depth_sequence, dim=0).squeeze(1).unsqueeze(0)  # [1, seq_len, H, W]
        state_seq = torch.stack(state_sequence, dim=0).squeeze(1).unsqueeze(0)  # [1, seq_len, 3]

        # --- 5. Perform Inference ---
        log.info(f"Running model inference for step {i}...")
        with torch.no_grad():
            try:
                if policy_type == "multimodal":
                    policy_head_type = cfg.get("model").get("policy_head_type")
                    if policy_head_type == "gru":
                        ...
                    elif policy_head_type == "mlp":
                        ...
                    elif policy_head_type == "transformer":
                        log.debug("Running Transformer (Depth) policy inference.")
                        output = model(
                            depth_seq,  # Normalized
                            state_seq   # Normalized
                        )

                    else:
                        raise NotImplementedError(f"Policy head type '{policy_head_type}' not recognized for multimodal policy.")



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
            
            collector.C.setJointState([path_dataset[0], path_dataset[1], path_dataset[2], 1, 0, 0, 0])
            collector.C.view(True)

            collector.C.setJointState(q0)
            collector.C.delFrame("target_book_0")
            collector.C.view(False)

            collector.spawn_books_scene()
            collector.C.view(True)

            
        if cfg["model"]["action_dim"] == 9:
            pose7d = pose_9d_to_7d(output.squeeze().cpu().numpy())
            log.info(f"Computed 7D pose: {pose7d}")
            collector.C.setJointState(pose7d)
        elif cfg["model"]["action_dim"] == 3:
            if cfg["env"]["path_mode"] == "DELTA3D":
                delta_pos = output.squeeze().cpu().numpy()
                current_pose = collector.C.getJointState()
                new_pose = current_pose + np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0, 0, 0, 0])
                collector.C.setJointState(new_pose)

            else:   
                if cfg.get("model").get("policy_head_type") == "transformer":
                    output = denormalize_actions(output, normalization_stats["action_stats"])
                pos = output.squeeze().cpu().numpy()
                # Assuming a fixed orientation for the gripper
                if not SIMULATE:
                    collector.C.setJointState(np.array([pos[0], pos[1], pos[2], 1, 0, 0, 0]))
                else:
                    if i > 31:
                        pos[2] -= .001
                    for i in range(350):
                        sim.step([pos[0], pos[1], pos[2], 1, 0, 0, 0], 0.01, ry.ControlMode.position)


        collector.C.view(False)
        if isinstance(output, torch.Tensor):
            log.info(f"Output tensor shape: {output.shape}")
        elif isinstance(output, dict):
            log.info(f"Output dictionary keys: {output.keys()}")
            for k,v in output.items():
                if isinstance(v, torch.Tensor):
                    log.info(f"  {k} shape: {v.shape}")

    log.info("Policy evaluation/inference finished.")


if __name__ == "__main__":
    # Note: For this to run, you need to have the rest of the project
    # structure (configs, models, etc.) in place.
    eval_policy()