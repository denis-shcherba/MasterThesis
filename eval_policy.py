import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import logging
import os
from envs.create_env import ShelfPullDataCollector
from models.policy_head.policy_network import create_model
from data_handling.processing import pose_9d_to_7d, pose_7d_to_9d
from utils.data_utils import normalize_depth, normalize_state, denormalize_actions
import yaml
import robotic as ry

log = logging.getLogger(__name__)
SIMULATE = True

def preprocess_inference_input(raw_input_data: dict, cfg: DictConfig, device: torch.device) -> dict:
    """
    Preprocesses a single raw input data point for inference without point clouds.
    """
    log.info("Preprocessing inference input...")
    processed_input = {}
    model_cfg = cfg.model

    # --- Process State (Only if state_dim > 0) ---
    if model_cfg.get('state_dim', 0) > 0:
        if 'robot_state' not in raw_input_data:
            log.error("Expected 'robot_state' in raw_input_data but it's missing.")
            raise ValueError("'robot_state' missing.")

        robot_state_raw = raw_input_data['robot_state']
        if not isinstance(robot_state_raw, np.ndarray):
            raise TypeError("'robot_state' must be a NumPy array.")
        if robot_state_raw.ndim != 1 or robot_state_raw.shape[0] != model_cfg.get('state_dim'):
            log.error(f"Dimension mismatch for 'robot_state'. Expected ({model_cfg.get('state_dim')},), got {robot_state_raw.shape}")
            raise ValueError(f"Robot state dimension should be {model_cfg.get('state_dim')}.")

        state_tensor = torch.from_numpy(robot_state_raw).float()
        processed_input['state'] = state_tensor.unsqueeze(0).to(device)

    # --- Final Checks ---
    required_keys = set()
    if model_cfg.get('state_dim', 0) > 0:
        required_keys.add('state')

    missing_keys = required_keys - set(processed_input.keys())
    if missing_keys:
        log.error(f"Missing required model inputs: {missing_keys}")
        raise RuntimeError("Preprocessing failed to produce required inputs.")

    log.info(f"Preprocessing complete. Final keys: {list(processed_input.keys())}")
    return processed_input


@hydra.main(config_path="configs", config_name="inference", version_base=None)
def eval_policy(cfg: DictConfig) -> None:
    log.info("Starting policy evaluation/inference...")
    log.info(f"Using experiment config: {cfg.experiment_name}")

    device_str = cfg.get("inference", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    log.info(f"Using device: {device}")

    collector = ShelfPullDataCollector(**cfg.env)
    collector.spawn_books_scene()
    collector.C.view(False)
    q0 = collector.C.getJointState()

    # Model
    log.info("Initializing model...")
    model = create_model(cfg.model).to(device)
    log.info(f"Model created: {type(model).__name__}")

    # Load checkpoint
    checkpoint_path_cfg = cfg.get("inference", {}).get("checkpoint_path", None)
    if checkpoint_path_cfg is None:
        log.error("Checkpoint path not found.")
        return

    if not os.path.isabs(checkpoint_path_cfg) and hydra.utils.get_original_cwd() != os.getcwd():
        checkpoint_path = os.path.join(hydra.utils.get_original_cwd(), checkpoint_path_cfg)
    else:
        checkpoint_path = checkpoint_path_cfg

    if not os.path.exists(checkpoint_path):
        log.error(f"Checkpoint file not found at {checkpoint_path}")
        return

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)['model_state_dict']
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    log.info("Model weights loaded successfully.")

    normalization_stats_path = cfg.get("inference", {}).get("normalization_stats_path", None)
    if normalization_stats_path is None:
        log.error("normalization_stats_path not found.")
        return

    with open(normalization_stats_path, 'r') as file:
        normalization_stats = yaml.safe_load(file)

    print("\nAction Stats:")
    print(normalization_stats['action_stats'])
    print("\nDepth Stats:")
    print(normalization_stats['depth_stats'])

    model.eval()
    log.info("Model set to evaluation mode.")

    depth_sequence = []
    state_sequence = []
    sequence_length = cfg.get("data", {}).get("sequence_length", 0)


    if SIMULATE:
        sim = ry.Simulation(collector.C, ry.SimulationEngine.physx, verbose=0)

    for i in range(64):
        depth_ = collector.render(
            observation_mode=cfg.get("observation_mode"),
        )
        depth = torch.from_numpy(depth_).float().to(device).unsqueeze(0)

        if cfg["model"]["policy_head_type"] == "transformer":
            depth = normalize_depth(depth, normalization_stats["depth_stats"])

        if cfg["model"]["action_dim"] == 9:
            robot_state = pose_7d_to_9d(collector.C.getJointState())
        elif cfg["model"]["action_dim"] == 3:
            robot_state = collector.C.getJointState()[:3]

        raw_inference_data = {"robot_state": robot_state}
        input_for_model = preprocess_inference_input(raw_inference_data, cfg, device)

        depth_sequence.append(depth)
        state_tensor = torch.tensor(input_for_model["state"], dtype=torch.float32, device=device).unsqueeze(0)

        if cfg["model"]["policy_head_type"] == "transformer":
            state_tensor = normalize_state(state_tensor, normalization_stats["action_stats"])

        state_sequence.append(state_tensor)

        if len(depth_sequence) > sequence_length:
            depth_sequence = depth_sequence[-sequence_length:]
            state_sequence = state_sequence[-sequence_length:]

        num_pad = sequence_length - len(depth_sequence)
        if num_pad > 0:
            dummy_depth = torch.zeros_like(depth)
            dummy_state = torch.zeros_like(state_tensor)
            depth_sequence = [dummy_depth] * num_pad + depth_sequence
            state_sequence = [dummy_state] * num_pad + state_sequence

        depth_seq = torch.stack(depth_sequence, dim=0).squeeze(1).unsqueeze(0)
        state_seq = torch.stack(state_sequence, dim=0).squeeze(1).unsqueeze(0)

        log.info(f"Running model inference for step {i}...")
        with torch.no_grad():
            output = model(depth_seq, state_seq)

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
                if not SIMULATE:
                    collector.C.setJointState(np.array([pos[0], pos[1], pos[2], 1, 0, 0, 0]))
                else:
                    for _ in range(50):
                        sim.step([pos[0], pos[1], pos[2], 1, 0, 0, 0], 0.01, ry.ControlMode.position)

        collector.C.view(False)
        log.info(f"Output tensor shape: {output.shape if isinstance(output, torch.Tensor) else 'dict'}")

    log.info("Policy evaluation/inference finished.")


if __name__ == "__main__":
    eval_policy()
