import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import logging
import os
from models.policy_head.policy_network import create_model
from data_handling.processing import pose_9d_to_7d, pose_7d_to_9d
from utils.data_utils import get_pc_from_depth, normalize_depth, normalize_state, denormalize_actions, get_cls_features, get_patch_features, get_sam_pointcloud
import yaml
import json
from hydra.core.hydra_config import HydraConfig
import robotic as ry
import gymnasium as gym
import envs  # noqa: F401  
import matplotlib
import matplotlib.pyplot as plt
from envs.high_level_methods import RobotEnviroment
from envs.utils import point_in_box_filtering, sample_points

log = logging.getLogger(__name__)
DEBUG_DEPTH = False
DEBUG_STATE = False

def show_state_input_seq(cfg, env, state_input_seq, color=[1, 0, 0, .9], prefix=""):
    for name in env.unwrapped.C.getFrameNames():
        if name.startswith(prefix+"previous_pos_"):
            env.unwrapped.C.delFrame(name)
    if state_input_seq.shape[1] == 3:
        for i in range(state_input_seq.shape[0]):
            # Use the same reverse indexing logic as your first function
            previous_pos = state_input_seq[-(i + 1)].cpu().numpy()
            env.unwrapped.C.addFrame(prefix+f"previous_pos_{i}").setPosition(previous_pos).setShape(ry.ST.sphere, [.015]).setColor(color)
            print("Previous Position:", previous_pos)
    elif state_input_seq.shape[1] == 7:
        C2 = ry.Config()
        C2.addConfigurationCopy(env.unwrapped.C)
        for i in range(state_input_seq.shape[0]):
            C2.setJointState(state_input_seq[-(i + 1)].cpu().numpy())

            env.unwrapped.C.addFrame(prefix+f"previous_pos_{i}").setPosition(C2.eval(ry.FS.position, ["l_gripper"])[0]).setShape(ry.ST.sphere, [.015]).setColor(color)

    env.unwrapped.C.view(True)

def test_depth_sequence(depth_batch):
    # take the first entry in the batch -> shape [10, 96, 96]
    for i in range(depth_batch.shape[0]):

        depth_seq = depth_batch[i]
        # create a figure with 2 rows x 4 cols for the 8 images
        # TODO handle different sequence lengths
        fig, axes = plt.subplots(2, 4, figsize=(15, 6))

        for j, ax in enumerate(axes.flat):
            ax.imshow(depth_seq[j], cmap='viridis')  # use 'gray' if you prefer
            ax.set_title(f"Depth {j}")
            ax.axis('off')

        plt.tight_layout()
        plt.show()

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

info_dicts =[]
@hydra.main(config_path="../configs", config_name="inference_regression", version_base=None)
def eval_policy(cfg: DictConfig) -> None:
    if cfg.env.on_real:
        matplotlib.use("Agg")   # headless backend, no X11 required
    
    log.info("Starting policy evaluation/inference...")
    log.info(f"Using experiment config: {cfg.experiment_name}")

    device_str = cfg.get("inference", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    log.info(f"Using device: {device}")

    # --- NEW: Get the padding strategy from the config ---
    # Defaults to 'zero' if not specified. Options: 'zero', 'copy'
    padding_strategy = cfg.get("inference", {}).get("padding_strategy", "zero")
    log.info(f"Using padding strategy: {padding_strategy}")

    torch.manual_seed(cfg.seed)
    if cfg.env.get("env", None) == "table" or cfg.env.get("env", None) == "TableEnv-v0":
        if cfg.observation_mode in ["points"]: # ...
            img_type = "BOX_POINTS"
        else:
            img_type = "DEPTH"

        env = gym.make("TableEnv-v0", obs_type="depth_rgb_agent_pos", q0=cfg.env.get("q0", [.0, .0, .0, -2., 0. ,2., -0.5]), obj=cfg.env.get("obj", "book"), img_type=img_type, robot_mode=cfg.env.robot_mode, path_mode=cfg.env.path_mode, camera_name=cfg.env.camera_name, simulate=cfg.env.simulate, botop=cfg.env.get("botop", False), on_real=cfg.env.get("on_real", False), seed=cfg.seed, collect_data=False, box_size_ranges=cfg.env.box_size_ranges, box_offset_ranges=cfg.env.box_offset_ranges, allow_book_yaw=cfg.env.get("allow_book_yaw", False), table_offset_ranges=cfg.env.table_offset_ranges, camera_offset_ranges=cfg.env.camera_offset_ranges, camera_rpy_ranges=cfg.env.camera_rpy_ranges, focal_length_range=cfg.env.focal_length_range, depth_noise_ranges=cfg.env.depth_noise_ranges, extras="WAYPOINTS")
    else:
        env = gym.make("ShelfEnv-v1", obs_type="depth_agent_pos", end_effector=cfg.env.get("end_effector", None), obj=cfg.env.get("obj", "book"), robot_mode=cfg.env.robot_mode, path_mode=cfg.env.path_mode, camera_name=cfg.env.camera_name, simulate=cfg.simulate, seed=cfg.seed, shelf_pos_xyz=cfg.env.shelf_pos_xyz, shelf_quaternion=cfg.env.shelf_quaternion, shelf_floor_offsets=cfg.env.shelf_floor_offsets, collect_data=False, box_size_ranges=cfg.env.box_size_ranges, allow_book_yaw=cfg.env.allow_book_yaw, focal_length_range=cfg.env.focal_length_range, extras="WAYPOINTS")
        env.unwrapped.C.view(True)

    
    action_execution_horizon = cfg.get("action_execution_horizon")

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

    model.eval()
    log.info("Model set to evaluation mode.")

    cm_errs = []
    
    if cfg.get("is_regression", False):
        log.info("Model is set for regression.")
        for i in range(cfg.num_eval_episodes):
            obs, info = env.reset()

            current_depth = torch.from_numpy(obs["depth"]).float().to(device).unsqueeze(0)
    
            if cfg.observation_mode == 'dino_cls':
                depth_obs = get_cls_features(current_depth)
            elif cfg.observation_mode == 'dino_patches':
                depth_obs = get_patch_features(current_depth)
            elif cfg.observation_mode == 'sam_points':
                if i == 0:
                    plt.imshow(current_depth.squeeze(0).cpu().numpy(), cmap='gray')
                    plt.show()
                current_rgb = torch.from_numpy(obs["rgb"]).float().cpu().numpy()
                current_rgb = (current_rgb * 255).astype(np.uint8)
                sam_pc = get_sam_pointcloud(env.unwrapped.C, cfg.env.camera_name, current_rgb, current_depth.squeeze(0).cpu().numpy())
                if sam_pc is None:
                    log.error("SAM point cloud is None, skipping this episode.")
                    continue
                else:
                    depth_obs = torch.from_numpy(sam_pc).to(device).float()
            elif cfg.observation_mode == 'points':
                center = env.unwrapped.C.getFrame("BOX_MASK").getPosition()
                box_size = env.unwrapped.C.getFrame("BOX_MASK").getSize()

                points = get_pc_from_depth(env.unwrapped.C, env.unwrapped.camera_name, current_depth.squeeze(0).cpu().numpy())

                plt.imshow(current_depth.squeeze(0).cpu().numpy(), cmap='gray')
                plt.show()
                env.unwrapped.C.addFrame("temp_pc").setPointCloud(points)
                env.unwrapped.C.view(True)

                if img_type == "BOX_POINTS":

                    points = point_in_box_filtering(points, (center, box_size), ignore_planes=[])
                points = sample_points(points, n_samples=1024)  



                depth_obs = torch.from_numpy(points).float().to(device).unsqueeze(0)

            else:
                depth_obs = current_depth # No normalization bc of droput (-1 depth)

            depth_obs = depth_obs.unsqueeze(1)  

            if cfg.observation_mode == 'sam_points':
                depth_obs = depth_obs.transpose(1, 0)

            with torch.no_grad():
                waypoint = model(depth_obs)

            #cm_err = np.linalg.norm(waypoint.cpu().numpy() - env.unwrapped.waypoint_pos)
            #cm_errs.append(cm_err)
            #print("Predicted waypoint:", cm_err)
            env.unwrapped.C.addFrame("predicted_waypoint").setPosition(waypoint.cpu().numpy()).setShape(ry.ST.sphere, [.02]).setColor([1, 0, 0, .9])
            env.unwrapped.C.view(True)

            roboenv = RobotEnviroment(env.unwrapped.C, sim=True)


            # roboenv.pull_way2way("predicted_waypoint", None, False)

            # bot = ry.BotOp(env.unwrapped.C, False)
            # bot.moveAutoTimed(roboenv.path)
            # while bot.getTimeToEnd() > 0:
            #     bot.wait(env.unwrapped.C, 0.1)

            #env.unwrapped.C.view(True)

        avg_cm_err = sum(cm_errs) / len(cm_errs)
        log.info(f"Average CM Error over {cfg.num_eval_episodes} episodes: {avg_cm_err:.5f} m")
        log.info(f"Average squared error thus {avg_cm_err**2}.")
        quit()

    normalization_stats_path = cfg.get("inference", {}).get("normalization_stats_path", None)
    if normalization_stats_path is None:
        log.error("normalization_stats_path not found.")
        return

    with open(normalization_stats_path, 'r') as file:
        normalization_stats = yaml.safe_load(file)

    info_dicts = []
    if cfg.model.policy_head_type == "diffusion":
        sequence_length = cfg.model.prediction_length
    else:
        sequence_length = cfg.model.context_length
        
    for evaluation in range(cfg.get("num_eval_episodes")):
        obs, info = env.reset()

        # History lists
        depth_sequence = []
        state_sequence = []
        
        action_chunk = None
        max_episode_length = 100

        for i in range(max_episode_length):
            if i % action_execution_horizon == 0:

                log.info(f"--- Step {i}: Generating new action chunk ---")
                
                # Get current observation and normalize it
                current_depth = torch.from_numpy(obs["depth"]).float().to(device).unsqueeze(0)
                current_state = torch.tensor(obs["agent_pos"], dtype=torch.float32, device=device).unsqueeze(0)
                if cfg.observation_mode == 'depth':
                    depth_obs = normalize_depth(current_depth, normalization_stats["depth_stats"])
                elif cfg.observation_mode == 'points':
                    center = env.unwrapped.C.getFrame("BOX_MASK").getPosition()
                    box_size = env.unwrapped.C.getFrame("BOX_MASK").getSize()

                    points = get_pc_from_depth(env.unwrapped.C, env.unwrapped.camera_name, current_depth.squeeze(0).cpu().numpy())

                    points = point_in_box_filtering(points, (center, box_size), ignore_planes=[])
                    points = sample_points(points, n_samples=1024)  

                    depth_obs = torch.from_numpy(points).float().to(device).unsqueeze(0)


                elif cfg.observation_mode == 'dino_cls':
                    depth_obs = get_cls_features(current_depth)

                normalized_current_state = normalize_state(current_state, normalization_stats["action_stats"])

                # --- MODIFIED: Fixed Zero Padding Logic ---
                if padding_strategy == 'zero':
                    # Create sequence with proper zero padding
                    if len(depth_sequence) == 0:
                        # First prediction: all zeros except last entry (current obs)
                        dummy_depth = torch.zeros_like(depth_obs)
                        dummy_state = torch.zeros_like(normalized_current_state)
                        
                        padded_depth_list = [dummy_depth] * (sequence_length - 1) + [depth_obs]
                        padded_state_list = [dummy_state] * (sequence_length - 1) + [normalized_current_state]
                    else:
                        # Subsequent predictions: zero pad + history + current
                        history_length = len(depth_sequence)
                        num_zeros_needed = max(0, sequence_length - history_length - 1)
                        
                        dummy_depth = torch.zeros_like(depth_obs)
                        dummy_state = torch.zeros_like(normalized_current_state)
                        
                        # Build sequence: [zeros] + [history] + [current]
                        padded_depth_list = ([dummy_depth] * num_zeros_needed + 
                                           depth_sequence[-min(history_length, sequence_length-1):] + 
                                           [depth_obs])
                        padded_state_list = ([dummy_state] * num_zeros_needed + 
                                           state_sequence[-min(history_length, sequence_length-1):] + 
                                           [normalized_current_state])
                        
                        # Ensure we don't exceed sequence_length
                        if len(padded_depth_list) > sequence_length:
                            padded_depth_list = padded_depth_list[-sequence_length:]
                            padded_state_list = padded_state_list[-sequence_length:]
                
                else:
                    raise ValueError(f"Unknown padding_strategy: {padding_strategy}")
                # ----------------------------------------------

                # Stack history into a batch for the model
                depth_seq = torch.stack(padded_depth_list, dim=1)
                state_seq = torch.stack(padded_state_list, dim=1)
                if DEBUG_DEPTH:
                    test_depth_sequence(depth_seq.cpu().numpy())
                if DEBUG_STATE:
                    state_seq_denorm = denormalize_actions(state_seq, normalization_stats["action_stats"]).squeeze()
                    show_state_input_seq(cfg, env, state_seq_denorm, color=[0, 1, 0, .9])
                with torch.no_grad():
                    action_chunk = model(depth_seq, state_seq)
                action_chunk = denormalize_actions(action_chunk, normalization_stats["action_stats"])
                print("Generated action chunk:", action_chunk.shape)
                # for i in range(8):
                #     env.unwrapped.C.addFrame(f"aaaa{i}").setShape(ry.ST.sphere, [.01]).setColor([0, 1, 1, .9]).setPosition(action_chunk[0, i, :3].cpu().numpy())
                # env.unwrapped.C.view(True)
                if DEBUG_STATE:
                    pass
                    #show_state_input_seq(cfg, env, action_chunk.squeeze(0))

            action_index_in_chunk = i % action_execution_horizon
            action = action_chunk[:, action_index_in_chunk, :].squeeze().cpu().numpy()
            
            log.info(f"Step {i}: Executing action {action_index_in_chunk} from chunk.")
            obs, reward, terminated, truncated, info = env.step(action)
            denormalized_seq = denormalize_actions(state_seq, normalization_stats["action_stats"])
            
            # Store the normalized observation in history (after action execution)
            depth = torch.from_numpy(obs["depth"]).float().to(device).unsqueeze(0)
            if cfg.observation_mode == 'depth':
                depth_obs = normalize_depth(depth, normalization_stats["depth_stats"])
            elif cfg.observation_mode == 'dino_cls':
                depth_obs = get_cls_features(depth)
                
            state_tensor = torch.tensor(obs["agent_pos"], dtype=torch.float32, device=device).unsqueeze(0)
            state_tensor = normalize_state(state_tensor, normalization_stats["action_stats"])
            
            depth_sequence.append(depth_obs)    # depth_obs?
            state_sequence.append(state_tensor)

            # Keep only the most recent observations (sliding window)
            if len(depth_sequence) > sequence_length:
                depth_sequence.pop(0)
                state_sequence.pop(0)

            if terminated or truncated:
                log.info(f"Episode finished at step {i}.")
                break

        log.info(f"Evaluation {evaluation} finished with distance to goal {info.get('distance_to_goal', 'N/A')} and success {info.get('success', 'N/A')}.")
        info_dicts.append(info)

    output_dir = HydraConfig.get().run.dir
    with open(os.path.join(output_dir, "data.json"), "w") as f:
        json.dump(info_dicts, f, indent=4, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log.info(f"Policy evaluation finished. Results saved to {output_dir}")

if __name__ == "__main__":
    eval_policy()