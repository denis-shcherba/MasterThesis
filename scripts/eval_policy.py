import cv2
import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import logging
import os
from models.policy_head.policy_network import create_model
from data_handling.processing import pose_9d_to_7d, pose_7d_to_9d
from utils.data_utils import get_pc_from_depth, normalize_depth, normalize_rgb, normalize_state, denormalize_actions, get_cls_features, get_patch_features, get_sam_pointcloud
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
import tracemalloc

tracemalloc.start()

matplotlib.use('Agg')
log = logging.getLogger(__name__)
DEBUG_DEPTH = False
DEBUG_STATE = False
DEBUG_RGB = False

import psutil
import gc
from collections import Counter

class MemoryTracker:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.last_mem = 0
        
    def report(self, episode_idx):
        # 1. Physical RAM usage
        curr_mem = self.process.memory_info().rss / (1024 ** 2)  # MB
        diff = curr_mem - self.last_mem
        
        # 2. Count Python objects to find reference leaks
        objs = gc.get_objects()
        type_counts = Counter([type(o).__name__ for o in objs])
        
        print(f"\n--- MEMORY REPORT [EPISODE {episode_idx}] ---")
        print(f"Total RAM: {curr_mem:.2f} MB (Change: +{diff:.2f} MB)")
        print(f"Top 5 Objects in Memory:")
        for name, count in type_counts.most_common(5):
            print(f"  - {name}: {count}")
        
        self.last_mem = curr_mem

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
@hydra.main(config_path="../configs", config_name="inference_table", version_base=None)
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
    obs_type = cfg.get("observation_mode", "depth").lower()
    if cfg.env.get("env", None) == "table" or cfg.env.get("env", None) == "TableEnv-v0":
        if cfg.observation_mode in ["points"]: # ...
            img_type = "BOX_POINTS"
        else:
            img_type = "DEPTH"


        env = gym.make("TableEnv-v0", obs_type=obs_type, q0=cfg.env.get("q0", [.0, .0, .0, -2., 0. ,2., -0.5]), obj=cfg.env.get("obj", "book"), img_type=img_type, robot_mode=cfg.env.robot_mode, path_mode=cfg.env.path_mode, camera_name=cfg.env.camera_name, simulate=cfg.env.simulate, botop=cfg.env.get("botop", False), on_real=cfg.env.get("on_real", False), seed=cfg.seed, collect_data=False, box_size_ranges=cfg.env.box_size_ranges, box_offset_ranges=cfg.env.box_offset_ranges, allow_book_yaw=cfg.env.get("allow_book_yaw", False), table_offset_ranges=cfg.env.table_offset_ranges, camera_offset_ranges=cfg.env.camera_offset_ranges, camera_rpy_ranges=cfg.env.camera_rpy_ranges, focal_length_range=cfg.env.focal_length_range, depth_noise_ranges=cfg.env.depth_noise_ranges, extras="WAYPOINTS")
    else:
        env = gym.make("ShelfEnv-v1", obs_type=obs_type, end_effector=cfg.env.get("end_effector", None), q0=cfg.env.get("q0", None), obj=cfg.env.get("obj", "book"), robot_mode=cfg.env.robot_mode, path_mode=cfg.env.path_mode, camera_name=cfg.env.camera_name, simulate=cfg.simulate, seed=cfg.seed, shelf_pos_xyz=cfg.env.shelf_pos_xyz, shelf_quaternion=cfg.env.shelf_quaternion, shelf_floor_offsets=cfg.env.shelf_floor_offsets, collect_data=False, box_size_ranges=cfg.env.box_size_ranges, allow_book_yaw=cfg.env.allow_book_yaw, focal_length_range=cfg.env.focal_length_range, extras="WAYPOINTS")

    
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
        
    # TODO handle different observation modes for real e.g. rgb in dino..
    # if cfg.observation_mode == 'dino_cls' or cfg.observation_mode == 'dino_patches':
    #     key = "rgb"
    # else:
    #     key = "depth"
    key = "depth"

    tracker = MemoryTracker() # <-- Initialize here
    output_dir = HydraConfig.get().run.dir

    for evaluation in range(cfg.get("num_eval_episodes")):


        obs, info = env.reset()
        # env.unwrapped.C.view(True, "new evaluation")
        # History lists
        depth_sequence = []
        state_sequence = []
        
        action_chunk = None
        max_episode_length = 85

        dist_to_target = float('inf')
        success = False
        video_frames = []

        for i in range(max_episode_length):

            if torch.cuda.is_available():
                print(f"CUDA allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
                print(f"CUDA reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
                print(f"CUDA max allocated: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
            raw_img = obs.get("raw_rgb") 

            if raw_img is not None:
                # A. Ensure it's a numpy array (it likely is, but safety first)
                if hasattr(raw_img, "cpu"): 
                    raw_img = raw_img.detach().cpu().numpy()
                
                # B. Create a copy to avoid mutating the observation
                frame = raw_img.copy()

                # C. Handle Shape: Models often like (C, H, W), Video needs (H, W, C)
                # Check if the first dimension is 3 (Color channels)
                if frame.ndim == 3 and frame.shape[0] == 3:
                    frame = np.transpose(frame, (1, 2, 0))
                
                # D. Handle Data Type: Models like Float (0..1), Video needs Int (0..255)
                if frame.dtype != np.uint8:
                    # If it's float, we assume it's in range [0, 1] or normalized
                    # If your normalize_rgb did mean/std subtraction, this might look weird
                    frame = (frame * 255).astype(np.uint8)
                
                video_frames.append(frame)

            if i % action_execution_horizon == 0:

                log.info(f"--- Step {i}: Generating new action chunk ---")
                
                # Get current observation and normalize it
                current_obs = torch.from_numpy(obs[key]).float().to(device).unsqueeze(0)
                current_state = torch.tensor(obs["agent_pos"], dtype=torch.float32, device=device).unsqueeze(0)
                if cfg.observation_mode == 'depth':
                    depth_obs = normalize_depth(current_obs, normalization_stats["obs_stats"])
                elif cfg.observation_mode == 'rgb':
                    depth_obs = normalize_rgb(current_obs)
                    depth_obs = depth_obs.permute(0, 3, 1, 2)  # Change to [B, C, H, W]
                elif cfg.observation_mode == 'points':
                    center = env.unwrapped.C.getFrame("BOX_MASK").getPosition()
                    box_size = env.unwrapped.C.getFrame("BOX_MASK").getSize()

                    points = get_pc_from_depth(env.unwrapped.C, env.unwrapped.camera_name, current_obs.squeeze(0).cpu().numpy())

                    points = point_in_box_filtering(points, (center, box_size), ignore_planes=[])
                    points = sample_points(points, n_samples=1024)  

                    depth_obs = torch.from_numpy(points).float().to(device).unsqueeze(0)


                elif cfg.observation_mode == 'dino_cls':
                    depth_obs = get_cls_features(current_obs)

                normalized_current_state = normalize_state(current_state, normalization_stats["action_stats"])

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
            #denormalized_seq = denormalize_actions(state_seq, normalization_stats["action_stats"])
            
            if info.get("distance_to_target", None) is not None:
                if info["distance_to_target"] < dist_to_target:
                    dist_to_target = info["distance_to_target"]

                    rgb, _ = env.unwrapped.getImageDepth()

            if info.get("success", None) is not None:
                if info["success"]:
                    success = True 

            # Store the normalized observation in history (after action execution)
            obs_tensor = torch.from_numpy(obs[key]).float().to(device).unsqueeze(0)
            if cfg.observation_mode == 'depth':
                depth_obs = normalize_depth(obs_tensor, normalization_stats["obs_stats"])
            elif cfg.observation_mode == 'dino_cls':
                depth_obs = get_cls_features(obs_tensor)
            elif cfg.observation_mode == 'dino_patches':
                pass
            elif cfg.observation_mode == 'rgb':
                depth_obs = normalize_rgb(obs_tensor)
                depth_obs = depth_obs.permute(0, 3, 1, 2)  # Change to [B, C, H, W]
                
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


        
        if len(video_frames) > 0:
            video_path = os.path.join(output_dir, f"eval_ep_{evaluation}.mp4")
            
            try:
                # 1. Get video dimensions from the first frame
                # Shape is usually (Height, Width, Channels)
                height, width, layers = video_frames[0].shape
                
                # 2. Initialize the VideoWriter
                # 'mp4v' is a standard codec for .mp4 files
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
                
                # 3. Write frames
                for frame in video_frames:
                    # IMPORTANT: OpenCV expects Blue-Green-Red (BGR), but we collected Red-Green-Blue (RGB).
                    # We convert it here so the colors look correct in the video.
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                    
                out.release() # Close the file properly
                log.info(f"Video saved to {video_path}")
                video_frames.clear()  # Move this here instead of at the end
                del video_frames
                video_frames = []  # Recreate empty list
                
            except Exception as e:
                log.error(f"Failed to save video: {e}")




        info["min_dist_to_target"] = dist_to_target
        info["success"] = success
        log.info(f"Evaluation {evaluation} finished with min dist to target {info.get('min_dist_to_target', 'N/A')}, last dist to target {info.get('distance_to_target', 'N/A')} and success {success}.")
        # rotate pi/2 negative as camera is rotated
        rgb_rot = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)

        cv2.imwrite(
            os.path.join(HydraConfig.get().run.dir, f"best_dist_to_target{evaluation}.png"),
            cv2.cvtColor(rgb_rot, cv2.COLOR_RGB2BGR)
        )
        info_dicts.append(info)

        with open(os.path.join(output_dir, "data.json"), "w") as f:
            json.dump(info_dicts, f, indent=4, default=lambda o: o.item() if hasattr(o, "item") else str(o))

        log.info(f"Policy evaluation finished. Results saved to {output_dir}")
        # 1. Clear Simulator Frames (The binary growth source)
        if hasattr(env.unwrapped, 'C'):
            for name in env.unwrapped.C.getFrameNames():
                if "previous_pos" in name or "temp_pc" in name or "pc_point" in name:
                    env.unwrapped.C.delFrame(name)
        
        # 2. Force Matplotlib to release GUI buffers
        plt.close('all')

        # 3. Explicitly kill large local references
        if 'rgb' in locals(): del rgb
        if 'raw_img' in locals(): del raw_img
        if 'frame' in locals(): del frame
        
        
        # 4. Final Hammer: GC and CUDA Cache
        gc.collect() 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        import objgraph
        import random

        # Print the names of 10 random leaked functions to see what they are
        functions = [o for o in gc.get_objects() if isinstance(o, type(lambda: None))]
        sample_size = min(len(functions), 10)
        print(f"Sample of functions in memory: {[f.__name__ for f in random.sample(functions, sample_size)]}")
        tracker.report(evaluation)
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        print("[ Top 10 Memory Consuming Lines ]")
        for stat in top_stats[:10]:
            print(stat)

        # Force a collection before checking
        gc.collect()

        print(f"\n--- LEAK ANALYSIS FOR EPISODE {evaluation} ---")
        # This shows what types of objects were created since the last call
        objgraph.show_growth(limit=10)

        # This is the "Smoking Gun": If functions/frames are leaking, find out what's holding them
        print("\n--- Why are there so many functions? ---")
        leaked_funcs = [o for o in gc.get_objects() if isinstance(o, type(lambda: None))]
        if len(leaked_funcs) > 1000: # Adjust threshold
            # Look at the most recent leaked functions
            objgraph.show_chain(
                objgraph.find_backref_chain(leaked_funcs[-1], objgraph.is_proper_module),
                filename='chain.png'
            )
            print("Graph saved to chain.png - this shows who is holding the reference!")

if __name__ == "__main__":
    eval_policy()