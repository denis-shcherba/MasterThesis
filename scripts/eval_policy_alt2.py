import cv2
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import logging
import os
import yaml
import json
import gc
import psutil
from collections import Counter
from hydra.core.hydra_config import HydraConfig
import robotic as ry
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import objgraph
# Custom imports (assumed existing in your path)
from models.policy_head.policy_network import create_model
from utils.data_utils import (get_pc_from_depth, normalize_depth, normalize_rgb, 
                             normalize_state, denormalize_actions, get_cls_features)
from envs.utils import point_in_box_filtering, sample_points

# Initialize tracking
matplotlib.use('Agg')
log = logging.getLogger(__name__)

# --- MEMORY TRACKER CLASS ---
class MemoryTracker:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.last_mem = 0
        
    def report(self, episode_idx):
        curr_mem = self.process.memory_info().rss / (1024 ** 2) 
        diff = curr_mem - self.last_mem
        objs = gc.get_objects()
        type_counts = Counter([type(o).__name__ for o in objs])
        
        print(f"\n--- MEMORY REPORT [EPISODE {episode_idx}] ---")
        print(f"Total RAM: {curr_mem:.2f} MB (Change: +{diff:.2f} MB)")
        print(f"Top 5 Objects: {type_counts.most_common(5)}")
        self.last_mem = curr_mem

# --- ISOLATED EPISODE FUNCTION ---
def run_single_episode(evaluation_idx, env, model, cfg, normalization_stats, device, output_dir, sequence_length, key):
    """
    All local objects created here are destroyed when this function returns.
    """
    obs, info = env.reset()
    depth_sequence = []
    state_sequence = []
    action_chunk = None
    max_episode_length = 85
    dist_to_target = float('inf')
    success = False
    video_frames = []
    action_execution_horizon = cfg.get("action_execution_horizon")
    final_rgb = None

    for i in range(max_episode_length):
        # 1. Video Frame Collection
        raw_img = obs.get("raw_rgb")
        if raw_img is not None:
            frame = raw_img.detach().cpu().numpy().copy() if hasattr(raw_img, "cpu") else raw_img.copy()
            if frame.ndim == 3 and frame.shape[0] == 3: frame = np.transpose(frame, (1, 2, 0))
            if frame.dtype != np.uint8: frame = (frame * 255).astype(np.uint8)
            video_frames.append(frame)

        # 2. Action Generation
        if i % action_execution_horizon == 0:
            current_obs = torch.from_numpy(obs[key]).float().to(device).unsqueeze(0)
            current_state = torch.tensor(obs["agent_pos"], dtype=torch.float32, device=device).unsqueeze(0)
            
            # Normalization logic
            if cfg.observation_mode == 'depth':
                depth_obs = normalize_depth(current_obs, normalization_stats["obs_stats"])
            elif cfg.observation_mode == 'rgb':
                depth_obs = normalize_rgb(current_obs).permute(0, 3, 1, 2)
            else:
                depth_obs = current_obs # Simplified for brevity
            
            norm_state = normalize_state(current_state, normalization_stats["action_stats"])

            # Sequence building (History)
            # Note: We create new lists here to avoid reference cycles with outer scope
            hist_len = len(depth_sequence)
            num_zeros = max(0, sequence_length - hist_len - 1)
            dummy_d = torch.zeros_like(depth_obs)
            dummy_s = torch.zeros_like(norm_state)
            
            padded_d = ([dummy_d] * num_zeros + depth_sequence[-(sequence_length-1):] + [depth_obs]) if hist_len > 0 else ([dummy_d] * (sequence_length-1) + [depth_obs])
            padded_s = ([dummy_s] * num_zeros + state_sequence[-(sequence_length-1):] + [norm_state]) if hist_len > 0 else ([dummy_s] * (sequence_length-1) + [norm_state])

            depth_seq = torch.stack(padded_d, dim=1)
            state_seq = torch.stack(padded_s, dim=1)

            with torch.no_grad():
                action_chunk = model(depth_seq, state_seq)
            action_chunk = denormalize_actions(action_chunk, normalization_stats["action_stats"])

        # 3. Environment Step
        idx = i % action_execution_horizon
        action = action_chunk[:, idx, :].squeeze().cpu().numpy()
        obs, reward, terminated, truncated, info = env.step(action)

        # 4. Metrics Tracking
        if info.get("distance_to_target", 100) < dist_to_target:
            dist_to_target = info["distance_to_target"]
            final_rgb = env.unwrapped.getImage()
        if info.get("success", False): success = True

        # 5. History Update
        # CRITICAL: Detach and move to CPU if not needed for next model pass to save GPU RAM
        depth_sequence.append(depth_obs.detach())
        state_sequence.append(norm_state.detach())
        if len(depth_sequence) > sequence_length:
            depth_sequence.pop(0)
            state_sequence.pop(0)

        if terminated or truncated: break

    # 6. Video Saving
    if video_frames:
        v_path = os.path.join(output_dir, f"eval_ep_{evaluation_idx}.mp4")
        h, w, _ = video_frames[0].shape
        out = cv2.VideoWriter(v_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (w, h))
        for f in video_frames: out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        out.release()

    # 7. Cleanup local variables explicitly
    results = {
        "min_dist_to_target": float(dist_to_target),
        "success": bool(success),
        "last_dist": float(info.get("distance_to_target", -1))
    }
    
    # Save the 'best' image
    if final_rgb is not None:
        rgb_rot = cv2.rotate(final_rgb, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(os.path.join(output_dir, f"best_dist_{evaluation_idx}.png"), cv2.cvtColor(rgb_rot, cv2.COLOR_RGB2BGR))

    return results

# --- MAIN EVALUATION LOOP ---
@hydra.main(config_path="../configs", config_name="inference_shelf", version_base=None)
def eval_policy(cfg: DictConfig) -> None:
    # Setup
    device = torch.device(cfg.get("inference", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = HydraConfig.get().run.dir
    tracker = MemoryTracker()
    
    # Model Loading
    model = create_model(cfg.model).to(device)
    checkpoint_path = cfg.inference.checkpoint_path # Assuming absolute or handled
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)['model_state_dict']
    model.load_state_dict({k.replace('module.', '', 1): v for k, v in state_dict.items()})
    model.eval()

    # Env Creation
    # (Using your existing gym.make logic here)
    obs_type = cfg.get("observation_mode", "depth").lower()
    img_type = "BOX_POINTS" if cfg.observation_mode == "points" else "DEPTH"
    env = gym.make("ShelfEnv-v1", obs_type=f"{obs_type}_agent_pos", end_effector=cfg.env.get("end_effector", None), q0=cfg.env.get("q0", None), obj=cfg.env.get("obj", "book"), robot_mode=cfg.env.robot_mode, path_mode=cfg.env.path_mode, camera_name=cfg.env.camera_name, simulate=cfg.simulate, seed=cfg.seed, shelf_pos_xyz=cfg.env.shelf_pos_xyz, shelf_quaternion=cfg.env.shelf_quaternion, shelf_floor_offsets=cfg.env.shelf_floor_offsets, collect_data=False, box_size_ranges=cfg.env.box_size_ranges, allow_book_yaw=cfg.env.allow_book_yaw, focal_length_range=cfg.env.focal_length_range, extras="WAYPOINTS")

    with open(cfg.inference.normalization_stats_path, 'r') as f:
        norm_stats = yaml.safe_load(f)

    seq_len = cfg.model.prediction_length if cfg.model.policy_head_type == "diffusion" else cfg.model.context_length
    
    info_dicts = []

    # --- MAIN LOOP ---
    for evaluation in range(cfg.get("num_eval_episodes")):
        # Run episode in isolated scope
        episode_results = run_single_episode(
            evaluation, env, model, cfg, norm_stats, device, output_dir, seq_len, "depth"
        )
        
        info_dicts.append(episode_results)
        

        # Save JSON data
        with open(os.path.join(output_dir, "data.json"), "w") as f:
            json.dump(info_dicts, f, indent=4)

        # FINAL GARBAGE COLLECTION
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        tracker.report(evaluation)
        print("Check for growth:")
        objgraph.show_most_common_types(limit=10)

if __name__ == "__main__":
    eval_policy()