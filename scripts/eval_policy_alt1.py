import cv2
import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import logging
import os
import yaml
import json
import random
import matplotlib
import matplotlib.pyplot as plt
from hydra.core.hydra_config import HydraConfig
import robotic as ry
import gymnasium as gym
import envs  # noqa: F401  

# Internal Utils
from models.policy_head.policy_network import create_model
from utils.data_utils import (get_pc_from_depth, normalize_depth, normalize_rgb, 
                              normalize_state, denormalize_actions, get_cls_features)
from envs.utils import point_in_box_filtering, sample_points

matplotlib.use('Agg')
log = logging.getLogger(__name__)

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def execute_single_episode(evaluation_idx, env, model, cfg, device, normalization_stats, sequence_length, key, output_dir):
    obs, info = env.reset()
    
    depth_sequence = []
    state_sequence = []
    video_frames = []
    action_execution_horizon = cfg.get("action_execution_horizon")
    max_episode_length = 130
    
    dist_to_target = float('inf')
    success = False
    final_rgb = None
    action_chunk_np = None

    for i in range(max_episode_length):
        # 1. Collect Video Frames (from raw_rgb if available)
        raw_img = obs.get("raw_rgb")
        if raw_img is not None:
            frame = raw_img.copy()
            if frame.ndim == 3 and frame.shape[0] == 3: frame = np.transpose(frame, (1, 2, 0))
            if frame.dtype != np.uint8: frame = (frame * 255).astype(np.uint8)
            video_frames.append(frame)

        # 2. Action Generation
        if i % action_execution_horizon == 0:
            with torch.no_grad():
                curr_obs_t = torch.from_numpy(obs[key]).float().to(device).unsqueeze(0)
                curr_state_t = torch.tensor(obs["agent_pos"], dtype=torch.float32, device=device).unsqueeze(0)

                if cfg.observation_mode == 'depth':
                    depth_obs = normalize_depth(curr_obs_t, normalization_stats["obs_stats"])
                else:
                    depth_obs = normalize_rgb(curr_obs_t).permute(0, 3, 1, 2)

                norm_state = normalize_state(curr_state_t, normalization_stats["action_stats"])

                # Maintain history sequence
                if len(depth_sequence) == 0:
                    padded_depth = [torch.zeros_like(depth_obs) for _ in range(sequence_length - 1)] + [depth_obs]
                    padded_state = [torch.zeros_like(norm_state) for _ in range(sequence_length - 1)] + [norm_state]
                else:
                    padded_depth = depth_sequence + [depth_obs]
                    padded_state = state_sequence + [norm_state]

                depth_seq = torch.stack(padded_depth, dim=1)
                state_seq = torch.stack(padded_state, dim=1)

                action_chunk = model(depth_seq, state_seq)
                action_chunk = denormalize_actions(action_chunk, normalization_stats["action_stats"])
                action_chunk_np = action_chunk.cpu().numpy()

        # 3. Step Environment
        action = action_chunk_np[0, i % action_execution_horizon, :]
        obs, reward, terminated, truncated, info = env.step(action)

        # 4. Metrics & "Best Image" Tracking
        current_dist = info.get("distance_to_target", 100)
        if current_dist < dist_to_target:
            dist_to_target = current_dist
            final_rgb, _ = env.unwrapped.getImageDepth() 
            
        if info.get("success", False): 
            success = True

        # 5. Update Sequences
        with torch.no_grad():
            obs_t = torch.from_numpy(obs[key]).float().unsqueeze(0)
            if cfg.observation_mode == 'depth':
                d_obs = normalize_depth(obs_t, normalization_stats["obs_stats"])
            else:
                d_obs = normalize_rgb(obs_t).permute(0, 3, 1, 2)

            s_t = torch.tensor(obs["agent_pos"], dtype=torch.float32).unsqueeze(0)
            s_norm = normalize_state(s_t, normalization_stats["action_stats"])

            depth_sequence.append(d_obs.cpu())
            state_sequence.append(s_norm.cpu())

        if len(depth_sequence) > (sequence_length - 1):
            depth_sequence.pop(0)
            state_sequence.pop(0)

        if terminated or truncated:
            break


    if video_frames:
        v_path = os.path.join(output_dir, f"eval_ep_{evaluation_idx}.mp4")
        h, w, _ = video_frames[0].shape
        out = cv2.VideoWriter(v_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (w, h))
        for f in video_frames: out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        out.release()

    if final_rgb is not None:
        rgb_rot = cv2.rotate(final_rgb, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(os.path.join(output_dir, f"best_dist_{evaluation_idx}.png"), cv2.cvtColor(rgb_rot, cv2.COLOR_RGB2BGR))

    return {
        "episode": evaluation_idx,
        "min_dist_to_target": float(dist_to_target),
        "success": bool(success),
        "last_dist_to_target": float(info.get("distance_to_target", -1))
    }

@hydra.main(config_path="../configs", config_name="inference_table", version_base=None)
def eval_policy(cfg: DictConfig) -> None:
    device = torch.device(cfg.get("inference", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = HydraConfig.get().run.dir
    os.makedirs(output_dir, exist_ok=True)

    model = create_model(cfg.model).to(device)
    checkpoint = torch.load(cfg.inference.checkpoint_path, map_location=device, weights_only=False)
    state_dict = {k.replace('module.', '', 1): v for k, v in checkpoint['model_state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()

    with open(cfg.inference.normalization_stats_path, 'r') as f:
        normalization_stats = yaml.safe_load(f)

    sequence_length = (
        cfg.model.prediction_length
        if cfg.model.policy_head_type == "diffusion"
        else cfg.model.context_length
    )

    obs_type = cfg.get("observation_mode", "depth").lower()
    img_type = "BOX_POINTS" if cfg.observation_mode == "points" else "DEPTH"
    
    all_results = []

    for evaluation in range(cfg.get("num_eval_episodes")):
        episode_seed = cfg.seed + evaluation
        seed_everything(episode_seed)

        env = gym.make(
            "TableEnv-v0",
            obs_type=obs_type,
            q0=cfg.env.get("q0", [.0, .0, .0, -2., 0., 2., -0.5]),
            obj=cfg.env.get("obj", "book"),
            robot_mode=cfg.env.robot_mode,
            path_mode=cfg.env.path_mode,
            camera_name=cfg.env.camera_name,
            simulate=cfg.env.simulate,
            botop=cfg.env.get("botop", False),
            on_real=cfg.env.get("on_real", False),
            seed=episode_seed,
            collect_data=False,
            box_size_ranges=cfg.env.box_size_ranges,
            box_offset_ranges=cfg.env.box_offset_ranges,
            allow_book_yaw=cfg.env.get("allow_book_yaw", False),
            table_offset_ranges=cfg.env.table_offset_ranges,
            camera_offset_ranges=cfg.env.camera_offset_ranges,
            camera_rpy_ranges=cfg.env.camera_rpy_ranges,
            focal_length_range=cfg.env.focal_length_range,
            depth_noise_ranges=cfg.env.depth_noise_ranges,
            extras="WAYPOINTS",
        )

        try:
            ep_results = execute_single_episode(
                evaluation, env, model, cfg, device, normalization_stats, 
                sequence_length, key="depth", output_dir=output_dir
            )
            all_results.append(ep_results)
            
            with open(os.path.join(output_dir, "results.json"), "w") as f:
                json.dump(all_results, f, indent=4)
                
        finally:
            env.close()
            del env

        plt.close('all')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    eval_policy()