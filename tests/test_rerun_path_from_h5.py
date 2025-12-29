import h5py
import numpy as np
import hydra
from omegaconf import DictConfig
import gymnasium as gym
import envs  # noqa: F401  


@hydra.main(config_path="../configs", config_name="inference_shelf", version_base=None)
def rerun_path_from_h5(cfg: DictConfig):
    if cfg.env.get("env", None) == "table" or cfg.env.get("env", None) == "TableEnv-v0":
        env = gym.make("TableEnv-v0", obs_type="rgb_agent_pos", q0=cfg.env.get("q0", [.0, .0, .0, -2., 0. ,2., -0.5]), obj=cfg.env.get("obj", "book"), robot_mode=cfg.env.robot_mode, path_mode=cfg.env.path_mode, camera_name=cfg.env.camera_name, simulate=cfg.env.simulate, botop=cfg.env.get("botop", False), on_real=cfg.env.get("on_real", False), seed=cfg.seed, collect_data=False, box_size_ranges=cfg.env.box_size_ranges, box_offset_ranges=cfg.env.box_offset_ranges, allow_book_yaw=cfg.env.get("allow_book_yaw", False), table_offset_ranges=cfg.env.table_offset_ranges, camera_offset_ranges=cfg.env.camera_offset_ranges, camera_rpy_ranges=cfg.env.camera_rpy_ranges, focal_length_range=cfg.env.focal_length_range, depth_noise_ranges=cfg.env.depth_noise_ranges, extras="WAYPOINTS")
    else:
        env = gym.make("ShelfEnv-v1", obs_type="depth_agent_pos", end_effector=cfg.env.get("end_effector", None), obj=cfg.env.get("obj", "book"), robot_mode=cfg.env.robot_mode, path_mode=cfg.env.path_mode, camera_name=cfg.env.camera_name, simulate=cfg.simulate, seed=cfg.seed, shelf_pos_xyz=cfg.env.shelf_pos_xyz, shelf_quaternion=cfg.env.shelf_quaternion, shelf_floor_offsets=cfg.env.shelf_floor_offsets, collect_data=False, box_size_ranges=cfg.env.box_size_ranges, allow_book_yaw=cfg.env.allow_book_yaw, focal_length_range=cfg.env.focal_length_range, extras="WAYPOINTS")
        env.unwrapped.C.view(True)

    env.reset()

    with h5py.File("shelf_demo_rest.h5", 'r') as f:
        demo_groups = [name for name in f.keys() if name.startswith('demo_')]
        total_demos = len(demo_groups)

        for i, demo_name in enumerate(demo_groups):
            print(f"  -> Processing {demo_name} ({i + 1}/{total_demos})...", end='\r')

            path = f[demo_name]['path'][:]      # (T, H, W)
            for q in path:
                env.step(q)
            quit()
            
if __name__ == "__main__":
    rerun_path_from_h5()