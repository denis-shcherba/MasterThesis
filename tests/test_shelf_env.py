import numpy as np
import matplotlib.pyplot as plt
import robotic as ry
import gymnasium as gym
import envs  # noqa: F401  
from envs.utils import rescale_img
import hydra
from omegaconf import DictConfig
import logging

log = logging.getLogger(__name__)

@hydra.main(config_path="../configs/env", config_name="shelf_hook", version_base=None)
def main(cfg: DictConfig):

    env = gym.make("ShelfEnv-v1", img_type="DEPTH", robot_mode=cfg.robot_mode, end_effector=None, path_mode="pos3d", shelf_pos_xyz=cfg.shelf_pos_xyz, shelf_quaternion=cfg.shelf_quaternion, shelf_floor_offsets=cfg.shelf_floor_offsets, camera_name=cfg.camera_name, simulate=True, seed=42, collect_data=True, box_size_ranges=cfg.box_size_ranges, allow_book_yaw=cfg.allow_book_yaw,  focal_length_range=cfg.focal_length_range)

    print()

    for i in range(100):
        env.reset()
        env.unwrapped.pull_block()


if __name__ == "__main__":
    main()
