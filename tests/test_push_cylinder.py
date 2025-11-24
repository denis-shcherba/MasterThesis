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

@hydra.main(config_path="../configs/env", config_name="table_3d_abs_push", version_base=None)
def main(cfg: DictConfig):

    env = gym.make("TableEnv-v0", img_type="DEPTH", obj="cylinder", robot_mode="normal", path_mode="pos3d", camera_name=cfg.camera_name, simulate=True, seed=42, collect_data=True, box_size_ranges=cfg.box_size_ranges, box_offset_ranges=cfg.box_offset_ranges, table_offset_ranges=cfg.table_offset_ranges, camera_offset_ranges=cfg.camera_offset_ranges, camera_rpy_ranges=cfg.camera_rpy_ranges, focal_length_range=cfg.focal_length_range)

    plt.ion()
    for i in range(100):
        env.reset()
        img, depth = env.unwrapped.getImageDepth()
        plt.imshow(img)
        plt.draw()
        plt.pause(0.001)
        plt.clf()
        env.unwrapped.push_cylinder()
    plt.ioff()

if __name__ == "__main__":
    main()
