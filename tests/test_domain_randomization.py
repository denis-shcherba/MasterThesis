import numpy as np
import matplotlib.pyplot as plt
import robotic as ry
import gymnasium as gym
import envs  # noqa: F401  
from envs.utils import rescale_img
import hydra
from omegaconf import DictConfig
import logging

SHOW_DEPTH = True
SHOW_RGB = False

log = logging.getLogger(__name__)

@hydra.main(config_path="../configs/env", config_name="table_3d_rel_real", version_base=None)
def main(cfg: DictConfig):

    env = gym.make("TableEnv-v0", img_type="DEPTH", robot_mode="taskspace", camera_name=cfg.camera_name, simulate=True, seed=42, collect_data=False, box_size_ranges=cfg.box_size_ranges, box_offset_ranges=cfg.box_offset_ranges, allow_book_yaw=cfg.allow_book_yaw, table_offset_ranges=cfg.table_offset_ranges, camera_offset_ranges=cfg.camera_offset_ranges, camera_rpy_ranges=cfg.camera_rpy_ranges, focal_length_range=cfg.focal_length_range)

    for i in range(100):
        env.reset()
        rgb, depth = env.unwrapped.getImageDepth()

        if SHOW_RGB:
            plt.imshow(rgb)
            plt.show()

        if SHOW_DEPTH:
            plt.imshow(depth, cmap='gray')
            plt.show()
            depth = depth[120:, 150:500]
            plt.imshow(depth, cmap='gray')
            plt.show()


if __name__ == "__main__":
    main()
