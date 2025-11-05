import numpy as np
import matplotlib.pyplot as plt
import robotic as ry
import gymnasium as gym
import envs  # noqa: F401  
from envs.utils import crop_or_rescale_img
import hydra
from omegaconf import DictConfig
import logging

SHOW_DEPTH = True
SHOW_RGB = False

log = logging.getLogger(__name__)

@hydra.main(config_path="../configs/env", config_name="table_3d_abs", version_base=None)
def main(cfg: DictConfig):

    env = gym.make("TableEnv-v0", img_type="DEPTH", robot_mode="taskspace", camera_name="cameraStatic", simulate=True, seed=42, collect_data=False, table_offset_ranges=cfg.table_offset_ranges, camera_offset_ranges=cfg.camera_offset_ranges, camera_rpy_ranges=cfg.camera_rpy_ranges, focal_length_range=cfg.focal_length_range)

    for i in range(100):
        env.reset()
        rgb, depth = env.unwrapped.getImageDepth()

        if SHOW_RGB:
            plt.imshow(rgb)
            plt.show()

        if SHOW_DEPTH:
            depth_scaled = crop_or_rescale_img(depth, False, True )
            plt.imshow(depth_scaled, cmap='gray')
            plt.show()


if __name__ == "__main__":
    main()
