import numpy as np
import matplotlib.pyplot as plt
import robotic as ry
import gymnasium as gym
import envs  # noqa: F401  
from envs.high_level_methods import RobotEnviroment
from envs.utils import rescale_img
import hydra
from omegaconf import DictConfig, open_dict
import logging
import robotic as ry
from omegaconf import DictConfig, open_dict


log = logging.getLogger(__name__)

@hydra.main(config_path="../configs/env", config_name="table_3d_abs_push_sim", version_base=None)
def main(cfg: DictConfig):
    make_photo(cfg)


def make_photo(cfg: DictConfig):
    env = gym.make("TableEnv-v0", q0= cfg.q0, botop=False, on_real=False, obj="cylinder", img_type="DEPTH", robot_mode=cfg.robot_mode, path_mode=cfg.path_mode, camera_name=cfg.camera_name, simulate=True, seed=42, collect_data=False, box_size_ranges=cfg.box_size_ranges,  focal_length_range=cfg.focal_length_range)
    env.reset()
    env.unwrapped.C.delFrame("cylinder")
    for i in env.unwrapped.C.getFrameNames():
        print(i)
    env.unwrapped.C.view(False)
    env.unwrapped.C.view(True)
    env.unwrapped.C.view(True)


if __name__ == "__main__":
    main()
