import numpy as np
import matplotlib.pyplot as plt
import robotic as ry
import gymnasium as gym
import envs  # noqa: F401  
from envs.high_level_methods import RobotEnviroment
from envs.utils import rescale_img
import hydra
from omegaconf import DictConfig
import logging
from envs.book_spawning import generate_uniform_box_params
import robotic as ry

log = logging.getLogger(__name__)

@hydra.main(config_path="../configs/env", config_name="shelf_hook_panda_real", version_base=None)
def main(cfg: DictConfig):

    #evaluate_hook_feasibility(cfg)
    collect_hook_data(cfg)



    # env = gym.make("ShelfEnv-v1", q0= cfg.q0, botop=cfg.botop, on_real=cfg.on_real, rotate_panda_base=cfg.rotate_panda_base, real_table=cfg.real_table, img_type="DEPTH", robot_mode=cfg.robot_mode, end_effector="hook", path_mode=cfg.path_mode, shelf_pos_xyz=cfg.shelf_pos_xyz, shelf_quaternion=cfg.shelf_quaternion, shelf_floor_offsets=cfg.shelf_floor_offsets, camera_name=cfg.camera_name, simulate=cfg.simulate, seed=42, collect_data=True, box_size_ranges=cfg.box_size_ranges, allow_book_yaw=cfg.allow_book_yaw,  focal_length_range=cfg.focal_length_range)
    # env.reset()
    # env.unwrapped.C.view(True)

    # book_pos = env.unwrapped.C.getFrame("target_book_0").getPosition()+np.array([-.08,.08,0]) 
    # del env.unwrapped.bot
    # RoboEnv = RobotEnviroment(env.unwrapped.C, sim=(not cfg.on_real), camera="cameraWrist", on_real=cfg.on_real)
    
    # env.unwrapped.C.getFrame("target_book_0").setPosition(book_pos)
    # env.unwrapped.C.view(False)

    # RoboEnv.hook_book("target_book_0")
    # env.unwrapped.C.view(True)

def collect_hook_data(cfg: DictConfig):
    env = gym.make("ShelfEnv-v1", q0= cfg.q0, botop=cfg.botop, on_real=cfg.on_real, rotate_panda_base=cfg.rotate_panda_base, real_table=cfg.real_table, img_type="DEPTH", robot_mode=cfg.robot_mode, end_effector="hook", path_mode=cfg.path_mode, shelf_pos_xyz=cfg.shelf_pos_xyz, shelf_quaternion=cfg.shelf_quaternion, shelf_floor_offsets=cfg.shelf_floor_offsets, camera_name=cfg.camera_name, simulate=True, seed=42, collect_data=True, box_size_ranges=cfg.box_size_ranges, allow_book_yaw=cfg.allow_book_yaw,  focal_length_range=cfg.focal_length_range)
    env.reset()
    env.unwrapped.C.view(True)
    box_size = (cfg.box_size_ranges['x'][0], cfg.box_size_ranges['y'][0], cfg.box_size_ranges['z'][0])
    margins = {'x_min': 0.05, 'x_max': 0.05, 'y_min': 0.02, 'y_max': 0.02}

    uniform_samples = generate_uniform_box_params(env.unwrapped.shelf_dims_for_spawning, box_size=box_size, grid_size=(10, 10), margins=margins)
    print(f"Generated {len(uniform_samples)} uniform samples.")
    del env.unwrapped.bot

    RoboEnv = RobotEnviroment(env.unwrapped.C, sim=(not cfg.on_real), camera="cameraWrist", on_real=cfg.on_real)

    success_count = 0
    
    C2 = ry.Config()
    C2.addConfigurationCopy(env.unwrapped.C)
    C2.delFrame("target_book_0")
    

    for sample in uniform_samples:
        RoboEnv.bot.moveTo(env.unwrapped.q0)
        while RoboEnv.bot.getTimeToEnd() > 0:
            RoboEnv.bot.wait(env.unwrapped.C)
        env.unwrapped._spawn_book(sample[0])
        # env.unwrapped.C.view(True)
        book_pos = env.unwrapped.C.getFrame("target_book_0").getPosition()
        success = RoboEnv.hook_book("target_book_0")
        if success:
            success_count += 1
            C2.addFrame(f"book_{success_count}").setColor([0,1,0]).setPosition(book_pos).setShape(ry.ST.box, sample[0][:3] )
    
    print(f"Hooking success rate: {success_count}/{len(uniform_samples)}")
    C2.view(True)

def evaluate_hook_feasibility(cfg: DictConfig):
    env = gym.make("ShelfEnv-v1", q0= cfg.q0, botop=False, on_real=False, rotate_panda_base=cfg.rotate_panda_base, real_table=cfg.real_table, img_type="DEPTH", robot_mode=cfg.robot_mode, end_effector="hook", path_mode=cfg.path_mode, shelf_pos_xyz=cfg.shelf_pos_xyz, shelf_quaternion=cfg.shelf_quaternion, shelf_floor_offsets=cfg.shelf_floor_offsets, camera_name=cfg.camera_name, simulate=True, seed=42, collect_data=True, box_size_ranges=cfg.box_size_ranges, allow_book_yaw=cfg.allow_book_yaw,  focal_length_range=cfg.focal_length_range)
    env.reset()
    box_size = (cfg.box_size_ranges['x'][0], cfg.box_size_ranges['y'][0], cfg.box_size_ranges['z'][0])
    margins = {'x_min': 0.05, 'x_max': 0.05, 'y_min': 0.02, 'y_max': 0.02}

    uniform_samples = generate_uniform_box_params(env.unwrapped.shelf_dims_for_spawning, box_size=box_size, grid_size=(10, 10), margins=margins)
    print(f"Generated {len(uniform_samples)} uniform samples.")
    RoboEnv = RobotEnviroment(env.unwrapped.C, sim=True, camera="cameraWrist", on_real=False)

    success_count = 0

    for sample in uniform_samples:
        env.reset()
        env.unwrapped._spawn_book(sample[0])
        
        success = RoboEnv.hook_book("target_book_0")
        if success:
            success_count += 1
    
    print(f"Hooking success rate: {success_count}/{len(uniform_samples)}")


def evaluate_hook_length():
    #TODO
    pass

if __name__ == "__main__":
    main()
