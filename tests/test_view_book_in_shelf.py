import hydra
from omegaconf import DictConfig
import logging
import gymnasium as gym
import envs.shelf_env  # noqa: F401  
import robotic as ry   
import matplotlib.pyplot as plt
import h5py
from envs.high_level_methods import RobotEnviroment
import numpy as np

log = logging.getLogger(__name__)

# ry.params_add({'DepthNoise/binocular_baseline': .05,
#   'DepthNoise/depth_smoothing': 1,
#   'DepthNoise/noise_all': .05,
#   'DepthNoise/noise_wide': 4.,
#   'DepthNoise/noise_local': .4,
#   'DepthNoise/noise_pixel': .04})


@hydra.main(config_path="../configs/env/", config_name="shelf_hook_panda", version_base=None)
def main(cfg: DictConfig):
    env = gym.make(cfg.env, save_obj_pos=True, img_type="DEPTH", robot_mode=cfg.robot_mode, end_effector=cfg.get("end_effector", None), task=cfg.task, obj=cfg.obj, path_mode=cfg.path_mode, q0=cfg.get("q0", [.0, .0, .0, -2., 0. ,2., -0.5]), camera_name=cfg.camera_name, box_size_ranges=cfg.box_size_ranges, box_offset_ranges=cfg.get("box_offset_ranges", None), table_offset_ranges=cfg.get("table_offset_ranges", None), camera_offset_ranges=cfg.get("camera_offset_ranges", None), camera_rpy_ranges=cfg.get("camera_rpy_ranges", None), focal_length_range=cfg.get("focal_length_range", None), depth_noise_ranges = cfg.get("depth_noise_ranges", None), extras=cfg.get("extras", ""), shelf_pos_xyz=cfg.get("shelf_pos_xyz", None), shelf_quaternion=cfg.get("shelf_quaternion", None), shelf_floor_offsets=cfg.get("shelf_floor_offsets", None), collect_data=False)

    env.unwrapped.C.view(True, "Initial Environment View, to close press q in the viewer")
    view_book_from_h5(env, view_in_between=True)    

    #visit_arena_markers(env.unwrapped.C, on_real=True)    
    #test_env_puck_positions(env)
    #run_puck_from_h5(env)


def test_env_puck_positions(env, iterations=50, view_in_between=False):

    cylinder_pos_list = []
    for i in range(iterations):
        env.reset()
        env.unwrapped.C.view(view_in_between, "Puck Position Test Env, to close press q in the viewer")
        cylinder_pos_list.append(env.unwrapped.C.getFrame("cylinder").getPosition().copy())
    for i in range(iterations):    
        env.unwrapped.C.addFrame(f"test_cylinder_{i}").setPosition(cylinder_pos_list[i]).setShape(ry.ST.cylinder, size=[0.03, 0.04]).setColor([0, 0, 0])

    env.unwrapped.C.view(True, "All Puck Positions, to close press q in the viewer")

    for i in range(iterations):    
        env.unwrapped.C.delFrame(f"test_cylinder_{i}")
    env.unwrapped.C.view(False, "All Puck Positions, to close press q in the viewer")


def view_book_from_h5(env, view_in_between=False):
    book_params_list = []
    with h5py.File("shelf_demo.h5", 'a') as f:
        demo_groups = [name for name in f.keys() if name.startswith('demo_')]
        total_demos = len(demo_groups)

        for i, demo_name in enumerate(demo_groups):
            print(f"  -> Processing {demo_name} ({i + 1}/{total_demos})...", end='\r')
            
            # Read the entire depth array into memory
            book_params = f[demo_name]['book'][:] 
            

            original_shape = book_params.shape
            print(book_params)
            book_params_list.append(book_params)

            env.unwrapped._spawn_book(list(book_params[0]), i)
            env.unwrapped.C.view(view_in_between, f"h5 data point {i}")

    for i in range(len(book_params_list)):    
        env.unwrapped.C.addFrame(f"test_book_{i}").setPosition(book_params_list[i]).setShape(ry.ST.cylinder, size=[0.03, 0.04]).setColor([0, 0, 0])

    env.unwrapped.C.view(True, "All Book Positions, to close press q in the viewer")

    for i in range(len(book_params_list)):    
        env.unwrapped.C.delFrame(f"test_book_{i}")
    env.unwrapped.C.view(False, "All Book Positions, to close press q in the viewer")



if __name__ == "__main__":
    main()
