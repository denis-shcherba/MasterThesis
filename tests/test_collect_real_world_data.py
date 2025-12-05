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


@hydra.main(config_path="../configs/env/", config_name="table_3d_abs_push", version_base=None)
def main(cfg: DictConfig):
    env = gym.make(cfg.env, simulate=False, on_real=True, botop=True, save_obj_pos=True, img_type="DEPTH", robot_mode=cfg.robot_mode, end_effector=cfg.get("end_effector", None), task=cfg.task, obj=cfg.obj, path_mode=cfg.path_mode, q0=cfg.get("q0", [.0, .0, .0, -2., 0. ,2., -0.5]), camera_name=cfg.camera_name, box_size_ranges=cfg.box_size_ranges, box_offset_ranges=cfg.get("box_offset_ranges", None), table_offset_ranges=cfg.get("table_offset_ranges", None), camera_offset_ranges=cfg.get("camera_offset_ranges", None), camera_rpy_ranges=cfg.get("camera_rpy_ranges", None), focal_length_range=cfg.get("focal_length_range", None), depth_noise_ranges = cfg.get("depth_noise_ranges", None), extras=cfg.get("extras", ""), shelf_pos_xyz=cfg.get("shelf_pos_xyz", None), shelf_quaternion=cfg.get("shelf_quaternion", None), shelf_floor_offsets=cfg.get("shelf_floor_offsets", None), collect_data=True)

    env.unwrapped._draw_arena_grid()
    env.unwrapped.C.view(True, "Initial Environment View, to close press q in the viewer")
    visit_arena_grid(env, on_real=True)    


def visit_arena_grid(env, straight_line=False, on_real=False):
    
    RoboEnv = RobotEnviroment(env.unwrapped.C, sim=(not on_real), camera="cameraWrist", on_real=on_real)

    # RoboEnv.move_to_point(env.unwrapped.C.getFrame("target").getPosition()-np.array([0, 0, .025]), straight_line=True, straight_gripper=True, accumulated_collisions = False)
    # quit()

    for i in range(80, 83):    
        #env.reset(options={"get_obs": False})
        pos = env.unwrapped.C.getFrame(f"arena_grid_{i}").getPosition()
        RoboEnv.move_to_point(pos+np.array([0, 0, .055]), straight_line=True, straight_gripper=True, accumulated_collisions = False)
        env.unwrapped.C.view(True, f"Visiting arena grid point {i} of {8*12}")
        # env.reset(options={"obj_pos": pos})
        # env.unwrapped.C.view(True)

        env.unwrapped._setup_scene(obj_pos=pos)    
        if on_real:
            RoboEnv.bot.home(RoboEnv.C)
        success  = RoboEnv.push_frame_to("cylinder", env.unwrapped.C.getFrame("target").getPosition(), get_observation=True)

        if success:
            demo_group = env.unwrapped.h5file.create_group(f"demo_{i}")

            se3_path = np.zeros((RoboEnv.path.shape[0], 9))

            C2 = ry.Config()
            C2.addConfigurationCopy(env.unwrapped.C)
            for i in range(RoboEnv.path.shape[0]):
                C2.setJointState(RoboEnv.path[i])
                ee_pose = C2.eval(ry.FS.pose, ["l_gripper"])[0]

                q = ry.Quaternion().set(ee_pose[3:])
                R = q.getMatrix()

                se3_path[i, :3] = ee_pose[:3]  # Position
                se3_path[i, 3:9] = np.array([R[0:3, 0], R[0:3, 1]]).flatten()  # Rotation
 
            if env.unwrapped.path_mode == "taskspace":
                demo_group.create_dataset("path", data=se3_path)
            elif env.unwrapped.path_mode == "pos3d" or env.unwrapped.robot_mode == "pos3d_delta" or env.unwrapped.robot_mode == "pos3d_rel":
                demo_group.create_dataset("path", data=se3_path[:, :3])
            if env.unwrapped.img_type.upper() == "DEPTH":
                demo_group.create_dataset(
                "depth", 
                data=RoboEnv.depth_image,
                compression="gzip",
                compression_opts=4
                )

            elif env.unwrapped.img_type.upper() == "RGB":
                demo_group.create_dataset(
                "rgb", 
                data=RoboEnv.rgb_image,
                compression="gzip",
                compression_opts=4
                )



if __name__ == "__main__":
    main()
