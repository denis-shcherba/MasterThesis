import hydra
from omegaconf import DictConfig
import logging
import gymnasium as gym
import envs.shelf_env  # noqa: F401  

log = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="data_collection", version_base=None)
def main(cfg: DictConfig):

    log.info("Starting policy evaluation/inference...")

    run_data_collection(cfg)


def run_data_collection(cfg: dict):
    print(f"Running with config: {cfg}")

    collector = None  # Initialize for finally block
    env = gym.make(cfg.env.env, save_obj_pos=True, obs_type="RGBDEPTH", robot_mode=cfg.env.robot_mode, end_effector=cfg.env.get("end_effector", None), task=cfg.env.task, obj=cfg.env.obj, path_mode=cfg.env.path_mode, q0=cfg.env.get("q0", [.0, .0, .0, -2., 0. ,2., -0.5]), margins=cfg.env.get("margins", {}), camera_name=cfg.env.camera_name, box_size_ranges=cfg.env.box_size_ranges, box_offset_ranges=cfg.env.get("box_offset_ranges", None), table_offset_ranges=cfg.env.get("table_offset_ranges", None), camera_offset_ranges=cfg.env.get("camera_offset_ranges", None), camera_rpy_ranges=cfg.env.get("camera_rpy_ranges", None), focal_length_range=cfg.env.get("focal_length_range", None), depth_noise_ranges = cfg.env.get("depth_noise_ranges", None), extras=cfg.get("extras", ""), shelf_pos_xyz=cfg.env.get("shelf_pos_xyz", None), shelf_quaternion=cfg.env.get("shelf_quaternion", None), shelf_floor_offsets=cfg.env.get("shelf_floor_offsets", None), collect_data=False, rotate_panda_base=cfg.env.get("rotate_panda_base", True), hook_base_length=cfg.env.get("hook_base_length", 0.4), hook_tip_length=cfg.env.get("hook_tip_length", 0.14), hook_width=cfg.env.get("hook_width", 0.02))

    samples = 20
    success = 0
    for i in range(samples):
        env.reset()
        succ = env.unwrapped.roboenv.hook_book("target_book_0")
        
        if succ:
            success += 1
    print(f"Success rate: {success}/{samples}")


if __name__ == "__main__":
    main()
