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
    env = gym.make("TableEnv-v0", img_type="BOX_POINTS", robot_mode=cfg.env.robot_mode, q0=cfg.env.get("q0", [.0, .0, .0, -2., 0. ,2., -0.5]), camera_name=cfg.env.camera_name, box_size_ranges=cfg.env.box_size_ranges, box_offset_ranges=cfg.env.box_offset_ranges, allow_book_yaw=cfg.env.allow_book_yaw, table_offset_ranges=cfg.env.table_offset_ranges, camera_offset_ranges=cfg.env.camera_offset_ranges, camera_rpy_ranges=cfg.env.camera_rpy_ranges, focal_length_range=cfg.env.focal_length_range, depth_noise_ranges = cfg.env.depth_noise_ranges, extras=cfg.get("extras", ""), collect_data=True)

    while env.unwrapped.demo_id < cfg.num_samples:
        env.reset()
        env.unwrapped.collect_data()

    print("Data collection experiment finished.")


if __name__ == "__main__":
    main()
