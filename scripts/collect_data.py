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
    env = gym.make("TableEnv-v0", img_type="DEPTH", robot_mode=cfg.env.robot_mode, camera_name=cfg.env.camera_name, simulate=cfg.env.simulate, seed=cfg.seed, collect_data=True)

    while env.unwrapped.demo_id < cfg.num_samples:
        env.reset()
        env.unwrapped.collect_data()

    print("Data collection experiment finished.")


if __name__ == "__main__":
    main()
