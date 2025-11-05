import hydra
from omegaconf import DictConfig
import logging
import gymnasium as gym
import envs # noqa: F401  
import matplotlib.pyplot as plt
import robotic as ry

log = logging.getLogger(__name__)
ry.params_add({'DepthNoise/binocular_baseline': .0,
  'DepthNoise/depth_smoothing': 1,
  'DepthNoise/noise_all': .1,
  'DepthNoise/noise_wide': .2,
  'DepthNoise/noise_local': .01,
  'DepthNoise/noise_pixel': .01})


@hydra.main(config_path="../configs", config_name="data_collection", version_base=None)
def main(cfg: DictConfig):

    log.info("Starting policy evaluation/inference...")

    run_data_collection(cfg)


def run_data_collection(cfg: dict):
    print(f"Running with config: {cfg}")

    collector = None  # Initialize for finally block
    # env = gym.make("TableEnv-v0", img_type="DEPTH", robot_mode=cfg.env.robot_mode, camera_name=cfg.env.camera_name, simulate=True, botop=True, seed=cfg.seed, collect_data=True)

    # env.reset()

    # env.unwrapped.C.addFrame("measureBox", "l_panda_base").setShape(ry.ST.box, [.118, .04, .04]).setRelativePosition([-.21, 0, 0.]).setColor([0,1,1,.9])
    # env.unwrapped.C.view(True)
    
    # rgb, depth = env.unwrapped.bot.getImageAndDepth("cameraWrist")
    # plt.imshow(depth)
    # plt.show() 
    # del env

    env = gym.make("TableEnv-v0", img_type="DEPTH", robot_mode=cfg.env.robot_mode, camera_name=cfg.env.camera_name, simulate=True, botop=False, seed=cfg.seed, collect_data=True)

    env.reset()
    rgb, depth = env.unwrapped.sim._sim.getImageAndDepth()
    plt.imshow(depth)
    plt.show()

    camview = ry.CameraView(env.unwrapped.C)
    camview.setCamera(env.unwrapped.C.getFrame('cameraWrist'))
    rgb, depth = camview.computeImageAndDepth(env.unwrapped.C, True)
    plt.imshow(depth)
    plt.show()
    print("Data collection experiment finished.")


if __name__ == "__main__":
    main()
