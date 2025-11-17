import hydra
from omegaconf import DictConfig
import logging
import gymnasium as gym
import envs.shelf_env  # noqa: F401  
import robotic as ry   

log = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="data_collection", version_base=None)
def main(cfg: DictConfig):

    log.info("Starting policy evaluation/inference...")

    run_data_collection(cfg)


def run_data_collection(cfg: dict):
    print(f"Running with config: {cfg}")

    collector = None  # Initialize for finally block
    env = gym.make("TableEnv-v0", img_type="BOX_POINTS", q0=cfg.env.q0, robot_mode=cfg.env.robot_mode, camera_name=cfg.env.camera_name, box_size_ranges=cfg.env.box_size_ranges, box_offset_ranges=cfg.env.box_offset_ranges, allow_book_yaw=cfg.env.allow_book_yaw, table_offset_ranges=cfg.env.table_offset_ranges, camera_offset_ranges=cfg.env.camera_offset_ranges, camera_rpy_ranges=cfg.env.camera_rpy_ranges, focal_length_range=cfg.env.focal_length_range, depth_noise_ranges = cfg.env.depth_noise_ranges, extras=cfg.get("extras", ""), collect_data=False)
    env.reset()

    CameraView = ry.CameraView(env.unwrapped.C)
    CameraView.setCamera(env.unwrapped.C.getFrame(cfg.env.camera_name))
    _, depth = CameraView.computeImageAndDepth(env.unwrapped.C)
    point_cloud = ry.depthImage2PointCloud(depth, CameraView.getFxycxy())

    points = point_cloud.reshape(-1, 3) 

    cameraPose = env.unwrapped.C.getFrame(cfg.env.camera_name).getPose()
    rot = ry.Quaternion().set([cameraPose[3:]]).getMatrix()
    points = (rot @ points.T).T  + cameraPose[:3]

    env.unwrapped.C.addFrame("temp_pc").setPointCloud(points)
    env.unwrapped.C.view(True)


    print("Data collection experiment finished.")


if __name__ == "__main__":
    main()
