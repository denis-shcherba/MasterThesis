import hydra
from omegaconf import DictConfig
import logging
import gymnasium as gym
import envs.shelf_env  # noqa: F401  
import robotic as ry   
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

# ry.params_add({'DepthNoise/binocular_baseline': .05,
#   'DepthNoise/depth_smoothing': 1,
#   'DepthNoise/noise_all': .05,
#   'DepthNoise/noise_wide': 4.,
#   'DepthNoise/noise_local': .4,
#   'DepthNoise/noise_pixel': .04})


@hydra.main(config_path="../configs", config_name="data_collection", version_base=None)
def main(cfg: DictConfig):

    log.info("Starting policy evaluation/inference...")

    run_data_collection(cfg)


def run_data_collection(cfg: dict):
    print(f"Running with config: {cfg}")

    collector = None  # Initialize for finally block
    env = gym.make("TableEnv-v0", img_type="BOX_POINTS", q0=cfg.env.get("q0", [.0, .0, .0, -2., 0. ,2., -0.5]), robot_mode=cfg.env.robot_mode, camera_name=cfg.env.camera_name, box_size_ranges=cfg.env.box_size_ranges, box_offset_ranges=cfg.env.box_offset_ranges, allow_book_yaw=cfg.env.allow_book_yaw, table_offset_ranges=cfg.env.table_offset_ranges, camera_offset_ranges=cfg.env.camera_offset_ranges, camera_rpy_ranges=cfg.env.camera_rpy_ranges, focal_length_range=cfg.env.focal_length_range, depth_noise_ranges = cfg.env.depth_noise_ranges, extras=cfg.get("extras", ""), collect_data=False)
    env.reset()

    CameraView = ry.CameraView(env.unwrapped.C)
    CameraView.setCamera(env.unwrapped.C.getFrame(cfg.env.camera_name))
    _, depth = CameraView.computeImageAndDepth(env.unwrapped.C, False)
    point_cloud = ry.depthImage2PointCloud(depth, CameraView.getFxycxy())
    plt.imshow(depth, cmap='gray')
    plt.show()

    points = point_cloud.reshape(-1, 3) 

    cameraPose = env.unwrapped.C.getFrame(cfg.env.camera_name).getPose()
    rot = ry.Quaternion().set([cameraPose[3:]]).getMatrix()
    points = (rot @ points.T).T  + cameraPose[:3]

    env.unwrapped.C.addFrame("temp_pc").setPointCloud(points)
    env.unwrapped.C.view(True)

    test_c = ry.Config()
    test_c.addFrame("base_frame_ugly").setShape(ry.ST.box, size=[100, 100, .01])
    test_c.addFrame("pc").setPointCloud(points, colors=_)
    test_c.view(True)


    print("Data collection experiment finished.")


if __name__ == "__main__":
    main()
