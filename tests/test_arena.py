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

    run_data_collection(cfg)

def find_q0(C):
    
    C.addFrame("target_center").setPosition([0, .34, .69]).setShape(ry.ST.marker, size=[.5])
    C.view(True)
    komo = ry.KOMO(C, phases=1, slicesPerPhase=1, kOrder=0, enableCollisions=False)
    # komo.addObjective(times=[], feature=ry.FS.jointState, frames=[], type=ry.OT.sos, scale=[1e-1],target=env.unwrapped.q0)

    

    komo.addObjective([], ry.FS.positionDiff, ['l_gripper', 'target_center'], ry.OT.eq, [1e1, 1e1, 0])
    komo.addObjective([], ry.FS.vectorZ, ['l_gripper'], ry.OT.eq, 1e1, [0, 0, 1])
    komo.addObjective([], ry.FS.negDistance, ['l_gripper', 'target_center'], ry.OT.ineq, 1, [-0.5])
    komo.addObjective([], ry.FS.scalarProductXY, ['l_gripper', 'target_center'], ry.OT.eq, 1e1)

    #komo.addObjective([], ry.FS.positionDiff, ['l_gripper', 'target'], ry.OT.eq, np.eye(3)-np.outer(delta, delta))


    ret = ry.NLP_Solver(komo.nlp(), verbose=0) .solve()
    #print(ret)

    komo.view(True, "IK solution")
    path = komo.getPath()

    return path[0]

def run_data_collection(cfg: dict):
    print(f"Running with config: {cfg}")

    collector = None  # Initialize for finally block
    env = gym.make("TableEnv-v0", img_type="BOX_POINTS", robot_mode="normal", collect_data=False, camera_name="cameraWrist", obj="cylinder")
    env.reset()
    
    q0 = find_q0(env.unwrapped.C)
    del env

    env = gym.make("TableEnv-v0", img_type="BOX_POINTS", q0=q0, sim = False, botop=True, on_real=True, robot_mode="normal", collect_data=False, camera_name="cameraWrist", obj="cylinder")
    env.reset()
    env.unwrapped.stream_rgb()
    
    # CameraView = ry.CameraView(env.unwrapped.C)
    # CameraView.setCamera(env.unwrapped.C.getFrame(cfg.env.camera_name))
    # _, depth = CameraView.computeImageAndDepth(env.unwrapped.C, False)
    # point_cloud = ry.depthImage2PointCloud(depth, CameraView.getFxycxy())
    # plt.imshow(depth, cmap='gray')
    # plt.show()

    # points = point_cloud.reshape(-1, 3) 

    # cameraPose = env.unwrapped.C.getFrame(cfg.env.camera_name).getPose()
    # rot = ry.Quaternion().set([cameraPose[3:]]).getMatrix()
    # points = (rot @ points.T).T  + cameraPose[:3]

    # env.unwrapped.C.addFrame("temp_pc").setPointCloud(points)
    # env.unwrapped.C.view(True)

    # test_c = ry.Config()
    # test_c.addFrame("base_frame_ugly").setShape(ry.ST.box, size=[100, 100, .01])
    # test_c.addFrame("pc").setPointCloud(points, colors=_)
    # test_c.view(True)



if __name__ == "__main__":
    main()
