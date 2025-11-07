"""Module to run trajectories in simulation."""
import time

import numpy as np
import robotic as ry
from robotic import SimulationEngine
from envs.utils import point_in_box_filtering, rescale_img
import cv2
import matplotlib.pyplot as plt 

class Simulator:
    """Wrapper class for ry Simulator, with functionality to run a simulation."""

    def __init__(
        self,
        config: ry.Config,
        engine: SimulationEngine = SimulationEngine.physx,
        verbose: int = 0,
        camera: str = "cameraStatic",
        base_removal: bool = False,  # if true, shelf will be removed from observation
        visualize: bool = False,    # TODO?
        observation_mode: str = "DEPTH"
    ):
        self._sim = ry.Simulation(config, engine, verbose=verbose)
        self._sim.setSimulateDepthNoise(True)
        self.config = config
        self.init_state = self._sim.getState()
        self.camera = camera
        self.points = []
        self.rgb = []
        self.depth = []
        self._sim.selectSensor(camera)
        self.base_removal = base_removal
        self.observation_mode = observation_mode

    def getDepth(self, crop: bool = False, rescale: bool = True, crop_size: int = 96, rescale_size: int = 96) -> np.ndarray:
        _, depth = self._sim.getImageAndDepth()
        if crop == True:
            depth = depth[120:, 150:500]

        elif rescale == True:
            depth = rescale_img(depth, rescale_size)

        return depth
    

    def getPoints(self, n_samples=4096, vis=False):
        _, depth = self._sim.getImageAndDepth()

        CameraView = ry.CameraView(self.config)
        CameraView.setCamera(self.config.getFrame(self.camera))
        fx, fy, cx, cy = CameraView.getFxycxy()
        print([fx, fy, cx, cy])
        point_cloud = self._sim.depthData2pointCloud(depth, [fx, fy, cx, cy])
        
        points = point_cloud.reshape(-1, 3) 

        if self.base_removal:
            t = self.config.getFrame(self.camera).getPosition()
            R = ry.Quaternion().set(self.config.getFrame(self.camera).getPose()[3:]).getMatrix()

            # Correct transformation: Rotate first, then translate
            points = (R @ points.T).T + t
            # Hardcoded for big_box_inside_0_2 currently
            box = self.config.getFrame("big_box_inside_0_2")
            points = point_in_box_filtering(points, (box.getPosition(), box.getSize()[:3]), ignore_planes=["min_x"])

            points = (R.T @ (points - t).T).T

        if vis:        
            self.config.getFrame(self.camera).setPointCloud(points)
        # randomly sample points if more than n_samples        
        if len(points) > n_samples:
            indices = np.random.choice(len(points), n_samples, replace=False)
            sampled_points = points[indices]
            
            return sampled_points
        else:
            self.config.view(True)
            return points  

    def getRGB(self, rescale: bool = True, crop: bool = False, crop_size: int = 96, rescale_size: int = 96) -> np.ndarray:
        rgb, _ = self._sim.getImageAndDepth()

        if crop:
            original_height, original_width, _ = rgb.shape
            left = (original_width - crop_size) // 2
            top = (original_height - crop_size) // 2
            right = left + crop_size
            bottom = top + crop_size

            rgb = rgb[top:bottom, left:right, :]

        if rescale:
            rgb = cv2.resize(rgb, (rescale_size, rescale_size), interpolation=cv2.INTER_LINEAR)

        return rgb

    def run_trajectory_position_control(
        self,
        path: np.ndarray,
        n_steps: float,
        tau: float = 5e-4,
        capture_obs: bool = False,
        visualize: bool = False,
    ) -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray]: # type: ignore
        """Run a trajectory in simulation using the specified KOMO instance.

        Args:
            path:
                The planned trajectory from KOMO.
            n_steps:
                The number of steps that the trajectory entails.
                For KOMO-based paths this is typically, phases * dur. per phase.
            tau:
                The time interval between steps in the simulation in seconds.
            capture_depth:
                If True, the depth image will be captured at each step. 
            visualize:
                If True, the simulation will be visualized.

        """
        sim_steps = int(n_steps // tau)
        for i, control_point in enumerate(path):
            if capture_obs:
                if self.observation_mode == "DEPTH":
                    depth = self.getDepth(crop=True, rescale=True)
                    # if i == 0:
                    #     plt.imshow(depth, cmap='gray')
                    #     plt.show()
                    self.depth.append(depth)
                elif self.observation_mode == "RGB":
                    rgb = self.getRGB()
                    self.rgb.append(rgb)

            for _ in range(10):
                self._sim.step(control_point, tau, ry.ControlMode.position)
                
                if visualize:
                    time.sleep(tau/5)
                    self.config.view()


    def run_trajectory_spline(
        self,
        path: np.ndarray,
        n_steps: float,
        tau: float = 5e-4,
        capture_points: bool = False,
        capture_rgb: bool = False,
        capture_depth: bool = False,
    ) -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray]: # type: ignore
        """Run a trajectory in simulation using the specified KOMO instance.

        Args:
            path:
                The planned trajectory from KOMO.
            n_steps:
                The number of steps that the trajectory entails.
                For KOMO-based paths this is typically, phases * dur. per phase.
            tau:
                The time interval between steps in the simulation in seconds.
            capture_points:
                If True, the point cloud will be captured at each step.
            capture_rgb:
                If True, the RGB image will be captured at each step.
            capture_depth:
                If True, the depth image will be captured at each step. 

        """
        sim_steps = int(n_steps // tau)
        times = np.linspace(n_steps / path.shape[-2], n_steps, path.shape[-2])
        self._sim.setSplineRef(path=path, times=times)

        interval = max(1, sim_steps // 32)
        observations_captured = 0
        for i in range(1, sim_steps + 1):
            if ((i - 1) % interval == 0) and (observations_captured < path.shape[-2]):
                if capture_points:
                    points = self.getPoints(vis=True)
                    self.points.append(points)
                if capture_rgb:
                    rgb = self.getRGB()
                    self.rgb.append(rgb)
                if capture_depth:
                    depth = self.getDepth()
                    self.depth.append(depth)
                observations_captured += 1
            
            self._sim.step([], tau, ry.ControlMode.spline)
            if i % 100 == 0:
                self.config.view()

