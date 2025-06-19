"""Module to run trajectories in simulation."""
import time

import numpy as np
import robotic as ry
from robotic import SimulationEngine
from envs.utils import point_in_box_filtering

class Simulator:
    """Wrapper class for ry Simulator, with functionality to run a simulation."""

    def __init__(
        self,
        config: ry.Config,
        engine: SimulationEngine = SimulationEngine.physx,
        verbose: int = 0,
        camera: str = "cameraStatic",
        base_removal: bool = False,  # if true, shelf will be removed from observation
    ):
        self._sim = ry.Simulation(config, engine, verbose=verbose)
        self.config = config
        self.init_state = self._sim.getState()
        self.camera = camera
        self.points = []
        self._sim.selectSensor(camera)
        self.base_removal = base_removal

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


    def run_trajectory(
        self,
        path: np.ndarray,
        n_steps: float,
        tau: float = 5e-4,
        capture_points: bool = False,
    ) -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray]: # type: ignore
        """Run a trajectory in simulation using the specified KOMO instance.

        Args:
            path:
                The planned trajectory from KOMO.
                Can also be an array of multiple trajectories.
            n_steps:
                The number of steps that the trajectory entails.
                For KOMO-based paths this is typically, phases * dur. per phase.
            n_scenes:
                Specifies how many scenes in parallel are being used.
            tau:
                The time interval between steps in the simulation in seconds.


        Returns:
            frame_trajectory: The sequence of frame states along the simulated traj.
            joint_trajectory: The sequence of joint states along the simulated traj.
        """
        sim_steps = int(n_steps // tau)
        times = np.linspace(n_steps / path.shape[-2], n_steps, path.shape[-2])
        self._sim.setSplineRef(path=path, times=times)

        # TODO think about this, maybe change control mode, and then better capture points every path
        interval = max(1, sim_steps // 32)
        points_captured = 0
        for i in range(1, sim_steps + 1):
            if capture_points and ((i - 1) % interval == 0) and (points_captured < path.shape[-2]):
                points = self.getPoints(vis=True)
                self.points.append(points)
                points_captured += 1
            self._sim.step([], tau, ry.ControlMode.spline)
            if i % 100 == 0:
                self.config.view()

