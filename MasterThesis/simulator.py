"""Module to run trajectories in simulation."""
import time

import numpy as np
import robotic as ry
from robotic import SimulationEngine


class Simulator:
    """Wrapper class for ry Simulator, with functionality to run a simulation."""

    def __init__(
        self,
        config: ry.Config,
        engine: SimulationEngine = SimulationEngine.physx,
        verbose: int = 0,
        camera: str = "camera",
    ):
        self._sim = ry.Simulation(config, engine, verbose=verbose)
        self.config = config
        self.init_state = self._sim.getState()
        self.camera = camera

    def getPoints(self, n_samples=1000):
        depth, rgb = self._sim.getImageAndDepth()

        CameraView = ry.CameraView(self.config)
        CameraView.setCamera(self.config.getFrame("camera"))
        fx, fy, cx, cy = CameraView.getFxycxy()
        point_cloud = self._sim.depthData2pointCloud(depth, [fx, fy, cx, cy])
        
        # Reshape to get all points
        points = point_cloud.reshape(-1, 3) 

        # randomply sample points if more than n_samples        
        if len(points) > n_samples:
            indices = np.random.choice(len(points), n_samples, replace=False)
            sampled_points = points[indices]
            
            return sampled_points
        else:
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
        # self._sim.moveGripper("gripper", .01)
        # TODO think about control mode, capture points every path 
        for i in range(1, sim_steps + 1):
            self._sim.step([], tau, ry.ControlMode.spline)
            if i % 100 == 0:
                self.config.view()

