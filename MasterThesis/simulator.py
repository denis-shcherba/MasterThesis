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
    ):
        self._sim = ry.Simulation(config, engine, verbose=verbose)
        self.config = config
        self.init_state = self._sim.getState()

    def reset(self) -> None:
        """Reset the simulation and the configuration."""
        # Note: has to be in this specific order because spline has to be for reset state
        self._sim.setState(*self.init_state)
        self._sim.resetSplineRef()

    def run_trajectory(
        self,
        path: np.ndarray,
        n_steps: float,
        tau: float = 5e-3,
        real_time: bool = False,
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
            real_time:
                Whether the display will be in real_time or not.

        Returns:
            frame_trajectory: The sequence of frame states along the simulated traj.
            joint_trajectory: The sequence of joint states along the simulated traj.
        """
        sim_steps = int(n_steps // tau)
        times = np.linspace(n_steps / path.shape[-2], n_steps, path.shape[-2])
        self._sim.setSplineRef(path=path, times=times)

        xs = np.empty((sim_steps + 1, *self.init_state[0].shape))
        qs = np.empty((sim_steps + 1, *self.init_state[1].shape))
        xdots = np.empty((sim_steps + 1, *self.init_state[2].shape))
        qdots = np.empty((sim_steps + 1, *self.init_state[3].shape))
        xs[0], qs[0], xdots[0], qdots[0] = self._sim.getState()
        assert np.allclose(xs[0], self.init_state[0]), (
            "Init state must match env. init, "
            f"but difference is large at "
            f"{np.argwhere((xs[0] - self.init_state[0]) > .1)}"
        )
        for i in range(1, sim_steps + 1):
            self._sim.step([], tau, ry.ControlMode.spline)
            xs[i], qs[i], xdots[i], qdots[i] = self._sim.getState()
            if real_time:
                time.sleep(1e-2)

        # Reset simulation and environments
        self.reset()

        return xs, qs, xdots, qdots
