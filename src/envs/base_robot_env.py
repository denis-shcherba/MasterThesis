import gymnasium as gym
from gymnasium import spaces
import numpy as np
import robotic as ry
from importlib.resources import files
from envs.utils import point_in_box_filtering, rescale_img, sample_points
import abc
import matplotlib.pyplot as plt
from utils.data_utils import get_pc_from_depth

class BaseRobotEnv(gym.Env, abc.ABC):
    """
    An abstract base class for robot environments using the ry simulator.
    
    It handles common logic like:
    - Simulator initialization (C, sim)
    - Robot loading (floating, jointspace, etc.)
    - Camera setup
    - Observation space definition (pixels, depth, etc.)
    - Common observation retrieval (_get_obs)
    """
    
    # metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, 
                 obs_type="depth_agent_pos",
                 robot_mode="floating",
                 path_mode="SE39D",
                 simulate=True,
                 botop=False,
                 on_real=False,
                 camera_name="cameraStatic",
                 gripper="l_gripper",
                 seed=42):
        super().__init__()
        
        print(f"BaseRobotEnv __init__ for {self.__class__.__name__}")
        self.obs_type = obs_type
        self.robot_mode = robot_mode
        self.path_mode = path_mode
        self.simulate = simulate
        self.botop = botop
        self.on_real = on_real
        self.camera_name = camera_name
        self.C = ry.Config()
        self.seed = seed
        self.gripper = gripper

        np.random.seed(self.seed)

        # --- Setup Camera ---
        camera_quat = ry.Quaternion().setRollPitchYaw([-np.pi/2, np.pi/2, 0]) * ry.Quaternion().setRollPitchYaw([-.1, 0, 0])
        self.C.addFrame("worldCamera").setShape(ry.ST.camera, [.1]).setPosition([1,0,0]).setAttributes({"focalLength": .895}).setPosition([-.5, 0, 1.5]).setQuaternion(camera_quat.asArr())
        self.C.viewer().setCamera(self.C.getFrame("worldCamera"))
        # --- Setup Robot ---
        self._load_robot()
        self.q0 = self.C.getJointState()

        if "rel" in self.path_mode:
            self.last_pos = self.C.getFrame(self.gripper).getPosition()

        if self.simulate:
            self.sim = None 

        if self.botop:
            self.bot=None

        if self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Box(low=0, high=255, shape=(96, 96, 3), dtype=np.uint8),
                    "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                }
            )
        elif self.obs_type == "points_agent_pos":
            n_points = 4096
            self.observation_space = spaces.Dict(
                {
                    "points": spaces.Box(low=-np.inf, high=np.inf, shape=(n_points, 3), dtype=np.float32),
                    "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                }
            )
        elif self.obs_type == "depth_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "depth": spaces.Box(low=-np.inf, high=np.inf, shape=(96, 96), dtype=np.float32),
                    "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                }
            )
        elif self.obs_type == "depth_rgb_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "depth": spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32),
                    "rgb": spaces.Box(low=0, high=255, dtype=np.uint8),
                    "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                }
            )
        else:
            raise ValueError(f"Unknown observation type: {obs_type}")

        self.action_space = self._define_action_space()


    def _load_robot(self):
        """Loads the robot model based on self.robot_mode."""
        print(f"Loading robot in mode: {self.robot_mode}")
        if self.robot_mode == "normal":
            self.C.addFile(str(files("envs.scenes") / "single.g"))
            self.prefix = "l_"
            self.gripper_name = "l_gripper"
            self.palm_name = "l_palm"
            table = self.C.getFrame("table")
            if table:
                table.setShape(ry.ST.ssBox, size=[.5, 1., .1, .005]).setColor(np.array([242, 240, 216]) / 255)
            
            coll_camera_wrist = self.C.getFrame("panda_collCameraWrist")
            if coll_camera_wrist:
                 self.C.delFrame("panda_collCameraWrist")

        elif self.robot_mode == "floating":
            self.C.addFile(str(files("envs.scenes") / "floating.g"))
            self.gripper_name = "gripper"
            self.palm_name = "palm"
            self.prefix = ""
            
            current_q = self.C.getJointState()
            offset = np.array([.0, 0, .2]) 
            current_q[:len(offset)] += offset 
            self.C.setJointState(current_q)
        
        else:
            raise ValueError(f"Unknown ROBOT_MODE: {self.robot_mode}")

    def _get_obs(self):
        """Gets an observation from the environment (common logic)."""
        if self.path_mode == "jointspace" or self.robot_mode == "floating":
            agent_pos_raw = self.C.getJointState()
        elif self.path_mode == "pos3d":
            agent_pos_raw = self.C.getFrame(self.gripper_name).getPosition()
        elif self.path_mode == "pos3d_rel":
            agent_pos_raw = self.C.getFrame(self.gripper_name).getPosition()-self.last_pos
        elif self.path_mode == "taskspace":
            agent_pos_raw = np.zeros(9)
            pose = self.C.getFrame(self.gripper_name).getPose()
            q = ry.Quaternion().set(pose[3:])
            R = q.getMatrix()
            agent_pos_raw[:3] = pose[:3]  
            agent_pos_raw[3:9] = np.array([R[0:3, 0], R[0:3, 1]]).flatten()

        agent_pos = np.array(agent_pos_raw, dtype=np.float32)

        observation = {}
        if self.obs_type == "pixels_agent_pos":
            pixels = self.sim.getRGB(rescale=True, rescale_size=96)
            observation["pixels"] = pixels
        elif self.obs_type == "points_agent_pos":
            points = self.sim.getPoints(n_samples=4096, vis=True)
            observation["points"] = points
        elif self.obs_type == "depth_agent_pos" or self.obs_type == "depth_rgb_agent_pos":
            if self.botop:
                if self.on_real:
                    rgb, depth = self.bot.getImageAndDepth(self.camera_name)
                    # depth = depth[120:, 150:500]
                    # depth = rescale_img(depth, rescale_size=96)
                    pass # opencv?
                else:
                    rgb, depth = self.bot.getImageAndDepth(self.camera_name)
                    depth = rescale_img(depth, rescale_size=96)
            elif self.simulate:
                self.camview = ry.CameraView(self.C)

                self.camview.setCamera(self.C.getFrame(self.camera_name))
    
                rgb, depth = self.camview.computeImageAndDepth(self.C, False)


            observation["depth"] = depth
            if self.obs_type == "depth_rgb_agent_pos":
                observation["rgb"] = rgb

        observation["agent_pos"] = agent_pos
        return observation

    def close(self):
        """Cleans up resources."""
        if self.sim:
            del self.sim
            self.sim = None
        print(f"Closed {self.__class__.__name__}.")

    @abc.abstractmethod
    def _define_action_space(self):
        """Must be implemented by subclass. Should return a gym.spaces.Space object."""
        raise NotImplementedError

    @abc.abstractmethod
    def _setup_scene(self):
        """
        Must be implemented by subclass.
        This method is for adding task-specific objects (shelves, books, targets)
        to the self.C configuration.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def _get_info(self):
        """Must be implemented by subclass. Should return an info dictionary."""
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self, seed=None, options=None):
        """
        Must be implemented by subclass.
        Should call super().reset(seed=seed)
        """
        super().reset(seed=seed) 
        print(f"Resetting {self.__class__.__name__}.")

        if self.depth_noise_ranges is not None:
            ry.params_add({'DepthNoise/binocular_baseline': np.random.uniform(self.depth_noise_ranges['binocular_baseline'][0], self.depth_noise_ranges['binocular_baseline'][1]),
                'DepthNoise/depth_smoothing': np.random.uniform(self.depth_noise_ranges['depth_smoothing'][0], self.depth_noise_ranges['depth_smoothing'][1]),
                'DepthNoise/noise_all': np.random.uniform(self.depth_noise_ranges['noise_all'][0], self.depth_noise_ranges['noise_all'][1]),
                'DepthNoise/noise_wide': np.random.uniform(self.depth_noise_ranges['noise_wide'][0], self.depth_noise_ranges['noise_wide'][1]),
                'DepthNoise/noise_local': np.random.uniform(self.depth_noise_ranges['noise_local'][0], self.depth_noise_ranges['noise_local'][1]),
                'DepthNoise/noise_pixel': np.random.uniform(self.depth_noise_ranges['noise_pixel'][0], self.depth_noise_ranges['noise_pixel'][1])})
                
        if self.camera_offset_ranges is not None:
            # TODO
            if self.camera_name == "cameraStatic":
                self.C.getFrame(self.camera_name).setPosition(self.camera_base_pos+np.random.uniform(low=np.array([self.camera_offset_x_range[0], self.camera_offset_y_range[0], self.camera_offset_z_range[0]]), high=np.array([self.camera_offset_x_range[1], self.camera_offset_y_range[1], self.camera_offset_z_range[1]]), size=(3,)))
        
        if self.camera_rpy_ranges is not None:
            r, p, y = self.camera_base_rpy    
            self.C.getFrame(self.camera_name).setQuaternion(ry.Quaternion().setRollPitchYaw([r+np.deg2rad(np.random.uniform(self.camera_rpy_ranges['roll'][0], self.camera_rpy_ranges['roll'][1])),
                                                                                             p+np.deg2rad(np.random.uniform(self.camera_rpy_ranges['pitch'][0], self.camera_rpy_ranges['pitch'][1])),
                                                                                             y+np.deg2rad(np.random.uniform(self.camera_rpy_ranges['yaw'][0], self.camera_rpy_ranges['yaw'][1]))]).asArr())

        # if self.focal_length_range is not None:
        #     self.C.getFrame(self.camera_name).setAttributes({"focalLength": np.random.uniform(self.focal_length_range[0], self.focal_length_range[1])}) \


        if not self.on_real:
            self.C.setJointState(self.q0)
        if self.botop:
            del self.bot
        elif self.simulate:
            del self.sim


    @abc.abstractmethod
    def step(self, action):
        """Must be implemented by subclass. Should return obs, reward, terminated, truncated, info."""
        raise NotImplementedError

    def render(self):
        # Optional: implement a common render method if possible
        pass