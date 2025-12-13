import cv2
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import robotic as ry
from importlib.resources import files
from envs.utils import point_in_box_filtering, rescale_img, sample_points
import abc
import matplotlib.pyplot as plt
from utils.data_utils import get_pc_from_depth
import pyrealsense2 as rs

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
                 path_mode="taskspace",
                 simulate=True,
                 botop=False,
                 on_real=False,
                 camera_name="cameraStatic",
                 seed=42,
                 end_effector=None,
                 **kwargs
                ):
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
        if self.robot_mode == "floating":
            self.gripper = "gripper"
        elif self.robot_mode == "normal":
            self.gripper = "l_gripper"

        self.end_effector = end_effector
        
        if self.obs_type == "rgb_agent_pos":
            self.use_opencv = True
        else:
            self.use_opencv = False

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


        self.bot=False


        if self.use_opencv and self.on_real:
            # Initialize RealSense color-only pipeline (not aligned to depth)
            self._rs_pipeline = rs.pipeline()
            self._rs_config = rs.config()
            # Enable ONLY color stream to avoid implicit alignment
            self._rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self._rs_profile = self._rs_pipeline.start(self._rs_config)
            # Warm-up frames
            for _ in range(5):
                self._rs_pipeline.wait_for_frames()

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
        elif self.obs_type == "rgb_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "rgb": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
                    "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                }
            )
        else:
            raise ValueError(f"Unknown observation type: {obs_type}")

        self.action_space = self._define_action_space()


    def _rs_get_color(self):
        """Grab a color frame from RealSense and return RGB uint8 (H, W, 3)."""
        if self._rs_pipeline is None:
            return None
        frames = self._rs_pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        import numpy as np
        import cv2
        bgr = np.asanyarray(color_frame.get_data())
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    def _rs_shutdown(self):
            if self._rs_pipeline is not None:
                try:
                    self._rs_pipeline.stop()
                except Exception:
                    pass
                self._rs_pipeline = None

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
        
        if self.end_effector == "hook":

            if self.robot_mode == "floating":
                #self.C.setJointState(np.concat([self.C.getJointState()[:3], np.array([1, 0, 0, 0])]))
                #gripper_base(floatZ): { Q:"t(0 0 .1035) d(180 1 0 0) d(-90 0 0 1)", shape: marker, size: [.03] }
                self.C.getFrame("gripper_base").setQuaternion(ry.Quaternion().setRollPitchYaw([0, 3*np.pi/4, 0]).asArr())

            hook_base_length = 0.25
            hook_tip_length = 0.04
            hook_width = 0.02
            gripper_depth = 0.02

            self.C.addFrame("hook_base", self.gripper).setRelativePosition([0, 0, -(hook_base_length/2-gripper_depth/2)]).setShape(ry.ST.box, [hook_width, hook_width, hook_base_length]).setColor([0.7, 0.7, 0.7]).setContact(1)
            self.C.addFrame("hook_second", "hook_base").setRelativePosition([0, hook_tip_length/2-hook_width/2, -hook_base_length/2 ]).setShape(ry.ST.box, [hook_width, hook_tip_length, hook_width]).setColor([0.7, 0.7, 0.7]).setContact(1)
            self.C.addFrame("hook_tip", "hook_second").setRelativePosition([0, hook_tip_length/2, 0])




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
        elif self.path_mode == "posyaw":
            agent_pos_raw = np.zeros(4)
            pose = self.C.getFrame(self.gripper_name).getPose()
            agent_pos_raw[:3] = pose[:3]  
            sin_comp = self.C.eval(ry.FS.scalarProductXY, [self.gripper_name, "table"])[0]
            cos_comp = self.C.eval(ry.FS.scalarProductXX, [self.gripper_name, "table"])[0]
            agent_pos_raw[3] = np.arctan2(sin_comp, cos_comp)

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
                # if self.rescale:
                #   depth = rescale_img(depth, rescale_size=96)


            observation["depth"] = depth
            if self.obs_type == "depth_rgb_agent_pos":
                observation["rgb"] = rgb[100:, :, :]

        elif self.obs_type == "rgb_agent_pos":
            if self.botop:
                if self.on_real:
                    observation["rgb"] = self._rs_get_color()[100:, :, :]

        observation["agent_pos"] = agent_pos
        return observation

    def close(self):
        """Cleans up resources."""
        if self.sim:
            del self.sim
            self.sim = None
        if self.use_opencv and self.on_real:
            self._rs_shutdown()
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

    def save_data(self):
        demo_group = self.h5file.create_group(f"demo_{self.demo_id}")


        if self.path_mode == "taskspace":

            se3_path = np.zeros((self.roboenv.path.shape[0], 9))

            C2 = ry.Config()
            C2.addConfigurationCopy(self.C)
            for i in range(self.roboenv.path.shape[0]):
                C2.setJointState(self.roboenv.path[i])
                ee_pose = C2.eval(ry.FS.pose, [self.gripper])[0]

                q = ry.Quaternion().set(ee_pose[3:])
                R = q.getMatrix()

                se3_path[i, :3] = ee_pose[:3]  # Position
                se3_path[i, 3:9] = np.array([R[0:3, 0], R[0:3, 1]]).flatten()  # Rotation

            demo_group.create_dataset("path", data=se3_path)
        elif self.path_mode == "pos3d" or self.path_mode == "pos3d_rel":
            demo_group.create_dataset("path", data=self.roboenv.path)
        if self.img_type.upper() == "DEPTH":
            demo_group.create_dataset(
            "depth", 
            data=self.roboenv.depth_image,
            compression="gzip",
            compression_opts=4
            )

        elif self.img_type.upper() == "RGB":
            demo_group.create_dataset(
            "rgb", 
            data=self.roboenv.rgb_image,
            compression="gzip",
            compression_opts=4
            )

        elif self.img_type.upper() == "SAM_POINTS":
            demo_group.create_dataset(
            "points", 
            data=self.roboenv.points[0],
            compression="gzip",
            compression_opts=4
            )
        elif self.img_type.upper() == "BOX_POINTS":
            demo_group.create_dataset(
            "points", 
            data=self.roboenv.points,
            compression="gzip",
            compression_opts=4
            )
    
        if "WAYPOINTS" in self.extras.upper():
            demo_group.create_dataset("waypoints", data=self.waypoint_pos)

        # if ...  if save obj_params or so

        
        print(f"Collected Demo: {self.demo_id}")
        self.demo_id += 1

    def stream_rgb(self):
        while True:
            if self.botop:
                rgb, _ = self.bot.getImageAndDepth(self.camera_name)
            elif self.simulate:
                self.camview = ry.CameraView(self.C)
                self.camview.setCamera(self.C.getFrame(self.camera_name))

                rgb, _ = self.camview.computeImageAndDepth(self.C)
            # Convert from RGB (ry outputs RGB) → BGR (OpenCV expects BGR)
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            cv2.imshow("Camera Stream", frame)

            # quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()