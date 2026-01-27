# TODO

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Import your new base class
from envs.base_robot_env import BaseRobotEnv
from envs.simulator import Simulator
import robotic as ry
import h5py
from envs.book_spawning import generate_random_box_sizes
from envs.high_level_methods import RobotEnviroment
from envs.utils import gram_schmidt_orthonormalize

import robotic as ry
from envs.shelf import generate_shelf
from envs.book_spawning import generate_random_box_params
from envs.simulator import Simulator
from envs.utils import gram_schmidt_orthonormalize


class ShelfEnv(BaseRobotEnv):
    """
    A new environment for a different task (e.g., reaching a target).
    """
    def __init__(self,
                path_type="SE39D",
                box_size_ranges= {'x': (.1, .15), 'y': (.14, .23), 'z': (.009, .045)},
                box_offset_ranges= {'x': (-.05, .05), 'y': (-.05, .05)},
                allow_book_yaw=False,
                camera_name="wristCamera",
                extras="",
                collect_data=False,
                #domain randomization parameters
                camera_offset_ranges = None,
                camera_rpy_ranges = None,
                focal_length_range = (1.5, 1.5),
                depth_noise_ranges = None,
                margins = {},

                shelf_pos_xyz=None, # e.g. [.8, 0., .3]
                shelf_quaternion=None, # e.g. [1, 0, 0, 1] (w,x,y,z) or as expected by generate_shelf
                shelf_floor_offsets=None,
                q0=None,
                rotate_panda_base=True,
                task = "pull",
                visualize = False,
                num_boxes_per_sample=1,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.box_size_ranges = box_size_ranges
        self.box_offset_ranges = box_offset_ranges
        self.allow_book_yaw = allow_book_yaw
        self.num_boxes_per_sample = num_boxes_per_sample
        self.books = []
        self.task = task
        self.path_type = path_type 
        self.q0 = q0
        self.extras = extras
        self.obj = "book"
        self.rotate_panda_base = rotate_panda_base
        self.margins = margins

        self.camera_name = camera_name
        self.last_pos = np.array([0., 0., 0.])

        if self.q0 is None:
            self.q0 = self.C.getJointState()
        
        self.q0 = np.asarray(self.q0)

        if not self.on_real:
            self.C.setJointState(self.q0)

        self._create_shelf_scene(shelf_pos_xyz, shelf_quaternion, shelf_floor_offsets)

        self.camera_base_pos = self.C.getFrame(self.camera_name).getPosition()
        self.camera_base_rpy = ry.Quaternion().set(self.C.getFrame(self.camera_name).getQuaternion()).getRollPitchYaw()
        
        self.camera_offset_ranges = camera_offset_ranges
        if self.camera_offset_ranges is not None:
            self.camera_offset_x_range = camera_offset_ranges.x
            self.camera_offset_y_range = camera_offset_ranges.y
            self.camera_offset_z_range = camera_offset_ranges.z

        self.camera_rpy_ranges = camera_rpy_ranges
        if camera_rpy_ranges is not None:
            self.camera_pitch_range = camera_rpy_ranges.pitch # degrees
            self.camera_yaw_range = camera_rpy_ranges.yaw # degrees
            self.camera_roll_range = camera_rpy_ranges.roll  # degrees

        self.focal_length_range = focal_length_range 

        self.depth_noise_ranges = depth_noise_ranges
        if self.depth_noise_ranges is not None:
            self.depth_noise_active = depth_noise_ranges['active']
        else:
            self.depth_noise_active = False

        if self.robot_mode == "normal":
            self.C.getFrame("table").setShape(ry.ST.ssBox, size=[.5, 1, .1, .005]).setColor(np.array([242, 240, 216])/255)
            if self.rotate_panda_base:
                self.C.getFrame("l_panda_base").setPosition(self.C.getFrame("l_panda_base").getPosition() + np.array([0, -.08, .0])).setPoseByText("t(-0 -0.1 0.65) d(0 0 0 1)")
            print(self.C.getJointState())
            self.C.setJointState(self.q0)
            self.C.view(False)

        self.roboenv = RobotEnviroment(self.C, sim=self.simulate, gripper=self.gripper, observation_mode=self.obs_type.upper(), visualize=visualize, path_mode="SE39D", camera=self.camera_name, depth_noise=self.depth_noise_active)
        if collect_data: 
            self.h5file = h5py.File("shelf_demo_2.h5", "w")
            self.demo_id = 0
            
        self._setup_scene()

    def _create_shelf_scene(self, shelf_pos_xyz, shelf_quaternion, shelf_floor_offsets):
        # Placeholder for your simulator connection logic
        print("Connecting to custom simulator...")
        # Shelf setup
        self.shelf_pos = np.array(shelf_pos_xyz) if shelf_pos_xyz is not None else np.array([.8, 0., .3])
        _shelf_quaternion = shelf_quaternion if shelf_quaternion is not None else [1, 0, 0, 1]
        _shelf_floor_offsets = shelf_floor_offsets if shelf_floor_offsets is not None else [0.35, 0.43, 0.30, 0.18, 0.15, 0.2, 0.15, 0.15, 0.15, 0.12]

        generate_shelf(self.C, self.shelf_pos, w=.48, d=.44, h=2, base_quaternion=_shelf_quaternion, floor_offsets=_shelf_floor_offsets)

        self.C.addFrame("cameraWP", self.camera_name).setShape(ry.ST.marker, [.1])

        # Shelf frame for book manipulations (consistent naming from generate_shelf is assumed)
        self.shelf_bottom_frame_name = "big_xy_bottom_0_1" 
        self.shelf_bottom_frame = self.C.getFrame(self.shelf_bottom_frame_name)
        if not self.shelf_bottom_frame:
            raise RuntimeError(f"Shelf bottom frame '{self.shelf_bottom_frame_name}' not found. Check shelf generation logic or name.")

        shelf_size_params = self.shelf_bottom_frame.getSize() # [width, depth, thickness, radius]
        self.shelf_width = shelf_size_params[1]
        self.shelf_depth = shelf_size_params[0]
        self.shelf_plate_thickness = shelf_size_params[2] # Thickness of the bottom plate itself

        # This is the size used for generate_random_box_params, interpreted as the spawning surface dimensions.
        # The Z component here is the thickness of the plate books are spawned on.
        self.shelf_dims_for_spawning = (self.shelf_width, self.shelf_depth, self.shelf_plate_thickness)

        # Shelf corner for book positioning logic (bottom-left corner, Z at center of plate)
        shelf_center_pos = self.shelf_bottom_frame.getPosition()[:3]
        self.shelf_corner_ref_point = shelf_center_pos + np.array([-self.shelf_width/2, -self.shelf_depth/2, 0])

            
    def _spawn_book(self, book_params, i=0, prefix="target_book"):


        b_size_x, b_size_y, b_size_z, b_pos_x, b_pos_y, b_pos_z, b_yaw = book_params
        z_offset = (self.shelf_plate_thickness + b_size_z) / 2
        
        book_center_position = self.shelf_corner_ref_point + np.array([b_pos_x, b_pos_y, z_offset])
        
        q_orientation = ry.Quaternion().setRollPitchYaw([0, 0, b_yaw]) 
        
        frame_name = f"{prefix}_{i}"
        self.books.append(frame_name)
        self.C.addFrame(frame_name) \
            .setPosition(book_center_position) \
            .setQuaternion(q_orientation.asArr()) \
            .setShape(ry.ST.ssBox, size=[b_size_x, b_size_y, b_size_z, 0.005]) \
            .setColor([1, 0, 0]) \
            .setContact(1) \
            .setMass(.1) \
            .setAttributes({"friction": 1}) 
        
        # self.C.addFrame(f"corner_1")
        # self.C.getFrame(f"corner_1").setParent(self.C.getFrame(frame_name))
        # self.C.getFrame(f"corner_1", ) \
        #     .setRelativePosition(np.array([-b_size_x/2, -b_size_y/2, -b_size_z/2])) \
        #     # .setShape(ry.ST.marker, [.1]) 
        # self.books.append("corner_1")
            
        # self.C.addFrame(f"corner_2")
        # self.C.getFrame(f"corner_2").setParent(self.C.getFrame(frame_name))
        # self.C.getFrame(f"corner_2", ) \
        #     .setRelativePosition(np.array([b_size_x/2, -b_size_y/2, -b_size_z/2])) \
        #     # .setShape(ry.ST.marker, [.1]) 
        # self.books.append("corner_2")

        # self.C.addFrame(f"corner_3")
        # self.C.getFrame(f"corner_3").setParent(self.C.getFrame(frame_name))
        # self.C.getFrame(f"corner_3", ) \
        #     .setRelativePosition(np.array([-b_size_x/2, b_size_y/2, -b_size_z/2])) \
        #     # .setShape(ry.ST.marker, [.1]) 
        # self.books.append("corner_3")

            
        # self.C.addFrame(f"corner_4")
        # self.C.getFrame(f"corner_4").setParent(self.C.getFrame(frame_name))
        # self.C.getFrame(f"corner_4", ) \
        #     .setRelativePosition(np.array([b_size_x/2, b_size_y/2, -b_size_z/2])) \
        #     # .setShape(ry.ST.marker, [.1]) \
        # self.books.append("corner_4")

        self.C.view(False)
        
    def _spawn_books_scene(self):
        sample = generate_random_box_params(
            shelf_size=self.shelf_dims_for_spawning, # (shelf_width, shelf_depth, shelf_plate_thickness)
            box_size_ranges=self.box_size_ranges,
            num_samples=1,
            num_boxes=self.num_boxes_per_sample,
            allow_yaw=self.allow_book_yaw,
            margins=self.margins
        )

        for i, book_params in enumerate(sample):
            self._spawn_book(book_params[0], i)            
            if self.save_obj_pos and i == 0:
                self.book_pos = book_params

        # target at the middle of the shelf ending for goal evaluation
        if self.C.getFrame("target") is None:
            target = np.array([
                (self.shelf_bottom_frame.getPosition()[:2] + np.array([-self.shelf_depth/2, 0])),
            ])
            target = np.append(target, self.C.getFrame("target_book_0").getPosition()[2])

            self.C.addFrame("target").setShape(ry.ST.marker, .1).setPosition(target)

    def _delete_books(self):
        for book in self.books:
            self.C.delFrame(book)
        self.books = []


    def _define_action_space(self):
        if self.robot_mode == "jointspace":
            return spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        elif self.robot_mode == "floating":
            return spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        elif self.robot_mode == "taskspace":
            return spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        else:
            return spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

    def _setup_scene(self):
        # target_pos = np.array([.5, 0.1, .7]) 
        self._delete_books()
        self._spawn_books_scene()
        # self.C.getFrame("reach_target").setPosition(target_pos)


    def _get_info(self):
        book_pos = self.C.getFrame("target_book_0").getPosition()
        target_pos = self.C.getFrame("target").getPosition()

        corner_diff_1 = self.C.getFrame("target").getPosition()[0] - self.C.getFrame("corner_1").getPosition()[0] 
        corner_diff_2 = self.C.getFrame("target").getPosition()[0] - self.C.getFrame("corner_2").getPosition()[0]
        corner_diff_3 = self.C.getFrame("target").getPosition()[0] - self.C.getFrame("corner_3").getPosition()[0]
        corner_diff_4 = self.C.getFrame("target").getPosition()[0] - self.C.getFrame("corner_4").getPosition()[0]

        over_shelf = max(corner_diff_1, corner_diff_2, corner_diff_3, corner_diff_4)

        success = over_shelf > 0.025 
        return {"over_shelf": over_shelf, "success": success}

    def hook_block(self):
        success = self.roboenv.hook_book("target_book_0")

    def pull_block(self):
        success = self.roboenv.pull_real("target_book_0", self.C.getFrame("target").getPosition(), accumulated_collisions=True, get_observation=False)

    def getImageDepth(self, camera_name=None):
        if camera_name is None:
            camera_name = self.camera_name

        if self.botop:
            rgb, depth = self.bot.getImageAndDepth(camera_name)
        elif self.simulate:
            self.camview = ry.CameraView(self.C)
            self.camview.setCamera(self.C.getFrame(camera_name))

            rgb, depth = self.camview.computeImageAndDepth(self.C)
        return rgb, depth

    def getImage(self, camera_name=None):
        if camera_name is None:
            camera_name = self.camera_name

        if self.botop:
            rgb, _ = self.bot.getImageAndDepth(camera_name)
        elif self.simulate:
            self.camview = ry.CameraView(self.C)
            self.camview.setCamera(self.C.getFrame(camera_name))

            rgb, _ = self.camview.computeImageAndDepth(self.C)
        return rgb

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if not self.on_real:
            self.C.setJointState(self.q0)
        
        self._setup_scene()    

        if self.botop:
            self.bot = ry.BotOp(self.C, self.on_real)
            if self.on_real:
                self.bot.gripperMove(ry._left, 0)
                while not self.bot.gripperDone(ry._left):
                    self.bot.wait(self.C)
                
                    self.bot.sync(self.C)
                    hook_tip_pos = self.C.eval(ry.FS.position, ["hook_tip"])[0] + np.array([0,0,.08])

                    qHome = self.bot.get_q().copy()
                    
                    # KOMO Solver
                    komo = ry.KOMO(self.C, 1, 1, 0, False)
                    komo.addObjective(times=[], feature=ry.FS.jointState, frames=[], type=ry.OT.sos, scale=[1e-1], target=qHome)
                    komo.addObjective([], ry.FS.position, ['hook_tip'], ry.OT.eq, [1e1], hook_tip_pos)
                    ret = ry.NLP_Solver(komo.nlp(), verbose=0).solve()

                    q = komo.getPath()

                    # Movement Sequence
                    self.bot.moveTo(q[0])
                    while self.bot.getTimeToEnd() > 0:
                        self.bot.wait(self.C)
                        
                #self.bot.home(self.C)

                self.bot.moveTo(self.q0)
                while self.bot.getTimeToEnd() > 0:
                    self.bot.wait(self.C)
        elif self.simulate:
            self.sim = Simulator(self.C, engine=ry.SimulationEngine.physx, verbose=0, camera=self.camera_name)

        self.last_pos = self.C.getFrame(self.gripper_name).getPosition()
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info

    def collect_data(self):
        if self.task == "hook":
            success = self.roboenv.hook_book("target_book_0")

        elif self.task == "pull":
            pass

        if success:
            self.save_data()

    def step(self, action):
        # Your logic to apply an action to the environment
        # `action` will be a numpy array matching `self.action_space`
        #print(f"Executing action: {action}")
        
        if self.robot_mode == "floating":
            for _ in range(100):  # Simulate for 100 steps
                if len(self.C.getJointState()) == 4:
                    self.sim._sim.step([action[0], action[1], action[2], action[3]], 0.01, ry.ControlMode.position)
                elif len(self.C.getJointState()) == 3:
                    self.sim._sim.step([action[0], action[1], action[2]], 0.01, ry.ControlMode.position)
                self.C.view()
        elif self.robot_mode == "normal":
            if self.path_mode == "jointspace":
                if self.on_real and self.botop:
                    self.bot.move([action], [.5])
                    while self.bot.getTimeToEnd() > 0:
                        self.bot.wait(self.C)
                else:
                    for _ in range(100):
                        self.sim._sim.step(action, 0.01, ry.ControlMode.position)
                
            if self.path_mode == "taskspace" or self.path_mode == "pos3d" or self.path_mode == "pos3d_delta" or self.path_mode == "pos3d_rel":
                
                # clip minimum height for z to avoid collisions with table
                if "_delta" in self.path_mode:
                    pass
                else:
                    if action[2] < 0.67:
                        action[2] = 0.67
                
                komo = ry.KOMO()
                komo.setConfig(self.C, False)
                komo.setTiming(1, 1, 1., 0)
                
                komo.clearObjectives()
                komo.addControlObjective([], 0, 1e-1)

                if self.path_mode == "pos3d_delta":
                    komo.addObjective([], ry.FS.position, [self.gripper_name], ry.OT.sos, [1e2], action[:3] + self.last_pos)
                else:
                    komo.addObjective([], ry.FS.position, [self.gripper_name], ry.OT.sos, [1e2], action[:3])
                
                if self.path_mode == "taskspace":
                    rot_matrix = gram_schmidt_orthonormalize(action[3:])
                    quat = ry.Quaternion().setMatrix(rot_matrix).asArr()

                    # self.C.addFrame("temp_frame").setShape(ry.ST.marker, .3).setPosition(action[:3]).setQuaternion(quat)
                    # self.C.view(True)

                    komo.addObjective([], ry.FS.quaternion, [self.gripper_name], ry.OT.sos, [1e2], quat)
                sol = ry.NLP_Solver(komo.nlp())
                sol.setOptions(stopInners=1, damping=1e-4, verbose=0)
                ret = sol.solve()
                # komo.view(True, f'sol{s}')

                if self.botop:
                    self.bot.move([komo.getPath()[0]], [.5])
                    while self.bot.getTimeToEnd() > 0:
                        self.bot.wait(self.C)
                elif self.simulate:
                    for _ in range(20):
                        self.sim._sim.step(komo.getPath()[0], .01, ry.ControlMode.position)
                        self.C.view()
                self.last_pos = self.C.getFrame(self.gripper_name).getPosition()

        # --- After action, get the new results ---
        observation = self._get_obs()
        reward = 1.0 # TODO Your logic for calculating reward, if even necessary
        terminated = False # Your logic for whether the episode has ended (e.g., task success)
        truncated = False # Your logic for whether the episode was cut short (e.g., time limit)
        info = self._get_info()
        
        self.C.view(False)  # Update the view after the action
        # The step function MUST return these five values in this order
        return observation, reward, terminated, truncated, info
