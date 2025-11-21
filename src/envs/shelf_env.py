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
                img_type="DEPTH",
                box_size_ranges= {'x': (.1, .15), 'y': (.14, .23), 'z': (.009, .045)},
                box_offset_ranges= {'x': (-.05, .05), 'y': (-.05, .05)},
                allow_book_yaw=False,
                camera_name="wristCamera",
                extras="",
                collect_data=False,
                q0=[0. , -0.5,  0., -2.,  0.,  2., -0.5],
                #domain randomization parameters
                camera_offset_ranges = None,
                camera_rpy_ranges = None,
                focal_length_range = (1.5, 1.5),
                depth_noise_ranges = None,

                shelf_pos_xyz=None, # e.g. [.8, 0., .3]
                shelf_quaternion=None, # e.g. [1, 0, 0, 1] (w,x,y,z) or as expected by generate_shelf
                shelf_openings_small=None, # e.g. [4, 11]
                shelf_equidistant=False,

                num_boxes_per_sample=1,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.box_size_ranges = box_size_ranges
        self.box_offset_ranges = box_offset_ranges
        self.allow_book_yaw = allow_book_yaw
        self.num_boxes_per_sample = num_boxes_per_sample
        self.books = []
        
        self.path_type = path_type
        self.img_type = img_type
        self.extras = extras
        self.q0 = np.array(q0)

        self.camera_name = camera_name
        self.last_pos = np.array([0., 0., 0.])

        if not self.on_real:
            self.C.setJointState(self.q0)

        self._create_shelf_scene(shelf_pos_xyz, shelf_quaternion, shelf_openings_small, shelf_equidistant)

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

        self.C.getFrame("table").setShape(ry.ST.ssBox, size=[.5, 1, .1, .005]).setColor(np.array([242, 240, 216])/255)
        self.C.getFrame("l_panda_base").setPosition(self.C.getFrame("l_panda_base").getPosition() + np.array([0, -.08, .0]))
    
        if self.img_type.upper() == "BOX_POINTS":
            box_mask_height = 1
            box_mask_width = 1.2
            box_mask_depth = 1.75
            pos_offset_x = 0
            pos_offset_y = 0
            pos_offset_z = .03

            self.C.addFrame("BOX_MASK") \
                .setShape(ry.ST.box, size=[box_mask_width, box_mask_depth, box_mask_height]) \
                .setColor([1, 0, 0, .2]) \
                .setPosition(self.C.getFrame("table").getPosition()+np.array([pos_offset_x, pos_offset_y, pos_offset_z+self.C.getFrame("table").getSize()[2]/2+box_mask_height/2])) \
                
        # if collect_data:  
        #     self.h5file = h5py.File("table_demo.h5", "w")
        #     self.roboenv = RobotEnviroment(self.C, sim=self.simulate, gripper=self.gripper, observation_mode=self.img_type, visualize=False, path_mode="SE39D", camera=self.camera_name, depth_noise=self.depth_noise_active)
        #     self.demo_id = 0
            
        self._setup_scene()


    def _create_shelf_scene(self, shelf_pos_xyz, shelf_quaternion, shelf_openings_small, shelf_equidistant):
        # Placeholder for your simulator connection logic
        print("Connecting to custom simulator...")
        # Shelf setup
        self.shelf_pos = np.array(shelf_pos_xyz) if shelf_pos_xyz is not None else np.array([.8, 0., .3])
        _shelf_quaternion = shelf_quaternion if shelf_quaternion is not None else [1, 0, 0, 1]
        _shelf_openings_small = shelf_openings_small if shelf_openings_small is not None else [4, 11]
        
        generate_shelf(self.C, self.shelf_pos, base_quaternion=_shelf_quaternion,
                       openings_small=_shelf_openings_small, equidistant=shelf_equidistant)

        self.C.addFrame("cameraWP", self.camera_name).setShape(ry.ST.marker, [1])

        # Shelf frame for book manipulations (consistent naming from generate_shelf is assumed)
        self.shelf_bottom_frame_name = "big_xy_bottom_0_1" 
        self.shelf_bottom_frame = self.C.getFrame(self.shelf_bottom_frame_name)
        if not self.shelf_bottom_frame:
            raise RuntimeError(f"Shelf bottom frame '{self.shelf_bottom_frame_name}' not found. Check shelf generation logic or name.")

        shelf_size_params = self.shelf_bottom_frame.getSize() # [width, depth, thickness, radius]
        self.shelf_width = shelf_size_params[0]
        self.shelf_depth = shelf_size_params[1]
        self.shelf_plate_thickness = shelf_size_params[2] # Thickness of the bottom plate itself

        # This is the size used for generate_random_box_params, interpreted as the spawning surface dimensions.
        # The Z component here is the thickness of the plate books are spawned on.
        self.shelf_dims_for_spawning = (self.shelf_width, self.shelf_depth, self.shelf_plate_thickness)

        # Shelf corner for book positioning logic (bottom-left corner, Z at center of plate)
        shelf_center_pos = self.shelf_bottom_frame.getPosition()[:3]
        self.shelf_corner_ref_point = shelf_center_pos + np.array([-self.shelf_width/2, -self.shelf_depth/2, 0])

    def _spawn_book(self, book_params, i=0, prefix="target_book"):
        b_size_x, b_size_y, b_size_z = book_params
        
        book_center_position = self.C.getFrame("table").getPosition() + np.array([0, .4,  b_size_z/2 + self.C.getFrame("table").getSize()[2]/2]) + np.array([ 
            np.random.uniform(self.box_offset_ranges['x'][0], self.box_offset_ranges['x'][1]),
            np.random.uniform(self.box_offset_ranges['y'][0], self.box_offset_ranges['y'][1]), 
            0])
            
        yaw = 0
        if self.allow_book_yaw:
            yaw = np.random.uniform(0, np.pi)
        q_orientation = ry.Quaternion().setRollPitchYaw([0, 0, yaw])   # TODO?
        
        frame_name = f"{prefix}_{i}"
        self.books.append(frame_name)
        self.C.addFrame(frame_name) \
            .setPosition(book_center_position) \
            .setQuaternion(q_orientation.asArr()) \
            .setShape(ry.ST.ssBox, size=[b_size_x, b_size_y, b_size_z, 0.005]) \
            .setColor([1, 0, 0]) \
            .setContact(1) \
            .setMass(.1) \
            .setAttributes({"friction": .01}) 
        
        if self.extras.upper() == "WAYPOINTS":
            self.C.addFrame("waypoint_marker").setPosition(self.C.getFrame(self.books[0]).getPosition()+np.array([0, 0, b_size_z/2])).setShape(ry.ST.marker, [.1]).setColor([0, 0, 1, .5])

            #self.waypoint_pos = self.C.eval(ry.FS.positionRel, [self.camera_name, "waypoint_marker"])[0]
            self.waypoint_pos = self.C.getFrame("waypoint_marker").getPosition()

        self.C.addFrame("target_p").setPose(self.C.getFrame(frame_name).getPose())

        self.C.addFrame("target").setParent(self.C.getFrame("target_p"))
        self.C.getFrame("target").setRelativePosition([.2, 0, 0]).setShape(ry.ST.marker, [.2]).setColor([0, 1, 0, .9])

        self.C.view(False)

        
    
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
            .setAttributes({"friction": .01}) 
        
        self.C.view(False)
        
    def _spawn_books_scene(self):
        sample = generate_random_box_params(
            shelf_size=self.shelf_dims_for_spawning, # (shelf_width, shelf_depth, shelf_plate_thickness)
            box_size_ranges=self.box_size_ranges,
            num_samples=1,
            num_boxes=self.num_boxes_per_sample,
            allow_yaw=self.allow_book_yaw
        )

        for i, book_params in enumerate(sample):
            self._spawn_book(book_params[0], i)            
            

        # target at the middle of the shelf ending for goal evaluation
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
        gripper_pos = self.C.getFrame(self.gripper_name).getPosition()
        target_pos = self.C.getFrame("target").getPosition()
        
        distance = np.linalg.norm(gripper_pos - target_pos)
        success = distance < 0.05 # Tighter tolerance for reaching
        
        return {"distance_to_target": distance, "success": success}

    def collect_data(self):

        success = self.roboenv.pull_real("target_book_0", self.C.getFrame("target").getPosition(), accumulated_collisions=True, get_observation=True, base="table")
        if success:
            demo_group = self.h5file.create_group(f"demo_{self.demo_id}")

            se3_path = np.zeros((self.roboenv.path.shape[0], 9))

            C2 = ry.Config()
            C2.addConfigurationCopy(self.C)
            for i in range(self.roboenv.path.shape[0]):
                C2.setJointState(self.roboenv.path[i])
                ee_pose = C2.eval(ry.FS.pose, ["l_gripper"])[0]

                q = ry.Quaternion().set(ee_pose[3:])
                R = q.getMatrix()
                if "delta" in self.robot_mode:
                    se3_path[i, :3] = ee_pose[:3] - self.last_pos
                    self.last_pos = ee_pose[:3]

                else:
                    se3_path[i, :3] = ee_pose[:3]  # Position
                    se3_path[i, 3:9] = np.array([R[0:3, 0], R[0:3, 1]]).flatten()  # Rotation
 
            if self.robot_mode == "taskspace":
                demo_group.create_dataset("path", data=se3_path)
            elif self.robot_mode == "pos3d" or self.robot_mode == "pos3d_delta" or self.robot_mode == "pos3d_rel":
                demo_group.create_dataset("path", data=se3_path[:, :3])
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
            
            print(f"Collected Demo: {self.demo_id}")
            self.demo_id += 1

    def save_data(self):
        pass

    def getImageDepth(self):
        if self.botop:
            rgb, depth = self.bot.getImageAndDepth(self.camera_name)
        elif self.simulate:
            self.camview = ry.CameraView(self.C)
            self.camview.setCamera(self.C.getFrame(self.camera_name))

            rgb, depth = self.camview.computeImageAndDepth(self.C)
        return rgb, depth

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if not self.on_real:
            self.C.setJointState(self.q0)
            pass
        
        self._setup_scene()    

        if self.botop:
            self.bot = ry.BotOp(self.C, self.on_real)
            if self.on_real:
                self.bot.home(self.C)
                self.bot.moveTo(self.q0)
                while self.bot.getTimeToEnd() > 0:
                    self.bot.wait(self.C)
        elif self.simulate:
            self.sim = Simulator(self.C, engine=ry.SimulationEngine.physx, verbose=0, camera=self.camera_name)

        self.last_pos = self.C.getFrame(self.gripper_name).getPosition()
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info

    def step(self, action):
        # Your logic to apply an action to the environment
        # `action` will be a numpy array matching `self.action_space`
        #print(f"Executing action: {action}")
        
        if self.robot_mode == "floating":
            for _ in range(100):  # Simulate for 100 steps
                self.sim._sim.step([action[0], action[1], action[2]], 0.01, ry.ControlMode.position)
                self.C.view()
        elif self.robot_mode == "normal":
            if self.path_mode == "jointspace":
                for _ in range(100):
                    self.sim._sim.step(action, 0.01, ry.ControlMode.position)
            if self.path_mode == "taskspace" or self.path_mode == "pos3d" or self.path_mode == "pos3d_delta" or self.path_mode == "pos3d_rel":
                
                # clip minimum height for z to avoid collisions with table
                if "_delta" in self.robot_mode:
                    pass
                else:
                    if action[2] < 0.67:
                        action[2] = 0.67
                
                komo = ry.KOMO()
                komo.setConfig(self.C, False)
                komo.setTiming(1, 1, 1., 0)
                
                komo.clearObjectives()
                komo.addControlObjective([], 0, 1e-1)

                if self.robot_mode == "pos3d_delta":
                    komo.addObjective([], ry.FS.position, [self.gripper_name], ry.OT.sos, [1e2], action[:3] + self.last_pos)
                else:
                    komo.addObjective([], ry.FS.position, [self.gripper_name], ry.OT.sos, [1e2], action[:3])
                
                if self.robot_mode == "taskspace":
                    rot_matrix = gram_schmidt_orthonormalize(action[3:])
                    quat = ry.Quaternion().setMatrix(rot_matrix).asArr()

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
