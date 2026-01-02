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

class TableEnv(BaseRobotEnv):
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
                q0=[.0, .0, .0, -2., 0. ,2., -0.5],
                #domain randomization parameters
                table_offset_ranges = None,
                camera_offset_ranges = None,
                camera_rpy_ranges = None,
                focal_length_range = (1.5, 1.5),
                depth_noise_ranges = None,
                obj = "book",
                task = "pull",
                save_obj_pos=False,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.box_size_ranges = box_size_ranges
        self.box_offset_ranges = box_offset_ranges
        self.allow_book_yaw = allow_book_yaw
        self.books = []

        self.path_type = path_type
        self.img_type = img_type
        self.extras = extras
        self.q0 = np.array(q0)
        self.obj = obj
        self.obj_center = np.array([0., .34])
        self.task = task
        self.save_obj_pos = save_obj_pos

        self.camera_name = camera_name
        self.last_pos = np.array([0., 0., 0.])

        if not self.on_real:
            self.C.setJointState(self.q0)
        self.table_base_dims = np.array([1.2, 1.1, .1])  # Default table dimensions (width, depth, height)

        self.camera_base_pos = self.C.getFrame(self.camera_name).getPosition()
        self.camera_base_rpy = ry.Quaternion().set(self.C.getFrame(self.camera_name).getQuaternion()).getRollPitchYaw()

        #self.camera_base_pos = np.array([0., .56, 1.57]) 

        # Domain randomization parameters
        self.table_offset_ranges = table_offset_ranges
        if table_offset_ranges is not None:
            self.table_width_range = table_offset_ranges.width
            self.table_length_range = table_offset_ranges.length
            self.table_yaw_range = table_offset_ranges.yaw  # degrees
        
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

        self.C.getFrame("table").setShape(self.C.getFrame("table").getShapeType(), [1.2, 1.1, .1, .01]).setColor(np.array([242, 240, 216]) / 255)
        self.C.getFrame("l_panda_base").setPosition(self.C.getFrame("l_panda_base").getPosition() + np.array([0, -.08, .0]))
    
        self.C.getFrame("cameraStaticTableTop") \
            .setPosition(self.camera_base_pos) \
            .setQuaternion(ry.Quaternion().setRollPitchYaw([0, np.pi, np.pi]).asArr()) \

        self.table_base_height = self.C.getFrame("table").getPosition()[2] + self.C.getFrame("table").getSize()[2]/2


        if self.img_type.upper() == "BOX_POINTS":
            box_mask_height = .2
            box_mask_width = 1.2
            box_mask_depth = 1.75
            pos_offset_x = 0
            pos_offset_y = 0
            if self.on_real:
                pos_offset_z = .03
            else:
                pos_offset_z = .01

            self.C.addFrame("BOX_MASK") \
                .setShape(ry.ST.box, size=[box_mask_width, box_mask_depth, box_mask_height]) \
                .setColor([1, 0, 0, .1]) \
                .setPosition(self.C.getFrame("table").getPosition()+np.array([pos_offset_x, pos_offset_y, pos_offset_z+self.C.getFrame("table").getSize()[2]/2+box_mask_height/2])) \
                
        if collect_data:    # TODO parameters
            self.h5file = h5py.File("table_demo.h5", "w")
            self.roboenv = RobotEnviroment(self.C, sim=self.simulate, gripper=self.gripper, observation_mode=self.img_type, visualize=False, path_mode="SE39D", camera=self.camera_name, depth_noise=self.depth_noise_active)
            self.demo_id = 0
            
        self._setup_scene()

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

            self.waypoint_pos = self.C.getFrame("waypoint_marker").getPosition()

        self.C.addFrame("target").setPosition([.2, .3, .7]).setShape(ry.ST.marker, [.2]).setColor([0, 1, 0, .9])

        self.C.view(False)

    def _draw_arena(self):
        radius, height = 0.04, 0.03

        center = np.array([0, 0, self.table_base_height]) + np.concatenate((self.obj_center, np.array([0])))

        corners = [
            (self.box_offset_ranges['x'][0], self.box_offset_ranges['y'][0]),  # (-y, +x)
            (self.box_offset_ranges['x'][0], self.box_offset_ranges['y'][1]),  # (+y, +x)
            (self.box_offset_ranges['x'][1], self.box_offset_ranges['y'][0]),  # (-y, -x)
            (self.box_offset_ranges['x'][1], self.box_offset_ranges['y'][1])   # (+y, -x)
        ]

        for i, (dx, dy) in enumerate(corners):
            self.C.addFrame(f"arena_corner_{i}") \
                .setShape(ry.ST.marker, size=[.1]) \
                .setPosition(center + np.array([dx, dy, 0]))


    def _draw_arena_grid(self):
        # 1. Define the number of points you want along each axis
        num_x = 20   # Adjust as needed
        num_y = 13  # Adjust as needed

        center = np.array([0, 0, self.table_base_height]) + np.concatenate((self.obj_center, np.array([0])))

        # 2. Generate the ranges using linspace
        # This creates arrays of evenly spaced numbers from min to max
        x_range = np.linspace(self.box_offset_ranges['x'][0], self.box_offset_ranges['x'][1], num_x)
        y_range = np.linspace(self.box_offset_ranges['y'][0], self.box_offset_ranges['y'][1], num_y)

        # 3. Create the grid
        # xx and yy will be matrices containing all coordinate pairs
        xx, yy = np.meshgrid(x_range, y_range)

        # Flatten them to iterate easily
        grid_points = np.column_stack((xx.ravel(), yy.ravel()))

        

        # 4. Iterate and draw
        for i, (dx, dy) in enumerate(grid_points):
            # We add 'arena_grid_' prefix to keep names unique
            self.C.addFrame(f"arena_grid_{i}") \
                .setShape(ry.ST.marker, size=[.05]) \
            .setPosition(center + np.array([dx, dy, 0])) 

    def _spawn_cylinder_scene(self, center=None):
        radius, height = 0.04, 0.03
        
        if center is  None:
            center = np.array([0, 0, self.table_base_height]) + np.concatenate((self.obj_center, np.array([height/2]))) + np.array([ 
                np.random.uniform(self.box_offset_ranges['x'][0], self.box_offset_ranges['x'][1]),
                np.random.uniform(self.box_offset_ranges['y'][0], self.box_offset_ranges['y'][1]), 
                0])

        self.books.append("cylinder")
        self.C.addFrame("cylinder") \
            .setPosition(center) \
            .setShape(ry.ST.cylinder, size=[height, radius]) \
            .setColor([0, 0, 0]) \
            .setContact(1) \
            .setMass(.1) \
            .setAttributes({"friction": .01}) 
        
        if self.save_obj_pos:
            self.cylinder_pos = self.C.getFrame("cylinder").getPosition()

        self.C.addFrame("target").setPosition(np.concatenate((self.obj_center, np.array([.7])))).setShape(ry.ST.marker, [.2]).setColor([0, 1, 0, .9])

        self.C.view(False)

    def _spawn_books_scene(self):
        sample = generate_random_box_sizes(
            box_size_ranges=self.box_size_ranges,
            num_samples=1,
        )
        for i, book_params in enumerate(sample):
            self._spawn_book(book_params, i)            


    def _delete_books(self):
        for book in self.books:
            self.C.delFrame(book)
        if self.C.getFrame("target") is not None:
            self.C.delFrame("target")
            self.C.delFrame("target_p")
        if self.C.getFrame("waypoint_marker") is not None:
            self.C.delFrame("waypoint_marker")
        self.C.view(False)
        self.books = []

    def _define_action_space(self):
        if self.robot_mode == "floating":
            return spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        elif self.path_mode == "jointspace":
            return spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        elif self.path_mode == "taskspace":
            return spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        elif self.path_mode == "posyaw":
            return spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        else:
            return spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

    def _setup_scene(self, obj_pos=None):
        # target_pos = np.array([.5, 0.1, .7]) 
        
        self._delete_books()
        if self.obj == "book":
            self._spawn_books_scene()
        elif self.obj == "cylinder":
            self._spawn_cylinder_scene(center=obj_pos)
        # self.C.getFrame("reach_target").setPosition(target_pos)


    def _get_info(self):
        obj_pos = self.C.getFrame("cylinder").getPosition()
        target_pos = self.C.getFrame("target").getPosition()
        
        distance = np.linalg.norm(obj_pos[:2] - target_pos[:2])
        success = distance < 0.05 # Tighter tolerance for reaching
        
        return {"distance_to_target": distance, "success": success}

    def push_block(self):
        success = self.roboenv.push_frame_to("target_book_0", [0.2, .3, 0])
    
    def push_cylinder(self):
        success = self.roboenv.push_frame_to("cylinder", self.C.getFrame("target").getPosition(), get_observation=True)
        if success:
            self.waypoint_pos = self.roboenv.ways
        return success

    def collect_data(self):

        if self.task == "pull":
            success = self.roboenv.pull_real("target_book_0", self.C.getFrame("target").getPosition(), accumulated_collisions=True, get_observation=True, base="table")
        elif self.task == "push":
            success = self.push_cylinder()

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
 
            if self.path_mode == "taskspace":
                demo_group.create_dataset("path", data=se3_path)
            elif self.path_mode == "pos3d" or self.robot_mode == "pos3d_delta" or self.robot_mode == "pos3d_rel":
                demo_group.create_dataset("path", data=se3_path[:, :3])
            elif self.path_mode == "posyaw":
                yaw_path = np.zeros((self.roboenv.path.shape[0], 1))
                for i in range(self.roboenv.path.shape[0]):
                    C2.setJointState(self.roboenv.path[i])
                    sin_comp = C2.eval(ry.FS.scalarProductXY, ["l_gripper", "table"])[0]
                    cos_comp = C2.eval(ry.FS.scalarProductXX, ["l_gripper", "table"])[0]
                    yaw_path[i] = np.arctan2(sin_comp, cos_comp)
                demo_group.create_dataset("path", data=np.hstack((se3_path[:, :3], yaw_path)))
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
            
            if self.save_obj_pos:
                if self.obj == "cylinder":
                    demo_group.create_dataset("cylinder", data=self.cylinder_pos)
                    
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

        if self.table_offset_ranges is not None:
            table_width = self.table_base_dims[0] + np.random.uniform(self.table_width_range[0], self.table_width_range[1])
            table_length = self.table_base_dims[1] + np.random.uniform(self.table_length_range[0], self.table_length_range[1])
            table_yaw = np.random.uniform(self.table_yaw_range[0], self.table_yaw_range[1])

            self.C.getFrame("table").setShape(ry.ST.ssBox, [table_width, table_length, self.C.getFrame("table").getSize()[2], self.C.getFrame("table").getSize()[3]])
            self.C.getFrame("table").setQuaternion([np.cos(np.deg2rad(table_yaw/2)), 0, 0,  np.sin(np.deg2rad(table_yaw/2))]) 

        if not self.on_real:
            self.C.setJointState(self.q0)
            pass
        
        obj_pos = None
        get_obs = True
        if options is not None:
            obj_pos = options.get("obj_pos", None)
            get_obs = options.get("get_obs", True)

        self._setup_scene(obj_pos=obj_pos)    
        if self.botop:
            self.bot = ry.BotOp(self.C, self.on_real)
            if self.on_real:
                self.bot.home(self.C)
                self.bot.moveTo(self.q0)
                while self.bot.getTimeToEnd() > 0:
                    self.bot.wait(self.C)
                self.bot.gripperMove(ry._left, 0)
                while not self.bot.gripperDone(ry._left):
                    self.bot.wait(self.C)
        elif self.simulate:
            self.sim = Simulator(self.C, engine=ry.SimulationEngine.physx, verbose=0, camera=self.camera_name)

        self.last_pos = self.C.getFrame(self.gripper_name).getPosition()
        if get_obs:
            observation = self._get_obs()
        else:
            #TODO
            observation = {}
            observation["depth"] = None
            observation["agent_pos"] = None

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
        elif self.path_mode == "jointspace":
            for _ in range(100):
                self.sim._sim.step(action, 0.01, ry.ControlMode.position)
        elif self.path_mode == "taskspace" or self.path_mode == "pos3d" or self.path_mode == "pos3d_delta" or self.path_mode == "pos3d_rel" or self.path_mode == "posyaw":
            
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

                komo.addObjective([], ry.FS.quaternion, [self.gripper_name], ry.OT.sos, [1e2], quat)
            elif self.path_mode == "posyaw":
                komo.addObjective([], ry.FS.vectorZ, [self.gripper_name], ry.OT.eq, [1], [0, 0, 1])
                target_yaw = action[3]

                # 2. Calculate the target scalar products
                target_cos = np.cos(target_yaw)
                target_sin = np.sin(target_yaw)

                komo.addObjective([], ry.FS.scalarProductXX, [self.gripper_name, "table"], ry.OT.sos, [1e1], [target_cos])
                komo.addObjective([], ry.FS.scalarProductXY, [self.gripper_name, "table"], ry.OT.sos, [1e1], [target_sin])
            
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
        else:
            self.C.setJointState([action[0], action[1], action[2]])  # Assuming the first 7 values are joint angles

        # --- After action, get the new results ---
        observation = self._get_obs()
        reward = 1.0 # TODO Your logic for calculating reward, if even necessary
        terminated = False # Your logic for whether the episode has ended (e.g., task success)
        truncated = False # Your logic for whether the episode was cut short (e.g., time limit)
        info = self._get_info()
        
        self.C.view(False)  # Update the view after the action
        # The step function MUST return these five values in this order
        return observation, reward, terminated, truncated, info
