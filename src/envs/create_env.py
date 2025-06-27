import numpy as np
import robotic as ry 
import h5py
from envs.shelf import generate_shelf
from envs.high_level_methods import RobotEnviroment
from envs.book_spawning import generate_random_box_params

# Still experimental

class ShelfPullDataCollector:
    def __init__(self,
                 robot_mode="floating",
                 collect_data=True,
                 path_mode="SE39D",
                 simulate=True,
                 camera_name="cameraStatic",
                 shelf_pos_xyz=None, # e.g. [.8, 0., .3]
                 shelf_quaternion=None, # e.g. [1, 0, 0, 1] (w,x,y,z) or as expected by generate_shelf
                 shelf_openings_small=None, # e.g. [4, 11]
                 shelf_equidistant=False,
                 num_samples_books=10,
                 num_boxes_per_sample=1,
                 allow_book_yaw=False,
                 box_size_ranges=None, # e.g. {'x': (.1, .15), 'y': (.14, .23), 'z': (.009, .045)}
                 h5_filename="variable_demo.h5"):

        self.robot_mode = robot_mode
        self.collect_data = collect_data
        self.path_mode = path_mode
        self.simulate = simulate
        self.camera_name = camera_name
        self.h5_filename = h5_filename

        self.C = ry.Config()
        self.prefix = "l_"
        self.gripper_name = "l_gripper"
        self.palm_name = "l_palm"

        # Configure robot based on mode
        if self.robot_mode == "normal":
            self.C.addFile(ry.raiPath('../rai-robotModels/scenarios/pandaSingle.g'))
            table = self.C.getFrame("table")
            if table: # Check if table frame exists
                table.setShape(ry.ST.ssBox, size=[.5, 1., .1, .005]).setColor(np.array([242, 240, 216]) / 255)
            
            # Safely delete frame if it exists
            coll_camera_wrist = self.C.getFrame("panda_collCameraWrist")
            if coll_camera_wrist:
                 self.C.delFrame("panda_collCameraWrist")

        elif self.robot_mode == "floating":
            self.C.addFile(ry.raiPath('../rai-robotModels/scenarios/pandaFloatingFixGripper.g'))
            self.gripper_name = "gripper"
            self.palm_name = "palm"
            self.prefix = ""
            
            # Set initial floating base position if joints exist
            current_q = self.C.getJointState()
            # Assuming the first 7 DoFs are for the floating base [x,y,z, qx,qy,qz,qw] or similar
            # This requires pandaFloatingFixGripper.g to define these base joints.
            if len(current_q) >= 7 : # Check if there are enough joints for a floating base
                offset = np.array([.0, 0, .2, 0, 0, 0, 0]) # x,y,z translation, identity rotation
                current_q[:len(offset)] += offset # Apply to the base part of the joint state
                self.C.setJointState(current_q)
            elif len(current_q) > 0: # If some joints exist but not enough for full base, this might be an issue
                print(f"Warning: Floating robot mode, but initial joint state length is {len(current_q)}. Expected >= 7 for base.")
                # Potentially set a default base pose if appropriate
                # For now, we'll proceed, assuming the .g file and joint state are somewhat aligned.
            else: # No joints defined, set a default initial pose for the floating base
                 self.C.setJointState(np.array([.0, 0, .2, 0, 0, 0, 0])) # x,y,z, and 4 for quaternion (identity)
        else:
            raise ValueError(f"Unknown ROBOT_MODE: {self.robot_mode}")

        # Gripper friction
        finger1 = self.C.getFrame(self.prefix + "finger1")
        finger2 = self.C.getFrame(self.prefix + "finger2")
        if finger1: finger1.setAttribute("friction", 1e5)
        if finger2: finger2.setAttribute("friction", 1e5)
        
        self.q0 = self.C.getJointState().copy() # Store initial joint state

        # Shelf setup
        self.shelf_pos = np.array(shelf_pos_xyz) if shelf_pos_xyz is not None else np.array([.8, 0., .3])
        _shelf_quaternion = shelf_quaternion if shelf_quaternion is not None else [1, 0, 0, 1]
        _shelf_openings_small = shelf_openings_small if shelf_openings_small is not None else [4, 11]
        
        generate_shelf(self.C, self.shelf_pos, base_quaternion=_shelf_quaternion,
                       openings_small=_shelf_openings_small, equidistant=shelf_equidistant)

        self.C.addFrame("cameraWP", self.camera_name).setShape(ry.ST.marker, [.1])

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

        self.box_size_ranges = box_size_ranges if box_size_ranges is not None else {
            'x': (.1, .15),    # Book width
            'y': (.14, .23),   # Book depth
            'z': (.009, .045), # Book thickness
        }
        self.num_samples_books = num_samples_books
        self.num_boxes_per_sample = num_boxes_per_sample
        self.allow_book_yaw = allow_book_yaw

        # Shelf corner for book positioning logic (bottom-left corner, Z at center of plate)
        shelf_center_pos = self.shelf_bottom_frame.getPosition()[:3]
        self.shelf_corner_ref_point = shelf_center_pos + np.array([-self.shelf_width/2, -self.shelf_depth/2, 0])

        self.h5file = None
        self.demo_id = 0
        if self.collect_data:
            self.h5file = h5py.File(self.h5_filename, "w")
            print(f"Initialized HDF5 output to: {self.h5_filename}")

    def _spawn_books(self, sample_book_config_list):
        """Spawns books based on the provided configuration list for a single sample."""
        spawned_book_frames = []
        for i, book_params in enumerate(sample_book_config_list):
            # book_params: [size_x, size_y, size_z, pos_x_on_shelf, pos_y_on_shelf, yaw_angle]
            b_size_x, b_size_y, b_size_z, b_pos_x, b_pos_y, b_pos_z, b_yaw = book_params

            # Z-offset for book center relative to shelf_corner_ref_point's Z:
            # Places bottom of book on top surface of shelf plate.
            # shelf_corner_ref_point's Z is shelf_plate_center_z.
            # Top surface Z = shelf_plate_center_z + shelf_plate_thickness/2.
            # Book center Z = top_surface_z + b_size_z/2.
            # So, book_center_z = shelf_plate_center_z + shelf_plate_thickness/2 + b_size_z/2
            #                 = shelf_corner_ref_point[2] + (self.shelf_plate_thickness + b_size_z) / 2
            z_offset = (self.shelf_plate_thickness + b_size_z) / 2
            
            book_center_position = self.shelf_corner_ref_point + np.array([b_pos_x, b_pos_y, z_offset])
            
            q_orientation = ry.Quaternion().setRollPitchYaw([0, 0, b_yaw]) # Yaw around Z
            
            frame_name = f"target_book_{i}"
            self.C.addFrame(frame_name) \
                .setPosition(book_center_position) \
                .setQuaternion(q_orientation.asArr()) \
                .setShape(ry.ST.ssBox, size=[b_size_x, b_size_y, b_size_z, 0.005]) \
                .setColor(np.random.rand(3)) \
                .setContact(1) \
                .setMass(.1)
            spawned_book_frames.append(frame_name)
        return spawned_book_frames

    def spawn_books_scene(self, num_boxes=1, allow_yaw=False):
        """ Spawn a single book scene  """

        samples = generate_random_box_params(
            shelf_size=self.shelf_dims_for_spawning, # (shelf_width, shelf_depth, shelf_plate_thickness)
            box_size_ranges=self.box_size_ranges,
            num_samples=1,
            num_boxes=num_boxes,
            allow_yaw=allow_yaw
        )

        spawned_book_names = self._spawn_books(samples[0])


    def run_experiment(self, view_simulation_steps=True):
        """Runs the data collection experiment."""
        if view_simulation_steps:
            self.C.view() # Initial view of the setup

        # Generate all book configurations
        samples = generate_random_box_params(
            shelf_size=self.shelf_dims_for_spawning, # (shelf_width, shelf_depth, shelf_plate_thickness)
            box_size_ranges=self.box_size_ranges,
            num_samples=self.num_samples_books,
            num_boxes=self.num_boxes_per_sample,
            allow_yaw=self.allow_book_yaw
        )
        print(f"Generated {len(samples)} book configuration samples.")

        for sample_idx, current_sample_book_configs in enumerate(samples):
            print(f"\n--- Processing Sample {sample_idx + 1}/{len(samples)} ---")
            
            # Spawn books for the current sample
            # The original script implicitly targets "target_book_0"
            spawned_book_names = self._spawn_books(current_sample_book_configs)
            if not spawned_book_names:
                print("No books spawned for this sample. Skipping.")
                continue
            
            # We will target the first book spawned in this sample, as per original logic
            manipulation_target_book_name = spawned_book_names[0]
            target_book_frame = self.C.getFrame(manipulation_target_book_name)
            if not target_book_frame:
                print(f"Error: Could not find target book frame '{manipulation_target_book_name}'. Skipping sample.")
                # Clean up any other books from this sample if needed, though they'll be cleared later.
                continue


            if view_simulation_steps: self.C.view(True, f"Sample {sample_idx+1}: Books spawned, target is {manipulation_target_book_name}")

            # Define the pull target position (front-center of the shelf, at book's Z height)
            pull_destination_pos = np.array([
                self.shelf_bottom_frame.getPosition()[0] - self.shelf_depth / 2, # X: front edge of shelf
                self.shelf_bottom_frame.getPosition()[1],                        # Y: center of shelf width
                target_book_frame.getPosition()[2]                               # Z: current Z of target book center
            ])
            
            # Add a visual marker for the pull destination
            if self.C.getFrame("pull_destination_marker"): self.C.delFrame("pull_destination_marker")
            self.C.addFrame("pull_destination_marker").setShape(ry.ST.marker, .05).setPosition(pull_destination_pos).setColor([1,0,0]) # Red marker

            if view_simulation_steps: self.C.view(True, f"Sample {sample_idx+1}: Pull target marker set for {manipulation_target_book_name}")

            # Initialize RobotEnvironment for the current scene configuration
            roboenv = RobotEnviroment(self.C, sim=self.simulate, gripper=self.gripper_name)

            print(f"Attempting pull for '{manipulation_target_book_name}' towards {pull_destination_pos}...")
            success = roboenv.pull(manipulation_target_book_name, pull_destination_pos,
                                   accumulated_collisions=False, # Parameter from original script
                                   capture_points=self.collect_data)

            if success:
                print(f"Pull successful for '{manipulation_target_book_name}'.")
                if self.collect_data and self.h5file:
                    demo_group_name = f"demo_{self.demo_id}"
                    demo_group = self.h5file.create_group(demo_group_name)
                    print(f"Saving data to HDF5 group: {demo_group_name}")

                    path_data_to_save = None
                    if hasattr(roboenv, 'path') and roboenv.path is not None and roboenv.path.ndim == 2:
                        if self.path_mode == "JOINT7D":
                            path_data_to_save = roboenv.path
                        elif self.path_mode == "SE39D":
                            # Path expected to be [pos (3), quat_wxyz (4), ...]
                            if roboenv.path.shape[1] >= 7: # Need at least x,y,z, w,x,y,z
                                se3_path = np.zeros((roboenv.path.shape[0], 9))
                                for i in range(roboenv.path.shape[0]):
                                    pos = roboenv.path[i, :3]
                                    quat_wxyz = roboenv.path[i, 3:7] # Assuming w,x,y,z order
                                    
                                    q_ry = ry.Quaternion()
                                    q_ry.set(quat_wxyz) # sets w,x,y,z
                                    R = q_ry.getMatrix()
                                    
                                    se3_path[i, :3] = pos
                                    se3_path[i, 3:9] = np.array([R[0:3, 0], R[0:3, 1]]).flatten() # First two columns of R
                                path_data_to_save = se3_path
                            else:
                                print(f"Warning: Path mode SE39D, but roboenv.path has shape {roboenv.path.shape}. Expected at least 7 columns. Path not saved.")
                        elif self.path_mode == "SE38D":
                            print("Warning: SE38D path mode is not implemented. Path not saved.")
                        else:
                            print(f"Warning: Unknown path_mode '{self.path_mode}'. Path not saved.")
                    else:
                         print("Warning: roboenv.path not found, is None, or not 2D. Path data not saved.")


                    if path_data_to_save is not None:
                        demo_group.create_dataset("path", data=path_data_to_save)
                    
                    if hasattr(roboenv, 'points') and roboenv.points is not None:
                        # Assuming roboenv.points could be a list of point clouds or a single one.
                        # Original script saved roboenv.points[0] to a .npy file.
                        # For HDF5, let's be flexible or decide on a convention.
                        # If it's a list, maybe save each as a dataset or concatenate if appropriate.
                        # For now, if it's a list, save the first element, similar to original.
                        points_data = roboenv.points
                        if isinstance(points_data, list):
                            if len(points_data) > 0:
                                points_data = points_data[0] # Take the first point cloud if list
                            else:
                                points_data = None # Empty list
                        
                        if isinstance(points_data, np.ndarray):
                            demo_group.create_dataset("points", data=points_data)
                        elif points_data is not None:
                            print(f"Warning: roboenv.points is of type {type(points_data)}, expected numpy array or list of arrays. Points not saved.")
                    else:
                        print("Warning: roboenv.points attribute not found or is None. Points data not saved.")
                    
                    self.demo_id += 1
            else:
                print(f"Pull failed for '{manipulation_target_book_name}'.")

            # Cleanup: Delete all spawned books from this sample
            for book_name in spawned_book_names:
                if self.C.getFrame(book_name):
                    self.C.delFrame(book_name)
            if self.C.getFrame("pull_destination_marker"): 
                self.C.delFrame("pull_destination_marker")

            if view_simulation_steps: self.C.view(True, f"Sample {sample_idx+1}: Cleaned up books.")

            # Reset robot to its initial joint state
            self.C.setJointState(self.q0)
            
            # Reset finger joint specifically (if applicable for the robot model)
            # Ensure the joint name is correct for your robot model and prefix settings
            finger_joint_name = self.prefix + 'panda_finger_joint1' if self.prefix else 'panda_finger_joint1'
            # For Franka Panda, the gripper joint is often just 'panda_finger_joint1' without 'l_' if prefix is empty.
            # If using 'pandaFloatingFixGripper.g', prefix is "", so it would be 'panda_finger_joint1'
            # If using 'pandaSingle.g', prefix is "l_", so 'l_panda_finger_joint1'
            
            finger_joint = self.C.getFrame(finger_joint_name)
            if finger_joint:
                finger_joint.setJointState(np.array([.01])) # Example open position for Panda
            else:
                # Fallback for common alternative name if prefix logic was tricky
                alt_finger_joint_name = 'panda_finger_joint1' 
                if self.prefix and not finger_joint: # If prefix was used and main one not found
                    finger_joint = self.C.getJoint(alt_finger_joint_name)
                if finger_joint:
                    finger_joint.setJointState(np.array([.01]))
                # else:
                #     print(f"Warning: Could not find finger joint '{finger_joint_name}' or '{alt_finger_joint_name}' to reset.")


            if view_simulation_steps: self.C.view(True, f"Sample {sample_idx+1}: Robot reset to initial state.")

        print("\nAll samples processed.")
        if view_simulation_steps and hasattr(self.C, 'view_window') and self.C.view_window:
            print("Closing simulation view.")
            self.C.viewClose()

    def render(self, n_samples=4096):
        roboenv = RobotEnviroment(self.C, sim=self.simulate, gripper=self.gripper_name)
        points = roboenv.render(n_samples)
        return points

    def close(self):
        """Closes any open resources, like the HDF5 file."""
        if self.h5file:
            self.h5file.close()
            print(f"HDF5 file '{self.h5_filename}' closed.")
        # Close ry view window if it's still open and managed by this class instance
        if hasattr(self.C, 'view_window') and self.C.view_window is not None:
            try:
                self.C.viewClose()
                print("Robotic configuration view closed.")
            except Exception as e:
                print(f"Error closing robotic view: {e}")


# --- Example Usage ---
if __name__ == "__main__":
    # Define parameters for the data collection
    # These would typically come from a config file or script arguments
    experiment_config = {
        "robot_mode": "floating",  # "normal" or "floating"
        "collect_data": False,
        "path_mode": "SE39D",     # "JOINT7D", "SE38D", or "SE39D"
        "simulate": True,         # Whether the RobotEnvironment runs in simulation
        "camera_name": "cameraStatic", # or "cameraWrist"
        
        "shelf_pos_xyz": [.7, 0.05, .25], # Custom shelf position
        "shelf_quaternion": [1, 0, 0, 1], # Identity quaternion (w,x,y,z) for shelf orientation
        "shelf_openings_small": [4, 11],   # Shelf structure params for generate_shelf
        "shelf_equidistant": False,
        
        "num_samples_books": 10,        # Number of different book arrangements to try
        "num_boxes_per_sample": 1,     # Number of books per arrangement (original script used 1 active book)
        "allow_book_yaw": False,       # Allow books to have a random yaw
        "box_size_ranges": {           # Define ranges for book dimensions [width, depth, thickness]
            'x': (.07, .1),          # Book width (along shelf width)
            'y': (.1, .15),          # Book depth (along shelf depth)
            'z': (.01, .025),        # Book thickness (height)
        },
        "h5_filename": "my_shelf_pull_data.h5" # Output HDF5 filename
    }

    collector = None  # Initialize for finally block
    try:
        print("Initializing ShelfPullDataCollector...")
        collector = ShelfPullDataCollector(**experiment_config)
        
        print("Initialization complete. Starting data collection experiment...")
        collector.run_experiment(view_simulation_steps=True) # Set to False for headless execution
        
        print("Data collection experiment finished.")

    except Exception as e:
        print(f"An error occurred during the experiment: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if collector:
            print("Closing collector resources...")
            collector.close()
            print("Collector resources closed.")