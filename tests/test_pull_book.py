import numpy as np
import robotic as ry
import h5py
from envs.shelf import generate_shelf
from envs.high_level_methods import RobotEnviroment
from envs.book_spawning import generate_random_box_params

ROBOT_MODE = "floating" # "normal" or "floating"
COLLECT_DATA = False
PATH_MODE = "RegressPC2Pos" # "JOINT7D", "SE39D", "POS3D", "DELTA3D", "RegressPC2Pos"
SIMULATE = False 
CAMERA = "cameraStatic"  # or "cameraWrist"
BASE_REMOVAl = False # if true, shelf will be removed from observation

prefix = "l_"
C = ry.Config()

gripper = "l_gripper"
palm = "l_palm"

if ROBOT_MODE == "normal":
    C.addFile(ry.raiPath('../rai-robotModels/scenarios/pandaSingle.g'))
    C.getFrame("table").setShape(ry.ST.ssBox, size=[.5, 1, .1, .005]).setColor(np.array([242, 240, 216])/255)   # Real size [1.1, 1.2, .02, .005]
    C.delFrame("panda_collCameraWrist")

elif ROBOT_MODE == "floating":
    C.addFile(ry.raiPath('../rai-robotModels/scenarios/pandaFloatingFixGripper.g'))
    gripper = "gripper"
    palm = "palm"
    C.setJointState(C.getJointState() + np.array([.3, 0, .2, 0, 0, 0, 0]))
    prefix = ""

C.getFrame(prefix+"finger1").setAttribute("friction", 1e5)
C.getFrame(prefix+"finger2").setAttribute("friction", 1e5)

q0 = C.getJointState()

# Shelf
pos = np.array([.8, 0., .3])
generate_shelf(C, pos, base_quaternion=[1, 0, 0, 1], openings_small=[4, 11], equidistant=False)

C.addFrame("cameraWP", CAMERA).setShape(ry.ST.marker, [.1]) 
C.view(False)

# Frame in use for book manipulations
shelfBottomFrame = C.getFrame("big_xy_bottom_0_1")

shelf_depth = shelfBottomFrame.getSize()[1]
shelf_width = shelfBottomFrame.getSize()[0]
shelf_height = shelfBottomFrame.getSize()[2]

# Example usage
shelf_size = (shelfBottomFrame.getSize()[0], shelfBottomFrame.getSize()[1], shelfBottomFrame.getSize()[2])  # Fixed shelf dimensions (X_s, Y_s, Z_s)

box_size_ranges = {  # Variable box dimensions
    'x': (.125, .125),  # X_b range
    'y': (.18, .18),  # Y_b range
    'z': (.02, .02),   # Z_b range
}

samples = generate_random_box_params(shelf_size, box_size_ranges, num_samples=2000, num_boxes= 1, allow_yaw=False)

shelf_corner = np.array([
    (shelfBottomFrame.getPosition()[:3] + np.array([-shelf_depth/2, -shelf_width/2, 0])),
])

demo_id = 0

if COLLECT_DATA:
    h5file = h5py.File("variable_demo.h5", "w")


for sample in samples:
    for book_params in sample:
        q = ry.Quaternion().setRollPitchYaw(([0,0, book_params[-1]]))

        for i, box in enumerate(sample):
            q = ry.Quaternion().setRollPitchYaw(([0,0, box[-1]]))
            C.addFrame(f"target_book_{i}") \
                .setPosition(shelf_corner + np.append(box[3:5], (shelf_height+box[2])/2)) \
                .setQuaternion(q.asArr()) \
                .setShape(ry.ST.ssBox, size=[box[0], box[1], box[2], 0.005]) \
                .setColor(np.random.rand(3)) \
                .setContact(1) \
                .setMass(.1)
        C.view(False)

        
        # target at the middle of the shelf ending
        target = np.array([
            (shelfBottomFrame.getPosition()[:2] + np.array([-shelf_depth/2, 0])),
        ])
        target = np.append(target, C.getFrame("target_book_0").getPosition()[2])

        C.addFrame("target").setShape(ry.ST.marker, .1).setPosition(target)


        roboenv = RobotEnviroment(C, sim=SIMULATE, gripper=gripper, base_removal=BASE_REMOVAl)

        success = roboenv.pull("target_book_0", target, accumulated_collisions=False, capture_points=COLLECT_DATA)
        

        print(q0)
        C.view(True)
        quit()

        if success and COLLECT_DATA:
            #np.save("pc.npy", roboenv.points[0])

            demo_group = h5file.create_group(f"demo_{demo_id}")

            if PATH_MODE == "JOINT7D":
                demo_group.create_dataset("path", data=roboenv.path)
            if PATH_MODE == "SE39D":
                se3_path = np.zeros((roboenv.path.shape[0], 9))
                
                for i in range(roboenv.path.shape[0]):
                    q = ry.Quaternion().set(roboenv.path[i][3:])
                    R = q.getMatrix()
                    # Combine position (3D) with first two rotation matrix columns (6D)
                    se3_path[i, :3] = roboenv.path[i][:3]  # Position
                    se3_path[i, 3:9] = np.array([R[0:3, 0], R[0:3, 1]]).flatten()  # Rotation
                
                # Now use se3_path instead of the original path
                demo_group.create_dataset("path", data=se3_path)

            elif PATH_MODE == "POS3D":
                demo_group.create_dataset("path", data=roboenv.path[:, :3])  # Only position

            elif PATH_MODE == "DELTA3D":
                delta_paths = np.empty((64, 3))
                delta_paths[0] = roboenv.path[0][:3]-C.getJointState()[:3]
                for i in range(1, roboenv.path.shape[0]):
                    delta_paths[i] = roboenv.path[i][:3] - roboenv.path[i-1][:3]
                demo_group.create_dataset("path", data=delta_paths)  # Only delta positions

            elif PATH_MODE == "RegressPC2Pos":
                print(roboenv.path.shape)
                demo_group.create_dataset("path", data=roboenv.path[31, :3])  # Only position

            
            demo_group.create_dataset("points", data=roboenv.points)
            demo_id += 1


        C.delFrame(f"target_book_0")
        C.view(False)
        C.setJointState(q0)
        
        C.getFrame(prefix+'panda_finger_joint1').setJointState(np.array([.01]))

if COLLECT_DATA:
    h5file.close()