import numpy as np
import robotic as ry
import h5py
from envs.shelf import generate_shelf
from envs.high_level_methods import RobotEnviroment
from envs.book_spawning import generate_random_box_params

ROBOT_MODE = "floating" # "normal" or "floating"
SIMULATE = True 
CAMERA = "cameraWrist"  # or "cameraWrist"
BASE_REMOVAl = False # if true, shelf will be removed from observation
DEBUG = False # pull debugging
OBSERVATION_MODE = "DEPTH" # "POINTCLOUD", "RGB", "DEPTH"
COMPRESS = True
RANDOM_COLOR = False
NUM_SAMPLES = 600
VISUALIZE = False  # If true, the simulation will be visualized
SAVE_BOOK_PARAMS = False  # If true, the parameters of the generated books will be saved to a file

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
    C.setJointState(C.getJointState() + np.array([.0, 0, .2]))
    prefix = ""

q0 = C.getJointState()

# World Camera pose
camera_quat = ry.Quaternion().setRollPitchYaw([-np.pi/2, np.pi/2, 0]) * ry.Quaternion().setRollPitchYaw([-.1, 0, 0])
C.addFrame("worldCamera").setShape(ry.ST.camera, [.1]).setAttribute("focalLength", .895).setPosition([-.5, 0, 1.5]).setQuaternion(camera_quat.asArr())
C.view_setCamera(C.getFrame("worldCamera"))

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
    'x': (.1, .15),  # X_b range
    'y': (.14, .23),  # Y_b range
    'z': (.009, .045),   # Z_b range
}

samples = generate_random_box_params(shelf_size, box_size_ranges, num_samples=NUM_SAMPLES, num_boxes= 1, allow_yaw=False)

shelf_corner = np.array([
    (shelfBottomFrame.getPosition()[:3] + np.array([-shelf_depth/2, -shelf_width/2, 0])),
])

demo_id = 0
err = []


C2 = ry.Config()
C2.addConfigurationCopy(C)

for sample in samples:
    for book_params in sample:
        for i, box in enumerate(sample):
            q = ry.Quaternion().setRollPitchYaw(([0,0, box[-1]]))
            C.addFrame(f"target_book_{i}") \
                .setPosition(shelf_corner + np.append(box[3:5], (shelf_height+box[2])/2)) \
                .setQuaternion(q.asArr()) \
                .setShape(ry.ST.ssBox, size=[box[0], box[1], box[2], 0.005]) \
                .setColor(np.random.rand(3) if RANDOM_COLOR else [1, 0, 0]) \
                .setContact(0) \
                .setMass(.1) \
                .setAttribute("friction", .01) 
        C.view(False)

        target = np.array([
            (shelfBottomFrame.getPosition()[:2] + np.array([-shelf_depth/2, 0])),
        ])
        target = np.append(target, C.getFrame("target_book_0").getPosition()[2])

        C.addFrame("target").setShape(ry.ST.marker, .1).setPosition(target)


        roboenv = RobotEnviroment(C, sim=SIMULATE, gripper=gripper, base_removal=BASE_REMOVAl, observation_mode=OBSERVATION_MODE, visualize=VISUALIZE)

        success = roboenv.pull_real("target_book_0", target, accumulated_collisions=True)
        
        if success:
            
            C2.addFrame(f"{demo_id}") \
                .setPosition(shelf_corner + np.append(box[3:5], (shelf_height+box[2])/2)) \
                .setQuaternion(q.asArr()) \
                .setShape(ry.ST.ssBox, size=[box[0], box[1], box[2], 0.005]) \
                .setColor(np.random.rand(3) if RANDOM_COLOR else [1, 0, 0, .99]) \
                .setContact(0) \

            demo_id += 1

        C.delFrame(f"target_book_0")
        C.view(False)
        C.setJointState(q0)
        
        C.getFrame(prefix+'panda_finger_joint1').setJointState(np.array([.01]))


C2.view(True)