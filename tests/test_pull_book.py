import numpy as np
import robotic as ry
import h5py
import MasterThesis.manipulation as manip
from MasterThesis.shelf import generate_shelf
from MasterThesis.high_level_methods import RobotEnviroment
from MasterThesis.book_spawning import generate_random_box_params

ROBOT_MODE = "floating" 
COLLECT_DATA = True
PATH_MODE = "JOINTS7D" # or "SE38D" or "SE39D"  
SIMULATE = True
CAMERA = "cameraStatic"  # or "cameraWrist"
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
    #C.addFile("../MasterThesis/configs/pandaFloatingFixGripper.g")

    gripper = "gripper"
    palm = "palm"
    C.setJointState(C.getJointState() + np.array([.3, 0, .2, 0, 0, 0, 0]))
    #C.getFrame('panda_finger_joint1').setJointState(np.array([.01]))
    prefix = ""

C.getFrame(prefix+"finger1").setAttribute("friction", 1e5)
C.getFrame(prefix+"finger2").setAttribute("friction", 1e5)

# ry.params_add({'physx/gripperKp': 0})
# ry.params_add({'physx/gripperKd': 0})

q0 = C.getJointState()
# Shelf
pos = np.array([.8, 0., .3])
generate_shelf(C, pos, base_quaternion=[1, 0, 0, 1], openings_small=[4, 11], equidistant=False)

C.addFrame("cameraWP", CAMERA).setShape(ry.ST.marker, [.1]) 
C.view(True)

color = [1., 0., 0.]

# Frame in use for our book manipulations
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

samples = generate_random_box_params(shelf_size, box_size_ranges, num_samples=10, num_boxes= 1, allow_yaw=False)

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
                .setQuaternion(q.getArr()) \
                .setShape(ry.ST.ssBox, size=[box[0], box[1], box[2], 0.005]) \
                .setColor(np.random.rand(3)) \
                .setContact(1) \
                .setMass(.1)
        C.view(True)

        
        # target at the middle of the shelf ending
        target = np.array([
            (shelfBottomFrame.getPosition()[:2] + np.array([-shelf_depth/2, 0])),
        ])
        target = np.append(target, C.getFrame("target_book_0").getPosition()[2])

        C.addFrame("target").setShape(ry.ST.marker, .1).setPosition(target)


        roboenv = RobotEnviroment(C, sim=SIMULATE, gripper=gripper)

        success = roboenv.pull("target_book_0", target, accumulated_collisions=False)

        if success and COLLECT_DATA:
            np.save("pc.npy", roboenv.points[0])

            demo_group = h5file.create_group(f"demo_{demo_id}")
            demo_group.create_dataset("path", data=roboenv.path)
            demo_group.create_dataset("points", data=roboenv.points)
            demo_id += 1


        C.delFrame(f"target_book_0")
        C.view(True)
        C.setJointState(q0)
        
        C.getFrame(prefix+'panda_finger_joint1').setJointState(np.array([.01]))

if COLLECT_DATA:
    h5file.close()