import numpy as np
import robotic as ry
import time
import MasterThesis.manipulation as manip
from MasterThesis.shelf import generate_shelf
from MasterThesis.high_level_methods import RobotEnviroment
from MasterThesis.book_spawning import generate_random_box_params

C = ry.Config()
C.addFile(ry.raiPath('../rai-robotModels/scenarios/pandaSingle.g'))


C.delFrame("panda_collCameraWrist")
C.getFrame("table").setShape(ry.ST.ssBox, size=[1., 1., .1, .02])

# Shelf
pos = np.array([1., 0., .3])
generate_shelf(C, pos, base_quaternion=[1, 0, 0, 1], openings_small=[4, 11], equidistant=False)

# for convenience, a few definitions:
gripper = "l_gripper"
palm = "l_palm"

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

samples = generate_random_box_params(shelf_size, box_size_ranges, num_samples=100)

target = np.array([
    (shelfBottomFrame.getPosition()[:3] + np.array([-shelf_depth/2, -shelf_width/2, 0])),
])

C.addFrame(f"lower_shelf_corner") \
    .setPosition(target) \
    .setShape(ry.ST.marker, size=[.2]) \
    
for i in range(0):
    print(samples[i])
    C.addFrame(f"target_book") \
        .setPosition(target + np.asarray(samples[i][3:6])) \
        .setShape(ry.ST.ssBox, size=[samples[i][0], samples[i][1], samples[i][2], 0.005]) \
        .setColor(np.random.rand(3)) \
        .setContact(1) \
        .setMass(.1)
    C.view(True)
    C.delFrame(f"target_book")
    C.view(False)


samples = generate_random_box_params(shelf_size, box_size_ranges, num_samples=100, allow_yaw=True)

def floats(input_string):
    return [float(num) for num in input_string.replace(',', ' ').split()]

    
for i in range(100):
    print(samples[i])
    q = ry.Quaternion().setRollPitchYaw(([0,0, samples[i][-1]]))
    C.addFrame(f"target_book") \
        .setPosition(target + np.append(samples[i][3:5], (shelf_height+samples[i][2])/2)) \
        .setQuaternion(q.getArr()) \
        .setShape(ry.ST.ssBox, size=[samples[i][0], samples[i][1], samples[i][2], 0.005]) \
        .setColor(np.random.rand(3)) \
        .setContact(1) \
        .setMass(.1)
    C.view(True)
    C.delFrame(f"target_book")
    C.view(False)
