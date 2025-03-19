import numpy as np
import robotic as ry
import time
import MasterThesis.manipulation as manip
from MasterThesis.shelf import generate_shelf
from MasterThesis.high_level_methods import RobotEnviroment
from MasterThesis.book_spawning import generate_random_box_positions

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

# Example usage
shelf_size = (shelfBottomFrame.getSize()[0], shelfBottomFrame.getSize()[1], shelfBottomFrame.getSize()[2])  # Fixed shelf dimensions (X_s, Y_s, Z_s)

box_size_ranges = {  # Variable box dimensions
    'x': (.1, .15),  # X_b range
    'y': (.14, .23),  # Y_b range
    'z': (.007, .04),   # Z_b range
}

samples = generate_random_box_positions(shelf_size, box_size_ranges, num_samples=100)

for i in range(10):
    print(samples[i])
    C.addFrame(f"target_book") \
        .setPosition([.64, .02, 1.18]) \
        .setShape(ry.ST.ssBox, size=[samples[i][0], samples[i][1], samples[i][2], 0.01]) \
        .setColor(np.random.rand(3)) \
        .setContact(1) \
        .setMass(.1)
    C.view(True)
    C.delFrame(f"target_book")
    C.view(False)

C.addFrame(f"target_book") \
    .setPosition([.64, .02, 1.18]) \
    .setShape(ry.ST.ssBox, size=[samples[-1][0], samples[-1][1], samples[-1][2], 0.01]) \
    .setColor(np.random.rand(3)) \
    .setContact(1) \
    .setMass(.1)
    

target = np.array([
    (shelfBottomFrame.getPosition()[:2] + np.array([-shelf_depth/2, 0])),
])
target = np.append(target, C.getFrame("target_book").getPosition()[2])

C.view(True)


roboenv = RobotEnviroment(C)
success = roboenv.pull("target_book", target)
