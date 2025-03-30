import numpy as np
import robotic as ry
import time
import MasterThesis.manipulation as manip
from MasterThesis.shelf import generate_shelf
from MasterThesis.high_level_methods import RobotEnviroment
from MasterThesis.book_spawning import generate_random_box_params
import cmath


C = ry.Config()
C.addFile(ry.raiPath('../rai-robotModels/scenarios/pandaSingle.g'))

C.delFrame("panda_collCameraWrist")
C.getFrame("table").setShape(ry.ST.ssBox, size=[.5, 1, .1, .005]).setColor(np.array([242, 240, 216])/255)   # Real size [1.1, 1.2, .02, .005]

# Shelf
pos = np.array([.8, 0., .3])
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

samples = generate_random_box_params(shelf_size, box_size_ranges, num_samples=50, allow_yaw=True)

shelf_corner = np.array([
    (shelfBottomFrame.getPosition()[:3] + np.array([-shelf_depth/2, -shelf_width/2, 0])),
])

for book_params in samples:
    q = ry.Quaternion().setRollPitchYaw(([0,0, book_params[-1]])).getArr()

    book_com = shelf_corner + np.append(book_params[3:5], (shelf_height+book_params[2])/2)
    C.addFrame(f"target_book") \
        .setPosition(book_com) \
        .setShape(ry.ST.ssBox, size=[book_params[0], book_params[1], book_params[2], 0.005]) \
        .setColor(np.random.rand(3)) \
        .setContact(1) \
        .setMass(.1) \
        .setQuaternion(q)
    C.view(True)


    delta = complex(-book_params[0]/2, 0)*cmath.exp(1j * book_params[-1]) # dir to push push dir waypoint from com
    C.addFrame("to_push_point").setShape(ry.ST.marker, size=[.5]).setPosition(book_com+np.array([delta.real, delta.imag, 0]))
    # target at the middle of the shelf ending
    target = np.array([
        (shelfBottomFrame.getPosition()[:2] + np.array([-shelf_depth/2, 0])),
    ])
    target = np.append(target, C.getFrame("target_book").getPosition()[2])

    C.addFrame("target").setShape(ry.ST.marker, [.1]).setPosition(target)
    C.view(True)


    roboenv = RobotEnviroment(C)
    success = roboenv.pull("target_book", target)

    C.delFrame(f"target_book")
    C.delFrame("to_push_point")
    C.view(False)