import numpy as np
import robotic as ry
import time
import MasterThesis.manipulation as manip
from MasterThesis.shelf import generate_shelf
from MasterThesis.high_level_methods import RobotEnviroment
from MasterThesis.book_spawning import generate_random_box_params
from MasterThesis.utils import find_nearest_cuboid_edge_center, sample_cuboid_edges

C = ry.Config()
C.addFile(ry.raiPath('../rai-robotModels/scenarios/pandaSingle.g'))

C.delFrame("panda_collCameraWrist")
C.getFrame("table").setShape(ry.ST.ssBox, size=[.5, 1, .1, .005]).setColor(np.array([242, 240, 216])/255)   # Real size [1.1, 1.2, .02, .005]
C.getFrame('l_panda_finger_joint1').setJointState(np.array([.01]))

# Shelf
pos = np.array([1, 0., .3])
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

for i, book_params in enumerate(samples):
    yaw = book_params[-1]
    #yaw = 2*np.pi*i/ len(samples)
    q = ry.Quaternion().setRollPitchYaw(([0,0, yaw])).getArr()


    book_com = shelf_corner + np.append(book_params[3:5], (shelf_height+book_params[2])/2)
    C.addFrame(f"target_book") \
        .setPosition(book_com) \
        .setShape(ry.ST.ssBox, size=[book_params[0], book_params[1], book_params[2], 0.005]) \
        .setColor(np.random.rand(3)) \
        .setContact(1) \
        .setMass(.1) \
        .setQuaternion(q)

    C.view(True)

    nearest_cuboid_edge_center = find_nearest_cuboid_edge_center(C, "target_book", yaw)
    
    points = sample_cuboid_edges(C, "target_book", yaw, samples=30, sides_rel=True, sides_to_sample=[True, True, False, True])

    # filter every point that has no bigger x coord than nearest_cuboid_edge_center
    points = [point for point in points if point[0]>nearest_cuboid_edge_center[0]]

    for j, point in enumerate(points):
        C.addFrame(f"sample{j}").setShape(ry.ST.sphere, size=[.01]).setPosition(point)

    C.addFrame("to_push_point").setShape(ry.ST.marker, size=[.5]).setPosition(nearest_cuboid_edge_center)

    C.view(True)

    roboenv = RobotEnviroment(C)

    for j, point in enumerate(points):
        success, path = roboenv.move_to_point_path(point)

        if success:
            C.getFrame(f"sample{j}").setColor([0, 1, 0])
        else:
            C.getFrame(f"sample{j}").setColor([1, 0, 0])

        C.view(False, "Calculating success score for push proposal")
    C.view(True, "All success samples")

    C.delFrame(f"target_book")
    C.delFrame("to_push_point")
    for j in range(len(points)):
        C.delFrame(f"sample{j}")


    C.view(False)