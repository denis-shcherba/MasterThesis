import numpy as np
import robotic as ry
from envs.shelf import generate_shelf
from envs.high_level_methods import RobotEnviroment
from envs.book_spawning import generate_random_box_params
from envs.utils import find_nearest_cuboid_edge_center, sample_cuboid_edges, choose_starting_point
import random

ROBOT_MODE = "floating" 
MAX_NUMBER_PUSHES = 5

C = ry.Config()

gripper = "l_gripper"
palm = "l_palm"

if ROBOT_MODE == "normal":
    C.addFile(ry.raiPath('../rai-robotModels/scenarios/pandaSingle.g'))
    C.delFrame("panda_collCameraWrist")
    C.getFrame("table").setShape(ry.ST.ssBox, size=[.5, 1, .1, .005]).setColor(np.array([242, 240, 216])/255)   # Real size [1.1, 1.2, .02, .005]
    C.getFrame('l_panda_finger_joint1').setJointState(np.array([.01]))

elif ROBOT_MODE == "floating":
    C.addFile(ry.raiPath('../rai-robotModels/scenarios/pandaFloatingFixGripper.g'))
    gripper = "gripper"
    palm = "palm"

q0 = C.getJointState()

# Shelf
pos = np.array([1, 0., .3])
generate_shelf(C, pos, base_quaternion=[1, 0, 0, 1], openings_small=[4, 11], equidistant=False)

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
    'z': (.025, .045),   # Z_b range
}

samples = generate_random_box_params(shelf_size, box_size_ranges, num_samples=50, allow_yaw=True)

shelf_corner = np.array([
    (shelfBottomFrame.getPosition()[:3] + np.array([-shelf_depth/2, -shelf_width/2, 0])),
])

for sample in samples:
    for book_params in sample:
        yaw = book_params[-1]
        #yaw = 2*np.pi*i/ len(samples)
        q = ry.Quaternion().setRollPitchYaw(([0,0, yaw])).asArr()


        book_com = shelf_corner + np.append(book_params[3:5], (shelf_height+book_params[2])/2)
        C.addFrame(f"target_book") \
            .setPosition(book_com) \
            .setShape(ry.ST.ssBox, size=[book_params[0], book_params[1], book_params[2], 0.005]) \
            .setColor(np.random.rand(3)) \
            .setContact(1) \
            .setMass(.1) \
            .setQuaternion(q)




    for i in range(MAX_NUMBER_PUSHES):
        nearest_cuboid_edge_center = find_nearest_cuboid_edge_center(C, "target_book", yaw)
        
        points = sample_cuboid_edges(C, "target_book", yaw, samples=10, sides_rel=True, sides_to_sample=[True, True, False, True])

        # filter every point that has no bigger x coord than nearest_cuboid_edge_center
        points = [point for point in points if point[0]>nearest_cuboid_edge_center[0]]

        for j, point in enumerate(points):
            C.addFrame(f"sample{j}").setShape(ry.ST.sphere, size=[.01]).setPosition(point).setContact(0)

        C.addFrame("to_push_point").setShape(ry.ST.marker, size=[.5]).setPosition(nearest_cuboid_edge_center)
    
        roboenv = RobotEnviroment(C, sim=True, gripper=gripper)

        pre_push_paths = []
        for j, point in enumerate(points):
            success, path = roboenv.move_to_point_path(point, minDistance=.1, accumulated_collisions=True)

            if success:
                C.getFrame(f"sample{j}").setColor([0, 1, 0, .9])
                pre_push_paths.append(path)
            else:
                C.getFrame(f"sample{j}").setColor([1, 0, 0, .9])

            C.view(False, "Calculating success score for push proposal")
        C.view(True, "All success samples")

        if len(pre_push_paths) != 0:
            #TODO with dict of path and point
            #starting_point = choose_starting_point(success_pushstart_proposal)
            path = random.choice(pre_push_paths)
            roboenv.run_path(path)
            
            C.view(True)
            roboenv.move_to_point(nearest_cuboid_edge_center, straight_line=True, accumulated_collisions=False)


        C.delFrame("to_push_point")
        C.setJointState(q0)
        C.view(False)
            
        for j in range(len(points)):
            C.delFrame(f"sample{j}")    

        # if isGraspable(C, "target_book"):
        #     break
    
    C.delFrame(f"target_book")
    C.view(False)