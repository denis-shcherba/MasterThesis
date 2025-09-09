import numpy as np
import robotic as ry
import h5py
from envs.shelf import generate_shelf
from envs.high_level_methods import RobotEnviroment
from envs.book_spawning import generate_random_box_params

ROBOT_MODE = "floating" # "normal" or "floating"
COLLECT_DATA = True
PATH_MODE = "POS3D" # "JOINT7DSPLINE", "SE39DSPLINE", "POS3DSPLINE", "DELTA3DSPLINE", "RegressPC2Pos", WAYplusTIMING
SIMULATE = True 
CAMERA = "cameraStatic"  # or "cameraWrist"
BASE_REMOVAl = False # if true, shelf will be removed from observation
DEBUG = False # pull debugging
OBSERVATION_MODE = "DEPTH" # "POINTCLOUD", "RGB", "DEPTH"
COMPRESS = True
RANDOM_COLOR = False
NUM_SAMPLES = 5000
VISUALIZE = False  # If true, the simulation will be visualized

noise_dict = {
    # "stateNoise": {
    #     "type": "singleGaussian",
    #     "prob": 0.1,
    #     "std": 0.0025,
    #     "mean": 0.0,
    # },
    # "depthNoise": {}
}

# State noise variants:  gaussianIntegratedOverPath, randomWaypoint, singleGaussian

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

# not necessary as it seems
# C.getFrame(prefix+"finger1").setAttribute("friction", 1)
# C.getFrame(prefix+"finger2").setAttribute("friction", 1)

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
                .setColor(np.random.rand(3) if RANDOM_COLOR else [1, 0, 0]) \
                .setContact(1) \
                .setMass(.1) \
                .setAttribute("friction", .01) 
        C.view(False)

        
        # target at the middle of the shelf ending
        target = np.array([
            (shelfBottomFrame.getPosition()[:2] + np.array([-shelf_depth/2, 0])),
        ])
        target = np.append(target, C.getFrame("target_book_0").getPosition()[2])

        C.addFrame("target").setShape(ry.ST.marker, .1).setPosition(target)


        roboenv = RobotEnviroment(C, sim=SIMULATE, gripper=gripper, base_removal=BASE_REMOVAl, observation_mode=OBSERVATION_MODE, visualize=VISUALIZE, path_mode=PATH_MODE, noise_dict=noise_dict)

        success = roboenv.pull("target_book_0", target, accumulated_collisions=False, get_observation=COLLECT_DATA)
        

        if success and COLLECT_DATA:
            #np.save("pc.npy", roboenv.points[0])

            demo_group = h5file.create_group(f"demo_{demo_id}")

            if PATH_MODE == "JOINT7DSPLINE":
                demo_group.create_dataset("path", data=roboenv.path)
            if PATH_MODE == "SE39DSPLINE":
                se3_path = np.zeros((roboenv.path.shape[0], 9))
                
                for i in range(roboenv.path.shape[0]):
                    q = ry.Quaternion().set(roboenv.path[i][3:])
                    R = q.getMatrix()
                    # Combine position (3D) with first two rotation matrix columns (6D)
                    se3_path[i, :3] = roboenv.path[i][:3]  # Position
                    se3_path[i, 3:9] = np.array([R[0:3, 0], R[0:3, 1]]).flatten()  # Rotation
                
                # Now use se3_path instead of the original path
                demo_group.create_dataset("path", data=se3_path)

            elif PATH_MODE == "POS3DSPLINE":
                # save the spline path 3D control points
                demo_group.create_dataset("path", data=roboenv.path[:, :3])  # Only position

            elif PATH_MODE == "POS3D":
                # save the spline path 3D control points
                demo_group.create_dataset("path", data=roboenv.path[:, :3])  # Only position


            elif PATH_MODE == "DELTA3DSPLINE":
                delta_paths = np.empty((64, 3))
                delta_paths[0] = roboenv.path[0][:3]-C.getJointState()[:3]
                for i in range(1, roboenv.path.shape[0]):
                    delta_paths[i] = roboenv.path[i][:3] - roboenv.path[i-1][:3]
                demo_group.create_dataset("path", data=delta_paths)  # Only delta positions

            elif PATH_MODE == "RegressPC2Pos":
                print(roboenv.path.shape)
                demo_group.create_dataset("path", data=roboenv.path[31, :3])  # Only position

            elif PATH_MODE == "WAYplusTIMING":
                # Save the waypoints and timing\
                demo_group.create_dataset("path", data=roboenv.path[:, :3])  
                demo_group.create_dataset("ways", data=roboenv.ways) 
                demo_group.create_dataset("timings", data=roboenv.timings) 

            if OBSERVATION_MODE == "POINTCLOUD":
                demo_group.create_dataset("points", data=roboenv.points)
            elif OBSERVATION_MODE == "RGB":
                if COMPRESS:
                        demo_group.create_dataset(
                        "rgb", 
                        data=roboenv.rgb_image,
                        compression="gzip",
                        compression_opts=4
                        )
                else:
                    demo_group.create_dataset("rgb", data=roboenv.rgb_image)
            
            elif OBSERVATION_MODE == "DEPTH":
                if COMPRESS:
                        demo_group.create_dataset(
                        "depth", 
                        data=roboenv.depth_image,
                        compression="gzip",
                        compression_opts=4
                        )
                else:
                    demo_group.create_dataset("depth", data=roboenv.depth_image)

            
            demo_id += 1

        elif success and DEBUG:
            err.append(np.linalg.norm(C.getFrame("target_book_0").getPosition() - target))

        C.delFrame(f"target_book_0")
        C.view(False)
        C.setJointState(q0)
        
        C.getFrame(prefix+'panda_finger_joint1').setJointState(np.array([.01]))


if DEBUG:
    print("Average error to target:", np.mean(err))
    print("Max error to target:", np.max(err))
    print("Min error to target:", np.min(err))

if COLLECT_DATA:
    h5file.close()