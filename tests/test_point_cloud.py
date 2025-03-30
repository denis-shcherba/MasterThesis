import numpy as np
import robotic as ry
from MasterThesis.shelf import generate_shelf
from MasterThesis.book_spawning import generate_random_box_params
from MasterThesis.utils import point_in_box_filtering, plot_box





C = ry.Config()
C.addFile(ry.raiPath('../rai-robotModels/scenarios/pandaSingle.g'))

C.getFrame("table").setShape(ry.ST.ssBox, size=[.5, .5, .1, .02])
C.getFrame("l_panda_base").setPosition(C.getFrame("l_panda_base").getPosition() + np.array([0, .25, 0]))

C.setJointState(C.getJointState()+np.array([0, -.3, 0, 0, 0, 0, 1.3]))
# Shelf
pos = np.array([0, 1, .3])
generate_shelf(C, pos, base_quaternion=[0, 0, 0, 1], openings_small=[4, 11], equidistant=False)

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

samples = generate_random_box_params(shelf_size, box_size_ranges, num_samples=1, allow_yaw=True)



for sample in samples:
    print(sample)
    q = ry.Quaternion().setRollPitchYaw(([0,0, sample[-1]]))
    C.addFrame(f"target_book") \
        .setPosition(target + np.append(sample[3:5], (shelf_height+sample[2])/2)) \
        .setQuaternion(q.getArr()) \
        .setShape(ry.ST.ssBox, size=[sample[0], sample[1], sample[2], 0.005]) \
        .setColor(np.random.rand(3)) \
        .setContact(1) \
        .setMass(.1)
    C.view(True)



pcl = C.addFrame('pcl')
bot = ry.BotOp(C, useRealRobot=False)

pcl = C.getFrame("pcl")
pcl.setShape(ry.ST.pointCloud, [2]) # the size here is pixel size for display
bot.sync(C)



while bot.getKeyPressed()!=ord('q'):
    image, depth, points = bot.getImageDepthPcl("cameraWrist", True)
    pcl.setPointCloud(points, image)
    point_cloud_ = points.reshape(-1, 3)
    pcl.setColor([1,0,0])
    bot.sync(C, .1)
     

# last minus accounts for inside box inaccuracy TODO
point_cloud = point_in_box_filtering(point_cloud_, (C.getFrame("big_box_inside_0_2").getPosition(), C.getFrame("big_box_inside_0_2").getSize()[:3]-np.array([.01, .01, .01])))

import open3d as o3d
point_cloud_o3d = o3d.geometry.PointCloud()
point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud)

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud_o3d], window_name="Open3D Point Cloud")
np.save("point_cloud.npy", point_cloud)

