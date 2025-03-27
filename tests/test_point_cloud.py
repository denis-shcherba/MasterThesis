import numpy as np
import robotic as ry
from MasterThesis.shelf import generate_shelf
from MasterThesis.book_spawning import generate_random_box_params
from MasterThesis.utils import point_in_box_filtering, plot_box
from sklearn.decomposition import PCA
from sklearn.linear_model import RANSACRegressor
import open3d as o3d

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

point_cloud_o3d = o3d.geometry.PointCloud()
point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud)

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud_o3d], window_name="Open3D Point Cloud")


def fit_plane_ransac(points, threshold=0.01, max_trials=1000):
    """Fits a plane to a point cloud using RANSAC."""
    X, Y, Z = points.T
    A = np.c_[X, Y, np.ones_like(X)]  # Plane equation: Ax + By + C = Z
    model = RANSACRegressor(residual_threshold=threshold, max_trials=max_trials)
    model.fit(A, Z)
    return model.estimator_.coef_, model.inlier_mask_

def oriented_bounding_box(points):
    """Computes the oriented bounding box using PCA."""
    pca = PCA(n_components=3)
    pca.fit(points)
    rotation = pca.components_  # Principal axes
    centered_points = points - np.mean(points, axis=0)
    rotated_points = centered_points @ rotation.T
    min_bounds = rotated_points.min(axis=0)
    max_bounds = rotated_points.max(axis=0)
    box_corners = np.array([
        [min_bounds[0], min_bounds[1], min_bounds[2]],
        [min_bounds[0], min_bounds[1], max_bounds[2]],
        [min_bounds[0], max_bounds[1], min_bounds[2]],
        [min_bounds[0], max_bounds[1], max_bounds[2]],
        [max_bounds[0], min_bounds[1], min_bounds[2]],
        [max_bounds[0], min_bounds[1], max_bounds[2]],
        [max_bounds[0], max_bounds[1], min_bounds[2]],
        [max_bounds[0], max_bounds[1], max_bounds[2]],
    ])
    box_corners = box_corners @ rotation + np.mean(points, axis=0)
    return box_corners, rotation

# Example usage:
N = 1000
points = point_cloud.copy()

# Fit a plane
plane_params, inliers = fit_plane_ransac(points)

# Compute the OBB
obb_corners, obb_rotation = oriented_bounding_box(points[inliers])

# Visualize in Open3D
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

obb = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(obb_corners))
obb.color = (1, 0, 0)  # Red

o3d.visualization.draw_geometries([pcd, obb])


import numpy as np
import open3d as o3d
from pyransac3d import Cuboid


# Fit a cuboid using RANSAC
cuboid_model = Cuboid()
best_cuboid, z = cuboid_model.fit(point_cloud, .05)

# Debug output
print("Best cuboid output:", best_cuboid)

# Ensure the output is valid
if best_cuboid is None or len(best_cuboid) < 3:
    raise ValueError("RANSAC failed to detect a valid cuboid.")

# Extract plane equations (ax + by + cz + d = 0)
plane1, plane2, plane3 = best_cuboid

# Function to compute the intersection of three planes
def plane_intersection(p1, p2, p3):
    A = np.array([p1[:3], p2[:3], p3[:3]])  # Coefficients of x, y, z
    b = -np.array([p1[3], p2[3], p3[3]])    # Constant terms (right-hand side)
    
    if np.linalg.det(A) == 0:  # Check if planes are parallel (singular matrix)
        raise ValueError("Planes do not form a valid cuboid (det(A) = 0).")
    
    return np.linalg.solve(A, b)  # Solve Ax = b

# Compute all 8 cuboid corners from plane intersections
planes = [plane1, plane2, plane3]
vertices = []
for i in range(8):
    sign = [(-1)**(i >> j & 1) for j in range(3)]  # Generate +1/-1 combinations
    modified_planes = [
        [planes[j][0], planes[j][1], planes[j][2], planes[j][3] * sign[j]] 
        for j in range(3)
    ]
    vertices.append(plane_intersection(*modified_planes))

vertices = np.array(vertices)

# Ensure correct shape
if vertices.shape != (8, 3):
    raise ValueError(f"Unexpected shape for vertices: {vertices.shape}. Expected (8, 3)")

print("Cuboid vertices:\n", vertices)

# Create Open3D point cloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)

# Define cuboid edges
lines = [
    [0, 1], [1, 3], [3, 2], [2, 0],  # Bottom face
    [4, 5], [5, 7], [7, 6], [6, 4],  # Top face
    [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
]

# Create Open3D line set for cuboid visualization
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(vertices)
line_set.lines = o3d.utility.Vector2iVector(lines)

# Display the point cloud and cuboid
o3d.visualization.draw_geometries([pcd, line_set])








def visualize_planes_and_point_cloud(point_cloud, plane_equations):
    # Convert point cloud to Open3D format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    
    # Create plane geometries
    plane_geometries = []
    for plane_eq in plane_equations:
        # Create a mesh plane
        [a, b, c, d] = plane_eq
        plane = o3d.geometry.TriangleMesh.create_plane(width=10, height=10)
        
        # Adjust plane position and orientation based on equation ax + by + cz + d = 0
        rotation = plane.get_rotation_matrix_from_xyz((0, 0, 0))
        plane.rotate(rotation, center=(0, 0, 0))
        
        # Translate plane to match equation
        center = -plane_eq[3] / np.sqrt(a**2 + b**2 + c**2) * np.array([a, b, c])
        plane.translate(center)
        
        # Color the plane
        plane.paint_uniform_color(np.random.rand(3))
        plane.compute_vertex_normals()
        
        plane_geometries.append(plane)
    
    # Visualize point cloud and planes
    o3d.visualization.draw_geometries([pcd] + plane_geometries)

# Example usage (replace with your actual data)
visualize_planes_and_point_cloud(point_cloud, best_cuboid)