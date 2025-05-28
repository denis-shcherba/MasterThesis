from cube_fitting import *
import robotic as ry
from scipy.spatial.transform import Rotation

if __name__ == "__main__":
    pcl = np.load("point_cloud.npy")
    q=ry.Quaternion().setMatrix([1,0,0,0,1,0,0,0,1])

    C = ry.Config()
    C.addFrame("pcl").setPointCloud(pcl)
    C.view(True)

    # Fit Axis-Aligned Bounding Box
    min_point, max_point, center, dimensions = fit_aabb(pcl)
    print("Axis-Aligned Bounding Box:")
    print(f"Min Point: {min_point}")
    print(f"Max Point: {max_point}")
    print(f"Center: {center}")
    print(f"Dimensions: {dimensions}")
    C.addFrame("fitted_box").setShape(ry.ST.box, size=dimensions).setPosition(center).setColor([.7,.7,.7,.5])
    C.view(True, "axis aligned bounding box")
    C.delFrame("fitted_box")
    C.view(False)


    # Oriented Bounding Box using PCA
    oriented_corners, axes, variances = principal_component_bounding_box(pcl)
    print("\nOriented Bounding Box using PCA:")
    print("Principal Axes:")
    print(axes)
    print("Variances:", variances)
    for i, corner in enumerate(oriented_corners):
        C.addFrame(f"corner{i}").setShape(ry.ST.sphere, size=[.005]).setColor([.2,.3,.8]).setPosition(corner)
    C.view(True, "Oriented Bounding Box using PCA")
    for i in range(len(oriented_corners)):
        C.delFrame(f"corner{i}")
    C.view(False)


    # Oriented Bounding Box using Convex Hull
    best_box, best_rotation = minimum_bounding_box_from_convex_hull(pcl)
    com, sizes = cuboid_corners_to_size_com(best_box)           #  corner points to cuboid center and sizes in x, y, z
    q=ry.Quaternion().setMatrix(best_rotation).getArr()         #  Rotation Matrix to  quaternion

    print(com)
    print("DIMS;", sizes)
    C.addFrame("test_box").setShape(ry.ST.box, size=sizes).setPosition(com).setQuaternion(q).setColor([1,1,1,.4])
    C.view(True, "Oriented Bounding Box using Convex Hull")



    