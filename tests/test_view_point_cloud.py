import robotic as ry
import numpy as np

if __name__ == "__main__":
    pcl = np.load("point_cloud.npy")
    print(pcl.shape)
    
    pcl = np.load("pc.npy")
    print(pcl.shape)

    C = ry.Config()
    C.addFrame("pcl").setPointCloud(pcl)
    C.view(True)
