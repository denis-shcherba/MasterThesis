import robotic as ry
import numpy as np
from shelf import generate_shelf

if __name__ == "__main__":
    C = ry.Config()
    C.addFile(ry.raiPath('../rai-robotModels/scenarios/pandaFloatingFixGripper.g'))
    pos = np.array([1., 0., .3])
    generate_shelf(C, pos, base_quaternion=[1, 0, 0, 1], openings_small=[4, 11], equidistant=False)

    pcl = np.load("pc.npy")
    print(pcl.shape)

    C.addFrame("pcl", "camera").setPointCloud(pcl)
    #C.getFrame("camera").setPointCloud(pcl)
    C.view(True)
