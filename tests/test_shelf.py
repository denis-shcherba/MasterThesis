import numpy as np
import robotic as ry
import time
import MasterThesis.manipulation as manip
from MasterThesis.shelf import generate_shelf
import MasterThesis.manipulation as manip
from MasterThesis.high_level_methods import RobotEnviroment

C = ry.Config()
C.addFile(ry.raiPath('../rai-robotModels/scenarios/pandaSingle.g'))

C.delFrame("panda_collCameraWrist")
C.getFrame("table").setShape(ry.ST.ssBox, size=[1., 1., .1, .02])

# Shelf
pos = np.array([-1., 0., .5])
generate_shelf(C, pos, base_quaternion=[1, 0, 0, 1])

# for convenience, a few definitions:
gripper = "l_gripper"
palm = "l_palm"

color = [1., 0., 0.]



C.addFrame(f"target_book") \
    .setPosition([.0, .15, .7]) \
    .setShape(ry.ST.ssBox, size=[.12, .18, .04, 0.01]) \
    .setColor(color) \
    .setContact(1) \
    .setMass(.1)


R = RobotEnviroment(C)
R.push("target_book", .1, 0)

C.view(True)
