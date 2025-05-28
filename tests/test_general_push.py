import robotic as ry
import manipulation as manip
import numpy as np
from shelf import generate_shelf
from high_level_methods import RobotEnviroment

C = ry.Config()
C.addFile(ry.raiPath('../rai-robotModels/scenarios/pandaSingle.g'))

C.addFrame('box', 'table') \
    .setJoint(ry.JT.rigid) \
    .setShape(ry.ST.ssBox, [.15,.06,.06,.005]) \
    .setRelativePosition([-.0,.3-.055,.095]) \
    .setContact(1) \
    .setMass(.1)


C.getFrame('l_panda_finger_joint1').setJointState(np.array([.01]))


C.getFrame("box").setRelativePosition([-.0,.3-.055,.095])
C.getFrame("box").setRelativeQuaternion([1.,0,0,0])

for i in range(10):
    roboenv = RobotEnviroment(C)
    success = roboenv.push_frame_to("box", [.0, .35, 0])