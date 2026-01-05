#TODO test for frame to follow only on submanifold

import robotic as ry

C = ry.Config()
C.addFile("$RAI_PATH/scenarios/pandaSingle.g")

C.addFrame("box").setShape(ry.ST.box, [0.1, 0.1, 0.1]).setPosition([0.5, 0, 0.7]).setColor([1,0,0])

C.getFrame("box").setParent(C.getFrame("l_gripper"))
C.getFrame("box").setJoint(ry.JT.transXYPhi)

C.setJointState([1, -0.5, 0, -2.0, 0, 1.5, 0])
C.view(True)
