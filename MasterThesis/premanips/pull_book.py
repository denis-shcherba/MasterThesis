import numpy as np
import robotic as ry
import time
import MasterThesis.manipulation as manip
from MasterThesis.shelf import generate_shelf


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
    .setPosition([-.64, .02, 1.23]) \
    .setShape(ry.ST.ssBox, size=[.12, .18, .04, 0.01]) \
    .setColor(color) \
    .setContact(1) \
    .setMass(.1)

C.addFrame(f"push_waypoint") \
    .setPosition([-.76, .02, 1.25]) \
    .setShape(ry.ST.marker, size=[.1]) \

q0 = C.getJointState()
C.view(True)

# compute a goal configuration
komo = ry.KOMO()
komo.setConfig(C, True)
komo.setTiming(1., 1, 5., 0)
komo.addControlObjective([], 0, 1e-0)
komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)
komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)
komo.addObjective([], ry.FS.positionDiff, [gripper, 'push_waypoint'], ry.OT.eq, [1e1])

ret = ry.NLP_Solver() \
    .setProblem(komo.nlp()) \
    .setOptions( stopTolerance=1e-2, verbose=4 ) \
    .solve()
print(ret)

# that's the goal configuration
qT = komo.getPath()[0]
C.setJointState(qT)
C.view(True, "IK solution")

#define a path finding problem
rrt = ry.RRT_PathFinder()
rrt.setProblem(C)
rrt.setStartGoal([q0], [qT])

ret = rrt.solve()
print(ret)
path = ret.x

# display the path
for t in range(0, path.shape[0]-1):
    C.setJointState(path[t])
    C.view()
    time.sleep(.1)