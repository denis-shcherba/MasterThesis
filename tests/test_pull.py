import robotic as ry
import MasterThesis.manipulation as manip
import numpy as np
from MasterThesis.shelf import generate_shelf
import time

C = ry.Config()
C.addFile(ry.raiPath('../rai-robotModels/scenarios/pandaSingle.g'))

midpoint = np.array([-0.105, 0.4, 0.705-.025])
C.addFrame("box") \
    .setPosition(midpoint) \
    .setShape(ry.ST.ssBox, size=[0.04, 0.12, 0.04, 0.001]) \
    .setColor([0, 0, 1]) \
    .setContact(1) \
    .setMass(.1)

#C.delFrame("panda_collCameraWrist")

# for convenience, a few definitions:
gripper = "l_gripper"
palm = "l_palm"
box = "box"
table = "table"

C.view()

def pull_orthogonal(object_, placePosition) -> bool:
    M = manip.ManipulationModelling()
    M.setup_pick_and_place_waypoints(C, gripper, object_, 1e-1)
    M.pull([1.,2.], object_, gripper, table)
    M.komo.addObjective([2.], ry.FS.position, [object_], ry.OT.eq, 1e1*np.array([[1,0,0],[0,1,0]]), placePosition)
    M.solve()
    if not M.feasible:
        return False

    M1 = M.sub_motion(0, accumulated_collisions=False)
    M1.retractPush([.0, .15], gripper, .03)
    M1.approachPush([.85, 1.], gripper, .03)
    path1 = M1.solve()
    if not M1.feasible:
        return False

    M2 = M.sub_motion(1, accumulated_collisions=False)
    path2 = M2.solve()
    if not M2.feasible:
         return False

    M1.play(C, 1.)
    C.attach(gripper, object_)
    M2.play(C, 1.)
    C.attach(table, object_)

    return True


attempt_count = 0
data = []
for l in range(attempt_count):
    
    action = "pull"
    if action == "pull":
        object_ = np.random.choice([box])
        success = pull_orthogonal(object_, placePosition = midpoint + np.random.uniform(-.1, .1, size=3)
)

    else:
        raise Exception(f'Action "{action}" is not defined!')
    
    data.append({"action": action, "success": success})
print(data)


del C

C = ry.Config()
C.addFile(ry.raiPath('../rai-robotModels/scenarios/pandaSingle.g'))

C.delFrame("panda_collCameraWrist")
C.getFrame("table").setShape(ry.ST.ssBox, size=[1., 1., .1, .02])

# Shelf
pos = np.array([1., 0., .3])
generate_shelf(C, pos, base_quaternion=[1, 0, 0, 1], openings_small=[4, 11], equidistant=False)

# for convenience, a few definitions:
gripper = "l_gripper"
palm = "l_palm"

color = [1., 0., 0.]

C.addFrame(f"target_book") \
    .setPosition([.64, .02, 1.18]) \
    .setShape(ry.ST.ssBox, size=[.12, .18, .04, 0.01]) \
    .setColor(color) \
    .setContact(1) \
    .setMass(.1)

C.addFrame(f"push_waypoint") \
    .setPosition([.64, .02, 1.25]) \
    .setShape(ry.ST.marker, size=[.1]) \

q0 = C.getJointState()
C.view(True)
success = pull_orthogonal("target_book", C.getFrame("target_book").getPosition() + np.array([-.1, 0, 0]))
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


for i, value in enumerate(path):
    C.setJointState(value)

    C.addFrame(f'way{i}'). setShape(ry.ST.marker, [.1]) .setPosition(C.getFrame("l_gripper").getPosition()) .setQuaternion(C.getFrame("l_gripper").getQuaternion())

C.setJointState(q0)
C.view(True)


#success = pull_orthogonal("target_book", C.getFrame("target_book").getPosition() + np.array([-.05, 0, 0]))
