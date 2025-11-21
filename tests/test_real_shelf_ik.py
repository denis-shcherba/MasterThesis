import numpy as np
import robotic as ry
from envs.shelf import generate_shelf
from envs.book_spawning import generate_random_box_params

BOTOP = True

C = ry.Config() 
C.addFile("$RAI_PATH/scenarios/pandaSingle.g")
C.getFrame("table").setShape(C.getFrame("table").getShapeType(), [1.2, 1.1, .1, .01]).setColor(np.array([242, 240, 216]) / 255)
C.getFrame("l_panda_base").setPosition(C.getFrame("l_panda_base").getPosition() + np.array([0, -.08, .0]))
C.addFrame("wall_behind_panda").setPosition([0, -.665, 1.4]).setShape(ry.ST.box, [2.5, 0.03, 1.5]).setContact(1)

pos = np.array([.98, 0.05, .18])
generate_shelf(C, pos, base_quaternion=[1, 0, 0, 1], openings_small=[4, 11], equidistant=False)

shelfBottomFrame = C.getFrame("big_xy_bottom_0_1")
shelf_depth = shelfBottomFrame.getSize()[1]
shelf_width = shelfBottomFrame.getSize()[0]
shelf_height = shelfBottomFrame.getSize()[2]
shelf_room_height = .3 # floor_entries[2] in shelf.py

# Example usage
shelf_size = (shelfBottomFrame.getSize()[0], shelfBottomFrame.getSize()[1], shelfBottomFrame.getSize()[2])  # Fixed shelf dimensions (X_s, Y_s, Z_s)

target = np.array([
    (shelfBottomFrame.getPosition()[:3]),
])

extra_floor_height = 0.015
C.addFrame("shelf_extra_floor").setPosition(target + np.array([0,0,shelf_height/2+extra_floor_height/2])).setShape(ry.ST.box, [shelf_depth, shelf_width, extra_floor_height]).setColor([1,1,1]).setContact(1)

#C.addFrame("red_cube").setPosition(C.getFrame("shelf_extra_floor").getPosition() + np.array([0, 0, extra_floor_height/2+.06/2]) + np.array([-.15, 0 ,0])).setShape(ry.ST.ssBox, [.06, .06, .06, .001]).setColor([1,0,0]).setContact(1)
C.addFrame("book_aof").setPosition(C.getFrame("shelf_extra_floor").getPosition() + np.array([0, 0, extra_floor_height/2+.021/2]) + np.array([-.1, 0 ,0])).setShape(ry.ST.ssBox, [.197, .128, .021, .001]).setColor([1,0,0]).setContact(1)


#C.addFrame("target").setShape(ry.ST.marker, [.2]).setPosition([.76, .08, 1.1])
C.addFrame("target", "book_aof").setShape(ry.ST.marker, [.2]).setRelativePosition([-.03, 0, .01])

C.view(True)

four_corners = [ 
    np.array([(shelfBottomFrame.getPosition()[:3] + np.array([-shelf_depth/2, -shelf_width/2, 0]))]),
    np.array([(shelfBottomFrame.getPosition()[:3] + np.array([-shelf_depth/2, shelf_width/2, 0]))]),
    np.array([(shelfBottomFrame.getPosition()[:3] + np.array([-shelf_depth/2, -shelf_width/2, shelf_room_height]))]),
    np.array([(shelfBottomFrame.getPosition()[:3] + np.array([-shelf_depth/2, shelf_width/2, shelf_room_height]))]),
]

for i, corner in enumerate(four_corners):
    C.addFrame(f"shelf_corner_{i}").setPosition(corner).setShape(ry.ST.marker, size=[.2])

q0 = C.getJointState()
M = ry.KOMO_ManipulationHelper()
#M.setup_motion(self.C, K=2, steps_per_phase=1, homing_scale=.1, acceleration_scale=1, accumulated_collisions=False, joint_limits=True, quaternion_norms=False)
M.setup_inverse_kinematics(C, accumulated_collisions=False, quaternion_norms=False)
M.komo.addObjective([], ry.FS.positionDiff, ["l_gripper", "target"], ry.OT.eq, [], 0.0)

M.solve()
q1 = M.komo.getPath()[0]

M.komo.view(True)

#(C, q0, q1) = M.komo.getSubProblem(0)

# rrt = ry.RRT_PathFinder()
# rrt.setProblem(C)
# rrt.setOptions(verbose=1, stepsize=.1, subsamples=4, maxIters=5000, p_connect=.5, collisionTolerance=.0001, useBroadCollisions=True)
# rrt.setStartGoal([q0], [q1])

# ret = rrt.solve()
# print(ret.x.shape)
# rrt.view(True)

# for i in ret.x:
#     C.setJointState(i)
#     C.checkConsistency()
#     C.view(True)


C.setJointState(q0)
if BOTOP:
    bot = ry.BotOp(C, True)
    bot.home(C)
    # while bot.getKeyPressed() != 'q':
    #     bot.hold(True, True)
    #     bot.sync(C)

    bot.moveTo(M.path[0])
    while(bot.getTimeToEnd() > 0):
        bot.wait(C)
    C.view(True)
    
    # bot.moveAutoTimed(ret.x)
    # while(bot.getTimeToEnd() > 0):
    #     bot.wait(C)
    # C.view(True)
    