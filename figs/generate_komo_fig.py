import robotic as ry
import numpy as np
import time

C = ry.Config()
C.addFile(ry.raiPath('scenarios/pandaSingle.g'))
C.view()



C = ry.Config()
C.addFile(ry.raiPath('scenarios/pandaSingle.g'))
C.addFrame('way1'). setShape(ry.ST.marker, [.1]) .setPosition([.4, .2, 1.])
C.addFrame('way2'). setShape(ry.ST.marker, [.1]) .setPosition([.4, .2, 1.4])
C.addFrame('way3'). setShape(ry.ST.marker, [.1]) .setPosition([-.4, .2, 1.])
C.addFrame('way4'). setShape(ry.ST.marker, [.1]) .setPosition([-.4, .2, 1.4])
qHome = C.getJointState()
C.view()

# only differences: the kOrder=2, control objective order 2, constrain final jointState velocity to zero
C.setJointState(qHome)
komo = ry.KOMO(C, 4, 10, 2, False)
komo.addControlObjective([], 0, 1e-1) # what happens if you change weighting to 1e0? why?
komo.addControlObjective([], 2, 1e0)
komo.addObjective([1], ry.FS.positionDiff, ['l_gripper', 'way1'], ry.OT.eq, [1e1])
komo.addObjective([2], ry.FS.positionDiff, ['l_gripper', 'way2'], ry.OT.eq, [1e1])
komo.addObjective([3], ry.FS.positionDiff, ['l_gripper', 'way3'], ry.OT.eq, [1e1])
komo.addObjective([4], ry.FS.positionDiff, ['l_gripper', 'way4'], ry.OT.eq, [1e1])
komo.addObjective([4], ry.FS.jointState, [], ry.OT.eq, [1e1], [], order=1)

ret = ry.NLP_Solver(komo.nlp(), verbose=0 ) .solve()
print(ret)
q = komo.getPath()
komo.view(True)
# print('size of path:', q.shape)

# for t in range(q.shape[0]):
#     C.setJointState(q[t])
#     C.view(False, f'waypoint {t}')
#     time.sleep(.1)


### --- Push --- ###
# change steps_per_phase in setup_point_to_point_motion to 32 or so

for i in range(4):
    C.delFrame(f'way{i+1}')
    
C.view(True)

C.addFrame("target").setShape(ry.ST.marker, [.1]).setPosition([.3, .3, .7])

def straight_push(M, time_interval, obj, gripper, table):
        """ Alternative to Manip.py straight_push with less objectives for more general use cases"""

        #start & end helper frames
        helperStart = f'_straight_pushStart_{gripper}_{obj}_{time_interval[0]}'
        #helperEnd = f'_straight_pushEnd_{gripper}_{obj}_{time_interval[1]}'
        if not M.komo.getConfig().getFrame(helperStart, False):
            # self.add_stable_frame(ry.JT.hingeZ, table, helperStart, obj, .3)
            helper_frame = M.komo.addFrameDof(helperStart, obj, ry.JT.hingeZ, True)
            # helper_frame.setAutoLimits()
            # helper_frame.joint.sampleUniform=1.

        #x-axis of A aligns with diff-pos of B AT END TIMEnot  (always backward diff)
        M.komo.addObjective([time_interval[1]], ry.FS.AlignYWithDiff, [helperStart, obj], ry.OT.eq, [1e0], [], 1)

        #gripper touch
        M.komo.addObjective([time_interval[0]], ry.FS.negDistance, [gripper, obj], ry.OT.eq, [1e1], [-.025])
        #gripper start position
        M.komo.addObjective([time_interval[0]], ry.FS.positionRel, [gripper, helperStart], ry.OT.eq, 1e1*np.array([[1., 0., 0.], [0., 0., 1.]]))
        M.komo.addObjective([time_interval[0]], ry.FS.positionRel, [gripper, helperStart], ry.OT.ineq, 1e1*np.array([[0., 1., 0.]]), [.0, -.02, .0])
        #gripper start orientation
        M.komo.addObjective([time_interval[0]], ry.FS.scalarProductYY, [gripper, helperStart], ry.OT.ineq, [-1e0], [.2])
        M.komo.addObjective([time_interval[0]], ry.FS.scalarProductYZ, [gripper, helperStart], ry.OT.ineq, [-1e0], [.2])
        M.komo.addObjective([time_interval[0]], ry.FS.vectorXDiff, [gripper, helperStart], ry.OT.eq, [1e0])
        M.freeze_relativePose([time_interval[1]], gripper, obj)
    
        return helperStart


table = "table"


C.addFrame('box') \
    .setPosition([-.25,.3,.7]) \
    .setShape(ry.ST.ssBox, size=[.06,.06,.06,.005]) \
    .setColor([1,.5,0]) \
    .setContact(1)
C.view()

object_ = "box"


info = f'push 1'
print('===', info)

M = ry.KOMO_ManipulationHelper(info)
# M.setup_pick_and_place_waypoints(self.C, l_gripper, object_, 1e-1, accumulated_collisions=True)

M.setup_sequence(C, 2, 1e-1, accumulated_collisions=False)
M.komo.addFrameDof('obj_trans', table, ry.JT.transXY, False, object_) #a permanent moving(!) transXY joint table->trans, and a snap trans->obj
M.komo.addRigidSwitch(1., ['obj_trans', object_])
pushStart = straight_push(M, [1.,2.], object_, "l_gripper", table)

M.komo.addObjective([2.], ry.FS.position, [object_], ry.OT.eq, 1e1*np.array([[1,0,0],[0,1,0]]), C.getFrame("target").getPosition()) 
M.solve(verbose=0)
if not M.ret.feasible:
    print("infeasible at M")
M.komo.view(True)
    #M.komo.report(True, True, True)

M1 = M.sub_motion(0, accumulated_collisions=False)
M1.retractPush([.0, .15], "l_gripper", .03)
M1.approachPush([.85, 1.], "l_gripper", .03)
M1.no_collisions([.15,.85], [object_, 'l_finger1'], .02)
M1.no_collisions([.15,.85], [object_, 'l_finger2'], .02)
M1.no_collisions([.15,.85], [object_, 'l_palm'], .02)
M1.no_collisions([], [table, 'l_finger1'], .0)
M1.no_collisions([], [table, 'l_finger2'], .0)
M1.solve(verbose=4)
path1 = M1.path
if not M1.ret.feasible:
    print("infeasible at M1")
    M1.komo.view(True)

M2 = M.sub_motion(1, accumulated_collisions=False)
#M2.komo.addObjective([2], ry.FS.position, [object_], ry.OT.eq, [1e1], placePosition)

M2.solve(verbose=4)
path2 = M2.path
if not M2.ret.feasible:
    print("infeasible at M2")
    M2.komo.view(True)

M1.komo.view(True)
M2.komo.view(True)