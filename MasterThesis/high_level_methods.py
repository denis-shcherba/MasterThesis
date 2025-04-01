import MasterThesis.manipulation as manip
from MasterThesis.simulator import Simulator
import numpy as np
import robotic as ry
import time


class RobotEnviroment:
    def __init__(self,
                 C: ry.Config,
                 visuals: bool=False,
                 verbose: int=0,
                 compute_collisions: bool=True,
                 sim: bool=False):
        self.C = C
        self.visuals = visuals
        self.sim = sim
        self.grabbed_frame = ""
        self.path = np.array([])
        self.compute_collisions = compute_collisions

    # KOMO implementation 
    def push_komo(self, frame: str, relative_y: float, relative_x: float = 0) -> bool:    
        komo = ry.KOMO()
        komo.setConfig(self.C, True)

        komo.setTiming(2, 20, 1., 2)

        komo.addControlObjective([], 0, 1e-2)
        komo.addControlObjective([], 1, 1e-1)
        komo.addControlObjective([], 2, 1e0)
        
        delta /= np.linalg.norm(delta)
        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)
        komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)

        #komo.addObjective([0,1], ry.FS.negDistance, ['l_gripper', 'mid_point'], ry.OT.ineq, [1], [-.1])
        komo.addObjective([1], ry.FS.positionDiff, ['l_gripper', 'way_start'], ry.OT.eq, [1e1])
        komo.addObjective([1,2], ry.FS.positionDiff, ['l_gripper', 'way_end'], ry.OT.eq, (np.eye(3)-np.outer(delta,delta)))

        komo.addObjective([2], ry.FS.positionDiff, ['l_gripper', 'way_end'], ry.OT.eq, [1e1])

        komo.addObjective([2], ry.FS.qItself, [], ry.OT.eq, [1e1], [], 1)   #no motion derivative of q vector ergo the velocity = 0 

        komo.addObjective([1, 2], ry.FS.vectorX, ['l_gripper'], ry.OT.eq, delta.reshape(1,3))
        komo.addObjective([1, 2], ry.FS.vectorZ, ['l_gripper'], ry.OT.eq, [1], [0,0,1])


    def push_frame_to(self, object_: str, placePosition) -> bool:
        gripper = "l_gripper"
        table = "table"

        info = f'push 1'
        print('===', info)

        M = ry.KOMO_ManipulationHelper(info)
        M.setup_sequence(self.C, 2, 1e-1, accumulated_collisions=False)
        M.komo.addFrameDof('obj_trans', table, ry.JT.transXY, False, object_) #a permanent moving(!) transXY joint table->trans, and a snap trans->obj
        M.komo.addRigidSwitch(1., ['obj_trans', object_])
        pushStart = M.straight_push([1.,2.], object_, gripper, table)

        M.komo.addObjective([2.], ry.FS.position, [object_], ry.OT.eq, 1e1*np.array([[1,0,0],[0,1,0]]), placePosition)
        M.solve()
        if not M.ret.feasible:
            return False

        M1 = M.sub_motion(0, accumulated_collisions=False)
        M1.retractPush([.0, .15], gripper, .03)
        M1.approachPush([.85, 1.], gripper, .03)
        M1.no_collisions([.15,.85], [object_, 'l_finger1'], .02)
        M1.no_collisions([.15,.85], [object_, 'l_finger2'], .02)
        M1.no_collisions([.15,.85], [object_, 'l_palm'], .02)
        M1.no_collisions([], [table, 'l_finger1'], .0)
        M1.no_collisions([], [table, 'l_finger2'], .0)
        M1.solve()
        if not M1.ret.feasible:
            return False

        M2 = M.sub_motion(1, accumulated_collisions=False)

        M2.solve()
        if not M2.ret.feasible:
            return False


        M1.play(self.C, 1.)
        self.C.attach(gripper, object_)
        M2.play(self.C, 1.)
        self.C.attach(table, object_)

        return True


    def move_to_point_path(self, point, relPos=None, straight_line = False, useRRT = False) -> bool:
        if relPos:
            if self.C.getFrame("_tmp_way") is None:
                self.C.addFrame('_tmp_way') \
                    .setShape(ry.ST.marker, [.1]) \
                    .setPosition(point)
            else:
                self.C.getFrame('_tmp_way') \
                    .setPosition(point) \

        gripper = "l_gripper"

        man = ry.KOMO_ManipulationHelper()
        man.setup_inverse_kinematics(self.C, accumulated_collisions=True)
        if relPos is None:
            man.komo.addObjective([1], ry.FS.position, [gripper], ry.OT.eq, 1, point)
        else:
            man.komo.addObjective([1], ry.FS.positionRel, [gripper, '_tmp_way'], ry.OT.eq, 1, relPos)

        if straight_line:
            #TODO
            # calucluate delta ...
            pass

        ret = man.solve()
        path = man.path
        print('    IK:', ret)
        
        if not ret.feasible:
            print('  -- infeasible')
            return False, None
            

        man = ry.KOMO_ManipulationHelper()
        man.setup_point_to_point_motion(self.C, path[0])
        
        ret = man.solve()
        print('  path:', ret)
        if not ret.feasible:
            print('  -- infeasible')
            return False, None
        
        return True, man.path

    def move_to_point(self, point, relPos=None, straight_line = False, useRRT = False) -> bool:
        
        feasible, path = self.move_to_point_path(point, relPos, straight_line, useRRT) 
        if feasible:
            if self.sim == True:
                #TODO prolly
                C2 = ry.Config()
                C2.addConfigurationCopy(self.C)
                sim = Simulator(C2)
                xs, qs, xdots, qdots = sim.run_trajectory(path, 2, real_time=True)

                self.C.setJointState(qs[-1])

            for t in range(path.shape[0]):
                self.C.setJointState(path[t])
                self.C.view(False)
                time.sleep(.05)
            
            return True
        else:
            print('  -- infeasible')
            return False

    def pull(self, object_, placePosition, debug=False):
        M = manip.ManipulationModelling()
        M.setup_pick_and_place_waypoints(self.C, "l_gripper", object_, 1e-1)
        M.pull([1.,2.], object_, "l_gripper", "table")
        M.komo.addObjective([2.], ry.FS.position, [object_], ry.OT.eq, 1e1, placePosition)
        M.solve()
        if not M.feasible:
            print("INFEASIBLE AT M")
            return False

        M1 = M.sub_motion(0, accumulated_collisions=False)
        M1.retractPush([.0, .15], "l_gripper", .03)
        M1.approachPush([.85, 1.], "l_gripper", .03)
        path1 = M1.solve()
        if not M1.feasible:
            print("INFEASIBLE AT M1")
            return False

        M2 = M.sub_motion(1, accumulated_collisions=False)
        path2 = M2.solve()
        if not M2.feasible:
            print("INFEASIBLE AT M2")
            return False

        M1.play(self.C, 1.)
        self.C.attach("l_gripper", object_)
        if debug:
            print(self.C.eval(ry.FS.negDistance, ["l_gripper", object_])[0])

            self.C.view(True)
        M2.play(self.C, 1.)
        self.C.attach("table", object_)

        return True

    def pivot(self):
        pass