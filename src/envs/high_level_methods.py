from envs.simulator import Simulator
import envs.manipulation as manip
import numpy as np
import robotic as ry
import time


class RobotEnviroment:
    def __init__(self,
                 C: ry.Config,
                 visuals: bool=False,
                 verbose: int=0,
                 compute_collisions: bool=True,
                 sim: bool=False,
                 gripper: str="l_gripper",
                 base_removal: bool=False,
                 observation_mode: str="POINTCLOUD",
                 visualize: bool=False,
                 path_mode: str=None,
                 noise_dict: dict={}) -> None:
        self.C = C
        self.visuals = visuals
        self.verbose = verbose
        self.sim = sim
        self.grabbed_frame = ""
        self.path = np.array([])
        self.compute_collisions = compute_collisions
        self.gripper = gripper
        self.base_removal = base_removal
        self.observation_mode = observation_mode
        self.visualize = visualize
        self.path_mode = path_mode
        self.noise_dict = noise_dict

    def push_frame_to(self, object_: str, placePosition) -> bool:
        table = "table"

        info = f'push 1'
        print('===', info)

        M = ry.KOMO_ManipulationHelper(info)
        M.setup_sequence(self.C, 2, 1e-1, accumulated_collisions=False)
        M.komo.addFrameDof('obj_trans', table, ry.JT.transXY, False, object_) #a permanent moving(!) transXY joint table->trans, and a snap trans->obj
        M.komo.addRigidSwitch(1., ['obj_trans', object_])
        pushStart = M.straight_push([1.,2.], object_, self.gripper, table)

        M.komo.addObjective([2.], ry.FS.position, [object_], ry.OT.eq, 1e1*np.array([[1,0,0],[0,1,0]]), placePosition)
        M.solve()
        if not M.ret.feasible:
            return False

        M1 = M.sub_motion(0, accumulated_collisions=False)
        M1.retractPush([.0, .15], self.gripper, .03)
        M1.approachPush([.85, 1.], self.gripper, .03)
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
        self.C.attach(self.gripper, object_)
        M2.play(self.C, 1.)
        self.C.attach(table, object_)

        return True


    def move_to_point_path(self, point, minDistance=None, straight_line = False, useRRT = False, accumulated_collisions = True) -> bool:

        if self.C.getFrame("_tmp_way") is None:
            self.C.addFrame('_tmp_way') \
                .setShape(ry.ST.marker, [.1]) \
                .setPosition(point)
        else:
            self.C.getFrame('_tmp_way') \
                .setPosition(point) \


        man = ry.KOMO_ManipulationHelper()
        man.setup_inverse_kinematics(self.C, accumulated_collisions=accumulated_collisions)
        if minDistance is None:
            man.komo.addObjective([1], ry.FS.position, [self.gripper], ry.OT.eq, 1, point)
        else:
            # straight line objective
            delta = point-self.C.getFrame("target_book").getPosition()
            delta /= np.linalg.norm(delta)
            print('delta:', delta)
            man.komo.addObjective([1], ry.FS.positionDiff, [self.gripper, '_tmp_way'], ry.OT.eq, [1e2*(np.eye(3)-np.outer(delta,delta))])
            man.komo.addObjective([1], ry.FS.negDistance, [self.gripper, "target_book"], ry.OT.ineq, 1e1, [-.05])

            # TODO not infront to push point
            #man.komo.addObjective([1], ry.FS.positionDiff, [self.gripper, "to_push_point"], ry.OT.ineq, [1, 0, 0])

        if straight_line:
            #TODO
            delta = point-self.C.getFrame(self.gripper).getPosition()
            delta /= np.linalg.norm(delta)
            man.komo.addObjective([], ry.FS.positionDiff, [self.gripper, '_tmp_way'], ry.OT.eq, [3e1*(np.eye(3)-np.outer(delta,delta))])

        ret = man.solve()

        path = man.path
        print('    IK:', ret)
        
        if not ret.feasible:
            print('  -- infeasible')
            return False, None
            

        man = ry.KOMO_ManipulationHelper()
        man.setup_point_to_point_motion(self.C, path[0], accumulated_collisions=accumulated_collisions)


        ret = man.solve()

        print('  path:', ret)
        if not ret.feasible:
            print('  -- infeasible')
            return False, None
        
        return True, man.path


    def run_path(self, path):
        if self.sim == True:
            sim = Simulator(self.C, verbose=self.verbose)
            sim.run_trajectory_spline(path, 2)
        else:
            for t in range(path.shape[0]):
                self.C.setJointState(path[t])
                self.C.view(False)
                time.sleep(.05)
        

    def move_to_point(self, point, minDistance=None, straight_line = False, useRRT = False, accumulated_collisions=True, book_point_line = False) -> bool:
        
        feasible, path = self.move_to_point_path(point, minDistance, straight_line, useRRT, accumulated_collisions) 
        if feasible:
            if self.sim == True:
                sim = Simulator(self.C, verbose=self.verbose)
                sim.run_trajectory_spline(path, 2)
            
            else:
                for t in range(path.shape[0]):
                    self.C.setJointState(path[t])
                    self.C.view(False)
                    time.sleep(.05)
            
            return True
        else:
            print('  -- infeasible')
            return False


    def pull(self, object_, placePosition, accumulated_collisions=True, get_observation=False) -> bool:
        self.C.addFrame("tmp").setPosition(self.C.getFrame(object_).getPosition())
        M = manip.ManipulationModelling()
        M.setup_pick_and_place_waypoints(self.C, self.gripper, object_, 1e-1, accumulated_collisions=accumulated_collisions)
        
        M.add_stable_frame(ry.JT.transXYPhi, "big_xy_bottom_0_1", '_pull_end', object_)
        M.komo.addObjective([1], ry.FS.vectorZ, [self.gripper], ry.OT.eq, [1e1], np.array([0,0,1]))
        M.komo.addObjective([2], ry.FS.vectorZ, [self.gripper], ry.OT.eq, [1e1], np.array([0,0,1]))
        M.komo.addObjective([1], ry.FS.vectorZ, [object_], ry.OT.eq, [1e1], np.array([0,0,1]))
        M.komo.addObjective([2], ry.FS.vectorZ, [object_], ry.OT.eq, [1e1], np.array([0,0,1]))
        M.komo.addObjective([2], ry.FS.positionDiff, [object_, '_pull_end'], ry.OT.eq, [1e1])
        print(self.C.getFrame(object_).getSize()[2])
        M.komo.addObjective([1], ry.FS.positionRel, [self.gripper, object_], ry.OT.eq, 1e2, np.array([0, 0, .01-.5*self.C.getFrame(object_).getSize()[2]]))

        M.komo.addObjective([2.], ry.FS.position, [object_], ry.OT.eq, 1e1, placePosition)
        M.komo.addObjective([1,2], ry.FS.position, [self.gripper], ry.OT.eq, [0, 0, 1], [self.C.getFrame(object_).getSize()[2]+self.C.getFrame(object_).getPosition()[2]])   

        M.solve()
        if not M.feasible:
            print("INFEASIBLE AT M")
            self.C.delFrame("tmp")
            return False

        M1 = M.sub_motion(0, accumulated_collisions=accumulated_collisions)
        M1.retractPush([.0, .15], self.gripper, .03)
        M1.approachPush([.85, 1.], self.gripper, .03)
        path1 = M1.solve()
        if not M1.feasible:
            print("INFEASIBLE AT M1")
            self.C.delFrame("tmp")
            return False

        M2 = M.sub_motion(1, accumulated_collisions=False)

        #target = self.C.getFrame("target").getPosition()
        # delta = np.array(target) - self.C.getFrame(object_).getPosition()
        # delta /= np.linalg.norm(delta)
        # projection_matrix = np.eye(3) - np.outer(delta, delta)
        # M2.komo.addObjective([1], ry.FS.positionDiff, [object_, "tmp"], ry.OT.eq, 1e1 * projection_matrix)
        # M2.komo.addObjective([.5,1], ry.FS.positionDiff, [object_, "tmp"], ry.OT.eq, 1e1 * np.array([0, 0, 1]))

        path2 = M2.solve()

        if not M2.feasible:
            print("INFEASIBLE AT M2")
            self.C.delFrame("tmp")

            return False
        
        if self.sim == True:
            # TODO calculate offset for fix force given PD properties
            offset = -.01
            # same as path2 + np.array([0, 0, offset, 0, 0, 0, 0]) for floating gripper
            path2_after_offset = []       
            C2 = ry.Config()
            C2.addConfigurationCopy(self.C)
            
            delta_x = np.array([0, 0, offset])  
            
            for q in path2:
                C2.setJointState(q)
                _, J = C2.eval(ry.FS.position, ['gripper'])
                delta_q = np.linalg.pinv(J) @ delta_x
                path2_after_offset.append(q + delta_q)
            
            path2 = path2_after_offset
            del C2

            sim = Simulator(self.C, verbose=self.verbose, base_removal=self.base_removal)
            if "SPLINE" in self.path_mode == "DELTA3DSPLINE":
                sim.run_trajectory_spline(np.array(path1), 2, capture_depth=get_observation)
                sim.run_trajectory_spline(np.asarray(path2_after_offset), 2, capture_depth=get_observation)
            else:
                sim.run_trajectory_position_control(np.array(path1), n_steps=2, tau=0.01, capture_depth=get_observation, visualize=self.visualize)
                sim.run_trajectory_position_control(np.array(path2_after_offset), n_steps=2,  tau=0.01, capture_depth=get_observation, visualize=self.visualize)


            if self.observation_mode == "POINTCLOUD":
                self.points = sim.points
            elif self.observation_mode == "RGB":
                self.rgb_image = sim.rgb
            elif self.observation_mode == "DEPTH":
                self.depth_image = sim.depth

        else:
            if self.visuals:
                M1.play(self.C, 1.)
                self.C.view(True)

                self.C.attach(self.gripper, object_)

                M2.play(self.C, 1.)

                self.C.attach("big_xy_bottom_0_1", object_)
            else:
                if get_observation:
                    sim = Simulator(self.C, verbose=self.verbose, base_removal=self.base_removal)
                    if self.observation_mode == "POINTCLOUD":
                        self.points = sim.getPoints(vis=True)
                    #self.C.setJointState(path1[-1])
                    #self.C.view(True)
                    elif self.observation_mode == "RGB":
                        self.rgb_image = sim.getRGB()

        self.C.delFrame("tmp")

        self.path = np.concatenate((path1, path2_after_offset), axis=0)
        return True

    def pull_point():
        #TODO
        pass

    def render(self, n_samples=4096):
        if self.sim:
            sim = Simulator(self.C, verbose=self.verbose)
            if self.observation_mode.upper() == "POINTCLOUD" or self.observation_mode.upper() == "POINTS":
                points = sim.getPoints(n_samples, vis=False)
                return points
            elif self.observation_mode.upper() == "RGB":
                pass   
            if self.observation_mode.upper() == "DEPTH":
                depth = sim.getDepth()
                return depth
        
        else:
            #TODO? maybe, maybe not
            self.C.view(True)
