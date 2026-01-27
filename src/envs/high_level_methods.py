from envs.simulator import Simulator
import envs.manipulation as manip
import numpy as np
import robotic as ry
import time
import pyrealsense2 as rs
import cv2

class RobotEnviroment:
    def __init__(self,
                 C: ry.Config,
                 visuals: bool=False,
                 verbose: int=0,
                 compute_collisions: bool=True,
                 sim: bool=False,
                 gripper: str="l_gripper",
                 base_removal: bool=False,
                 observation_mode: str="DEPTH",
                 visualize: bool=False,
                 path_mode: str="",
                 noise_dict: dict={},
                 camera: str="cameraStatic",
                 depth_noise = False,
                 on_real = False) -> None:
        self.C = C
        self.visuals = visuals
        self.verbose = verbose
        self.sim = sim
        self.on_real = on_real
        self.grabbed_frame = ""
        self.path = np.array([])
        self.compute_collisions = compute_collisions
        self.gripper = gripper
        self.base_removal = base_removal
        self.observation_mode = observation_mode
        self.visualize = visualize
        self.path_mode = path_mode
        self.noise_dict = noise_dict
        self.camera = camera
        self.depth_noise = depth_noise  
        if self.noise_dict:
            self.state_noise = self.noise_dict.get("stateNoise")
            self.depth_noise = self.noise_dict.get("depthNoise")

        if self.on_real:
            self.bot = ry.BotOp(self.C, True)
            self.bot.gripperMove(ry._left, 0)
            while not self.bot.gripperDone(ry._left):
                self.bot.wait(self.C)
            self._rs_pipeline = rs.pipeline()
            self._rs_config = rs.config()

            # Enable both streams independently
            # They will be time-synchronized but not spatially aligned
            self._rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self._rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

            self._rs_profile = self._rs_pipeline.start(self._rs_config)

            # Warm-up frames (to let auto-exposure settle)
            for _ in range(10):
                self._rs_pipeline.wait_for_frames()
        
        else:
            self._rs_pipeline = None
            self._rs_config = None
            self._rs_profile = None

    def _rs_get_color(self):
        """Grab a color frame from RealSense and return RGB uint8 (H, W, 3)."""
        if self._rs_pipeline is None:
            return None
        frames = self._rs_pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        import numpy as np
        import cv2
        bgr = np.asanyarray(color_frame.get_data())
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    def _rs_get_data(self):
        """Grab both color and depth frames simultaneously."""
        if self._rs_pipeline is None:
            return None, None
        
        # wait_for_frames blocks until both streams have a frame available
        frames = self._rs_pipeline.wait_for_frames()
        
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            return None, None
                
        # Process Color
        bgr = np.asanyarray(color_frame.get_data())
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        # Process Depth
        # This is a (480, 640) array of uint16 values
        depth = np.asanyarray(depth_frame.get_data())
        
        return rgb, depth

    def _rs_shutdown(self):
            if self._rs_pipeline is not None:
                try:
                    self._rs_pipeline.stop()
                except Exception:
                    pass
                self._rs_pipeline = None

    def move(self, q):
        if self.sim == True:
            self.C.setJointState(q)
            if self.visuals:
                self.C.view(False)
                time.sleep(.05)
        elif self.on_real:
            self.bot.moveTo(q)
            while(self.bot.getTimeToEnd() > 0):
                self.bot.sync(self.C, .1)

        else:
            self.C.setJointState(q)
            if self.visuals:
                self.C.view(False)
                time.sleep(.05)

    def hook_book(self, object_: str, base="big_xy_bottom_0_1") -> bool:
        direction_vec = self.C.getFrame(object_).getPosition() - self.C.getFrame("target").getPosition()
        direction_vec /= np.linalg.norm(direction_vec) 

        theta = np.arccos(np.dot(direction_vec, np.array([0, 1, 0])))
        theta_acute = min(theta, np.pi - theta)
        high_on_potenuse = self.C.getFrame(object_).getSize()[0]/(2 * np.sin(theta_acute))
        
        self.C.addFrame("hook_point").setPosition(self.C.getFrame(object_).getPosition() - np.array([0, 0, .01]) + (high_on_potenuse+.04) * direction_vec).setShape(ry.ST.marker, [.05]).setQuaternion(ry.Quaternion().setEuler([0, 0, -theta]).asArr())
        self.C.addFrame("end_point").setPosition(self.C.getFrame("target").getPosition() + (high_on_potenuse) * direction_vec)#.setShape(ry.ST.marker, [.2])

        M = manip.ManipulationModelling()
        M.setup_pick_and_place_waypoints(self.C, self.gripper, object_, 1e-1, accumulated_collisions=True)

        M.komo.addObjective([1], ry.FS.positionDiff, ['hook_tip', "hook_point"], ry.OT.eq, [1e2])
        # if self.C.getJointDimension() > 3:
        #     M.komo.addObjective([1], ry.FS.scalarProductYX, [self.gripper, "hook_point"], ry.OT.eq)

        M.komo.addObjective([2.], ry.FS.positionDiff, ["hook_tip", "end_point"], ry.OT.eq, [1, 1, 1])

        M.solve()
        if not M.feasible:
            print("INFEASIBLE AT M")
            self.C.delFrame("hook_point")
            return False
        M.komo.view(True)

        M1 = M.sub_motion(0, accumulated_collisions=True)
        M1.retractPush([.0, .15], self.gripper, .03)
        M1.approachPush([.85, 1.], self.gripper, .03)

        path1 = M1.solve()
        if not M1.feasible:
            #M1.komo.view(True)

            print("INFEASIBLE AT M1")
            self.C.delFrame("hook_point")
            return False
    

        M2 = M.sub_motion(1, accumulated_collisions=False)
        M2.komo.addObjective([0,1], ry.FS.positionDiff, ["hook_tip", "end_point"], ry.OT.eq, [0, 0, 1e1], [])   

        path2 = M2.solve()

        if not M2.feasible:
            print("INFEASIBLE AT M2")
            self.C.delFrame("hook_point")
            return False
        M2.komo.view(True)

        if self.sim == True:
            sim = Simulator(self.C, verbose=self.verbose, base_removal=self.base_removal, camera=self.camera, observation_mode=self.observation_mode, depth_noise=self.depth_noise)

            sim.run_trajectory_position_control(np.array(path1), n_steps=2, tau=0.01, capture_obs=True, visualize=True)
            sim.run_trajectory_position_control(np.array(path2), n_steps=2,  tau=0.01, capture_obs=True, visualize=True)

        elif self.on_real:
            # Use RealSense color-only for raw RGB capture (no depth alignment)
            path = np.concatenate((path1, path2), axis=0)

            imgs = []
            depth =[]
            qs = []
            timings = np.linspace(.15, 9.6, 64)
            self.bot.move(path, timings)
            i = 0
            self.bot.sync(self.C)

            t_start = self.bot.get_t()
            t_now = t_start
            t_last = 0
            for i in range(64):
                rgb, depth_data = self._rs_get_data()
                actual_q = self.bot.get_q()

                if rgb is not None:
                    imgs.append(rgb)
                if depth_data is not None:
                    depth.append(depth_data)
                qs.append(actual_q)

                t_now = self.bot.get_t() - t_start
                print(i, t_now)
                print("time since last frame:", t_now - t_last)
                t_last = t_now
                print(actual_q - path[i])
                # Still sync the robot so it moves along the path
                self.bot.sync(self.C, 0.15)


            print(timings)
            self.rgb_image = np.array(imgs, dtype=np.uint8)#[:, :260, 180:520, :]
            self.depth_image = np.array(depth, dtype=np.int16)
            self.captured_qs = np.array(qs, dtype=np.float32)
        else:
            if self.visuals:
                M1.play(self.C, 1.)
                self.C.attach(self.gripper, object_)
                M2.play(self.C, 1.)

                self.C.attach(base, object_)

        if "POINTCLOUD" in self.observation_mode or "SAM_POINTS" in self.observation_mode or "BOX_POINTS" in self.observation_mode:
            self.points = sim.points[0]
            if self.points[0] is None:
                return False
            for pc in self.points:
                if pc is None:
                    return False

        if "RGB" in self.observation_mode:
            if self.sim:
                self.rgb_image = sim.rgb
        if "DEPTH" in self.observation_mode:
            if self.sim:
                self.depth_image = sim.depth

        self.path = np.concatenate((path1, path2), axis=0)

        return True


    def push_point_to_point(self, startPoint, endPoint, pcl = None, bot=False) -> bool:
        # same as pushFrame to but for predicted start and end points
        table = "table"

        info = f'push 1'
        print('===', info)

        M = ry.KOMO_ManipulationHelper(info)
        # M.setup_pick_and_place_waypoints(self.C, self.gripper, object_, 1e-1, accumulated_collisions=True)

        M.setup_sequence(self.C, 2, 1e-1, accumulated_collisions=False)

        M.komo.addObjective([1.], ry.FS.positionDiff, [self.gripper, startPoint], ry.OT.eq, 1e1, [0, 0, .02])
        M.komo.addObjective([2.], ry.FS.positionDiff, [self.gripper, endPoint], ry.OT.eq, 1e1, [0, 0, .02])
        M.solve()
        if not M.ret.feasible:
            return False
        M1 = M.sub_motion(0, accumulated_collisions=False)
        M1.retractPush([.0, .15], self.gripper, .03)
        M1.approachPush([.85, 1.], self.gripper, .03)
        # TODO
        #M1.komo.addObjective([.15, .85], ry.FS.pairCollision_negScalar, [self.gripper, pcl], ry.OT.ineq, 1e1)
        # M1.no_collisions([.15,.85], [pcl, 'l_finger1'], .02)
        # M1.no_collisions([.15,.85], [pcl, 'l_finger2'], .02)
        # M1.no_collisions([.15,.85], [pcl, 'l_palm'], .02)
        M1.no_collisions([], [table, 'l_finger1'], .0)
        M1.no_collisions([], [table, 'l_finger2'], .0)
        M1.solve()
        path1 = M1.path
        if not M1.ret.feasible:
            return False

        M2 = M.sub_motion(1, accumulated_collisions=False)
        #M2.komo.addObjective([2], ry.FS.position, [object_], ry.OT.eq, [1e1], placePosition)

        M2.solve()
        path2 = M2.path
        if not M2.ret.feasible:
            return False


        if bot:
            bot.moveAutoTimed(path1)
            while(bot.getTimeToEnd() > 0):
                bot.wait(self.C, .1)
            bot.moveAutoTimed(path2)
            while(bot.getTimeToEnd() > 0):
                bot.wait(self.C, .1)
        else:
            sim = Simulator(self.C, verbose=self.verbose, base_removal=self.base_removal, camera=self.camera, observation_mode=self.observation_mode, depth_noise=self.depth_noise)

            sim.run_trajectory_position_control(np.array(path1), n_steps=.2, tau=0.01, capture_obs=False, visualize=True)
            sim.run_trajectory_position_control(np.array(path2), n_steps=.2,  tau=0.01, capture_obs=False, visualize=True)




        return True

    def straight_push(self, M, time_interval, obj, gripper, table):
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
        M.komo.addObjective([time_interval[0]], ry.FS.negDistance, [gripper, obj], ry.OT.eq, [1e1], [-.0175])
        #gripper start position
        M.komo.addObjective([time_interval[0]], ry.FS.positionRel, [gripper, helperStart], ry.OT.eq, 1e1*np.array([[1., 0., 0.], [0., 0., 1.]]))
        M.komo.addObjective([time_interval[0]], ry.FS.positionRel, [gripper, helperStart], ry.OT.ineq, 1e1*np.array([[0., 1., 0.]]), [.0, -.04, .0])
        #gripper start orientation
        #M.komo.addObjective([time_interval[0]], ry.FS.scalarProductYY, [gripper, helperStart], ry.OT.ineq, [-1e0], [.2])
        #M.komo.addObjective([time_interval[0]], ry.FS.scalarProductYZ, [gripper, helperStart], ry.OT.ineq, [-1e0], [.2])
        M.komo.addObjective([time_interval[0]], ry.FS.scalarProductYZ, [gripper, helperStart], ry.OT.eq, [1e0])   # Deno


        M.komo.addObjective([time_interval[0]], ry.FS.vectorXDiff, [gripper, helperStart], ry.OT.eq, [1e0])
        M.freeze_relativePose([time_interval[1]], gripper, obj)
    
        return helperStart

    def push_frame_to(self, object_: str, placePosition, get_observation) -> bool:
        table = "table"

        info = f'push 1'
        print('===', info)

        if self.on_real:
            self.C.getFrame(object_).setPosition(self.C.getFrame(object_).getPosition() + np.array([0,0,.027]))
            self.C.view(False)

        M = ry.KOMO_ManipulationHelper(info)
        # M.setup_pick_and_place_waypoints(self.C, self.gripper, object_, 1e-1, accumulated_collisions=True)

        M.setup_sequence(self.C, 2, 1e-1, accumulated_collisions=False)
        M.komo.addFrameDof('obj_trans', table, ry.JT.transXY, False, object_) #a permanent moving(!) transXY joint table->trans, and a snap trans->obj
        M.komo.addRigidSwitch(1., ['obj_trans', object_])
        pushStart = self.straight_push(M, [1.,2.], object_, self.gripper, table)

        M.komo.addObjective([2.], ry.FS.position, [object_], ry.OT.eq, 1e1*np.array([[1,0,0],[0,1,0]]), placePosition)
        M.solve()
        #M.komo.view(True)
        if not M.ret.feasible:
            print("infeasible at M")
            # M.komo.view(True)
            # M.komo.report(True, True, True)
            return False

        M1 = M.sub_motion(0, accumulated_collisions=False)
        M1.komo.addObjective([], ry.FS.vectorZ, [self.gripper], ry.OT.eq, 1, [0, 0, 1])
        M1.retractPush([.0, .15], self.gripper, .03)
        M1.approachPush([.85, 1.], self.gripper, .03)
        M1.no_collisions([.15,.85], [object_, 'l_finger1'], .02)
        M1.no_collisions([.15,.85], [object_, 'l_finger2'], .02)
        M1.no_collisions([.15,.85], [object_, 'l_palm'], .02)
        M1.no_collisions([], [table, 'l_finger1'], .0)
        M1.no_collisions([], [table, 'l_finger2'], .0)
        M1.solve()
        path1 = M1.path
        if not M1.ret.feasible:
            print("infeasible at M1")
            # M1.komo.view(True)
            return False

        M2 = M.sub_motion(1, accumulated_collisions=False)
        #M2.komo.addObjective([2], ry.FS.position, [object_], ry.OT.eq, [1e1], placePosition)
        M2.komo.addObjective([], ry.FS.vectorZ, [self.gripper], ry.OT.eq, 1, [0, 0, 1])


        M2.solve()
        path2 = M2.path
        if not M2.ret.feasible:
            print("infeasible at M2")
            # M2.komo.view(True)
            return False


        if self.sim == True:
            sim = Simulator(self.C, verbose=self.verbose, base_removal=self.base_removal, camera=self.camera, observation_mode=self.observation_mode, depth_noise=self.depth_noise)

            sim.run_trajectory_position_control(np.array(path1), n_steps=2, tau=0.01, capture_obs=get_observation, visualize=False)
            sim.run_trajectory_position_control(np.array(path2[:, :7]), n_steps=2,  tau=0.01, capture_obs=get_observation, visualize=False)

        elif self.on_real:
            # Use RealSense color-only for raw RGB capture (no depth alignment)
            path = np.concatenate((path1, path2[:, :7]), axis=0)

            imgs = []
            timings = np.linspace(.1, 10, 100)
            self.bot.move(path, timings)
            i = 0
            t_start = self.bot.get_t()
            while self.bot.getTimeToEnd() > 0:
                rgb = self._rs_get_color()
                if rgb is not None:
                    imgs.append(rgb)
                print(i, self.bot.get_t() - t_start)
                
                sin_comp = self.C.eval(ry.FS.scalarProductXY, [self.gripper, "table"])[0]
                cos_comp = self.C.eval(ry.FS.scalarProductXX, [self.gripper, "table"])[0]

                yaw = np.arctan2(sin_comp, cos_comp) 
                print("Yaw angle (rad):", yaw)
                i += 1
                self.bot.sync(self.C, .1)

            print(timings)


            self.rgb_image = np.array(imgs, dtype=np.uint8)

            # Cleanly stop the RealSense pipeline

            # for i in range(path1.shape[0]):
            #     # TODO smooth with overwrite after overshooting?
            #     self.bot.move([path1[i, :]], [.3])
            #     while self.bot.getTimeToEnd() > 0:
            #         self.bot.wait(self.C)

            # for i in range(path2.shape[0]):
            #     self.bot.move([path2[i, :7]], [.3])
            #     while self.bot.getTimeToEnd() > 0:
            #         self.bot.wait(self.C)


            # self.bot.moveAutoTimed(path1)
            # while(self.bot.getTimeToEnd() > 0):
            #     self.bot.wait(self.C, .1)
            # self.bot.moveAutoTimed(path2[:, :7])
            # while(self.bot.getTimeToEnd() > 0):
            #     self.bot.wait(self.C, .1)

        else:
            if self.visuals:
                M1.play(self.C, 1.)
                self.C.attach(self.gripper, object_)
                M2.play(self.C, 1.)


        if self.observation_mode == "POINTCLOUD" or self.observation_mode =="SAM_POINTS" or self.observation_mode =="BOX_POINTS":
            self.points = sim.points[0]
            if self.points[0] is None:
                return False
            for pc in self.points:
                if pc is None:
                    return False

        elif self.observation_mode == "RGB":
            if self.sim == True:
                self.rgb_image = sim.rgb
        elif self.observation_mode == "DEPTH":
            if self.sim == True:
                self.depth_image = sim.depth
                # self.C.addFrame("camera_marker", "cameraStaticTableTripod").setShape(ry.ST.marker, [.3])
                # self.C.view(True)
                # import matplotlib.pyplot as plt
                # plt.imshow(self.depth_image[0], cmap='gray')
                # plt.show()
                # plt.close()


        self.ways = []
        C2 = ry.Config()
        C2.addConfigurationCopy(self.C)
        
        C2.setJointState(path1[-1])
        self.ways.append(C2.getFrame(self.gripper).getPosition())
        
        C2.setJointState(path2[-1, :7])
        self.ways.append(C2.getFrame(self.gripper).getPosition())
        del C2


        self.path = np.concatenate((path1, path2[:, :7]), axis=0)
        return True


    def move_to_point_path(self, point, minDistance=None, straight_line = False, useRRT = False, straight_gripper=False, accumulated_collisions = True) -> bool:

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

        if straight_gripper:
            man.komo.addObjective([1], ry.FS.vectorZ, [self.gripper], ry.OT.eq, 1, [0, 0, 1])


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
        

    def move_to_point(self, point, minDistance=None, straight_gripper=False, straight_line = False, useRRT = False, accumulated_collisions=True, book_point_line = False) -> bool:
        
        feasible, path = self.move_to_point_path(point, minDistance, straight_line, useRRT, straight_gripper, accumulated_collisions) 
        if feasible:
            if self.sim == True:
                sim = Simulator(self.C, verbose=self.verbose, camera=self.camera)
                sim.run_trajectory_spline(path, 2)
            elif self.on_real:
                self.bot.moveTo(path[-1, :])
                while(self.bot.getTimeToEnd() > 0):
                    self.bot.wait(self.C, .1)
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
        
        # #  --- Noise addition for demos ---
        # path1Offset = np.zeros((32, 3))
        # path2Offset = np.zeros((32, 3))

        # if self.noise_dict:
        #     if self.state_noise.get("type") == "singleGaussian":  
        #         for i in range(len(path1Offset)):
        #             if np.random.rand() < self.state_noise.get("prob", 0):
        #                 path1Offset[i] += np.random.normal(self.state_noise.get("mean"), self.state_noise.get("std"), size=3)
        #         for i in range(len(path1Offset)):
        #             if np.random.rand() < self.state_noise.get("prob", 0):
        #                 path2Offset += np.random.normal(self.state_noise.get("mean"), self.state_noise.get("std"), size=3)

        #     elif self.state_noise.get("type") == "TS":
        #         # TODO add transition simulation
        #         pass

        #     # -----------------------------------

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

        # path1 += path1Offset
        # path2 += path2Offset

        if not M2.feasible:
            print("INFEASIBLE AT M2")
            self.C.delFrame("tmp")

            return False
        
        if self.sim == True:
            # TODO calculate offset for fix force given PD properties
            offset = -0.005
            # same as path2 + np.array([0, 0, offset, 0, 0, 0, 0]) for floating gripper
            path2_after_offset = []       
            C2 = ry.Config()
            C2.addConfigurationCopy(self.C)
            
            delta_x = np.array([0, 0, offset])  
            
            for q in path2:
                C2.setJointState(q)
                _, J = C2.eval(ry.FS.position, [self.gripper])
                delta_q = np.linalg.pinv(J) @ delta_x
                path2_after_offset.append(q + delta_q)
            
            path2 = path2_after_offset
            del C2
    
            sim = Simulator(self.C, verbose=self.verbose, base_removal=self.base_removal, obs=self.observation_mode)
            if "SPLINE" in self.path_mode:
                sim.run_trajectory_spline(np.array(path1), 2, capture_depth=get_observation)
                sim.run_trajectory_spline(np.asarray(path2), 2, capture_depth=get_observation)
            else:
                sim.run_trajectory_position_control(np.array(path1), n_steps=2, tau=0.01, capture_obs=get_observation, visualize=self.visualize)
                sim.run_trajectory_position_control(np.array(path2), n_steps=2,  tau=0.01, capture_obs=get_observation, visualize=self.visualize)


            if self.observation_mode == "POINTCLOUD" or "SAM_POINTS":
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

        if self.path_mode == "WAYplusTIMING":
            #TODO what about other cases (more pulls, pushes?)
            
            self.ways = []
            C2 = ry.Config()
            C2.addConfigurationCopy(self.C)
            
            C2.setJointState(path1[-1])
            self.ways.append(C2.getFrame(self.gripper).getPosition())
            
            C2.setJointState(path2_after_offset[-1])
            self.ways.append(C2.getFrame(self.gripper).getPosition())
            del C2

            self.timings = np.concatenate([np.full(32, 1), np.full(32, 2)])

        self.path = np.concatenate((path1, path2_after_offset), axis=0)
        return True

    def pull_real(self, object_, placePosition, accumulated_collisions=True, get_observation=False, base="big_xy_bottom_0_1") -> bool:
        self.C.addFrame("tmp").setPosition(self.C.getFrame(object_).getPosition())

        #q0 = self.C.getJointState()

        M = manip.ManipulationModelling()
        M.setup_pick_and_place_waypoints(self.C, self.gripper, object_, 1e-1, accumulated_collisions=accumulated_collisions)
        
        M.add_stable_frame(ry.JT.transXYPhi, base, '_pull_end', object_)


        # M.komo.addObjective([1], ry.FS.vectorZ, [self.gripper], ry.OT.eq, [1e1], np.array([0,0,1]))
        # M.komo.addObjective([2], ry.FS.vectorZ, [self.gripper], ry.OT.eq, [1e1], np.array([0,0,1]))

        M.komo.addObjective([2], ry.FS.positionDiff, [object_, '_pull_end'], ry.OT.eq, [1e1, 1e1, 0])
        print(self.C.getFrame(object_).getSize()[2])
        M.komo.addObjective([1], ry.FS.positionRel, [self.gripper, object_], ry.OT.eq, 1e2, np.array([0, 0, .5*self.C.getFrame(object_).getSize()[2]+.01]))
        M.komo.addObjective([2.], ry.FS.position, [object_], ry.OT.eq, 1e1, placePosition)

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
        
            # print("INFEASIBLE AT M1 with KOMO, trying RRT")

            # rrt = ry.RRT_PathFinder()
            # rrt.setProblem(self.C)
            # rrt.setOptions(verbose=1, stepsize=.1, subsamples=4, maxIters=5000, p_connect=.5, collisionTolerance=.0001, useBroadCollisions=True)
            # rrt.setStartGoal([q0], [M1.komo.getPath()[-1]])

            # ret = rrt.solve()
            # path1 = ret.x
            # #rrt.view(True)
            # if not ret.feasible:
            #     print("INFEASIBLE AT M1 with RRT")
            #     self.C.delFrame("tmp")
            #     return False

        M2 = M.sub_motion(1, accumulated_collisions=False)
        M2.komo.addObjective([0,1], ry.FS.position, [self.gripper], ry.OT.eq, [0, 0, 1e3], [], 1)   
        
        path2 = M2.solve()

        if not M2.feasible:
            print("INFEASIBLE AT M2")
            self.C.delFrame("tmp")

            return False
        
        if self.sim == True:
            offset = -0.01
            path2_after_offset = []       
            C2 = ry.Config()
            C2.addConfigurationCopy(self.C)
            
            delta_x = np.array([0, 0, offset])  
            
            for q in path2:
                C2.setJointState(q)
                _, J = C2.eval(ry.FS.position, [self.gripper])
                delta_q = np.linalg.pinv(J) @ delta_x
                path2_after_offset.append(q + delta_q)
            
            path2 = path2_after_offset
            del C2
    
            sim = Simulator(self.C, verbose=self.verbose, base_removal=self.base_removal, camera=self.camera, observation_mode=self.observation_mode, depth_noise=self.depth_noise)
            if "SPLINE" in self.path_mode:
                sim.run_trajectory_spline(np.array(path1), 2, capture_depth=get_observation)
                sim.run_trajectory_spline(np.asarray(path2), 2, capture_depth=get_observation)
            else:
                sim.run_trajectory_position_control(np.array(path1), n_steps=2, tau=0.01, capture_obs=get_observation, visualize=self.visualize)
                sim.run_trajectory_position_control(np.array(path2), n_steps=2,  tau=0.01, capture_obs=get_observation, visualize=self.visualize)


            if self.observation_mode == "POINTCLOUD" or self.observation_mode =="SAM_POINTS":
                self.points = sim.points
                if self.points[0] is None:
                    return False
                for pc in self.points:
                    if pc is None:
                        return False

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

                self.C.attach(base, object_)
            else:
                if get_observation:
                    sim = Simulator(self.C, verbose=self.verbose, base_removal=self.base_removal)
                    if self.observation_mode == "POINTCLOUD":
                        self.points = sim.getPoints(vis=True)
                    elif self.observation_mode == "RGB":
                        self.rgb_image = sim.getRGB()

        self.C.delFrame("tmp")

        self.waypoints = []
        C2 = ry.Config()
        C2.addConfigurationCopy(self.C)
        
        C2.setJointState(path1[-1])
        self.waypoints.append(C2.getFrame(self.gripper).getPosition())
        
        del C2

        if self.path_mode == "WAYplusTIMING":
            
            self.ways = []
            C2 = ry.Config()
            C2.addConfigurationCopy(self.C)
            
            C2.setJointState(path1[-1])
            self.ways.append(C2.getFrame(self.gripper).getPosition())
            
            C2.setJointState(path2_after_offset[-1])
            self.ways.append(C2.getFrame(self.gripper).getPosition())
            del C2

            self.timings = np.concatenate([np.full(32, 1), np.full(32, 2)])

        self.path = np.concatenate((path1, path2_after_offset), axis=0)
        return True

    def pull_way2way(self, pullWay, placeWay, accumulated_collisions=True) -> bool:

        M = manip.ManipulationModelling()
        #M.setup_motion(self.C, K=2, steps_per_phase=1, homing_scale=.1, acceleration_scale=1, accumulated_collisions=False, joint_limits=True, quaternion_norms=False)
        M.setup_sequence(self.C, 2, homing_scale=.01, velocity_scale=.1, accumulated_collisions=False, joint_limits=True, quaternion_norms=False)
        M.komo.addObjective([1], ry.FS.positionDiff, [self.gripper, pullWay], ry.OT.eq, [1])
        M.komo.addObjective([2], ry.FS.positionRel, [self.gripper, pullWay], ry.OT.eq, [1], [.1, 0, 0])
        M.komo.addObjective([1], ry.FS.vectorZ, [self.gripper], ry.OT.eq, [1e1], np.array([0,0,1]))
        M.komo.addObjective([2], ry.FS.vectorZ, [self.gripper], ry.OT.eq, [1e1], np.array([0,0,1]))

        M.solve()


        M1 = M.sub_motion(0, accumulated_collisions=False)
        M1.retractPush([.0, .15], self.gripper, .03)
        M1.approachPush([.85, 1.], self.gripper, .03)
        path1 = M1.solve()

        M2 = M.sub_motion(1, accumulated_collisions=False)
        M2.komo.addObjective([0,1], ry.FS.position, [self.gripper], ry.OT.eq, [0, 0, 1e3], [], 1)   

        path2 = M2.solve()

        #M1.komo.view(True)

        if not M.feasible:
            print("INFEASIBLE AT M")
            return False

        self.path = np.concatenate((path1, path2), axis=0)

        # if self.sim == True:
        #     sim = Simulator(self.C, verbose=self.verbose, base_removal=False, camera=self.camera)
        #     sim.run_trajectory_position_control(np.array(path1), n_steps=2, tau=0.1, capture_obs=False, visualize=True)
        #     sim.run_trajectory_position_control(np.array(path2), n_steps=2, tau=0.1, capture_obs=False, visualize=True)


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


