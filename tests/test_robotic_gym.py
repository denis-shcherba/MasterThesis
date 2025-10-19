#!/usr/bin/env python

# install the 'robotic' package: https://github.com/MarcToussaint/robotic/

import robotic as ry
import gymnasium as gym
import numpy as np
import time
import math
print('ry version:', ry.__version__, ry.compiled())

###############################################################################

class RoboticGym(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    tau_env = .05 #a gym environment step corresponds to 0.05 sec
    tau_sim = .01 #the underlying physics simulation is stepped with 0.01 sec (5 sim steps in each env step)
    time = 0.
    render_mode = 'human'

    viewSteps = False

    def __init__(self, time_limit, withArm, controlMode, random_reset, sim_verbose):
        self.time_limit = time_limit
        self.controlMode = controlMode
        self.random_reset = random_reset

        # create/load the robotic configuration
        self.setup_config(withArm)
        self.q0 = self.C.getJointState()
        self.X0 = self.C.getFrameState()
        n = self.q0.size

        self.ee = self.C.getFrame("plate")
        self.box = self.C.getFrame('box')
        self.ee_p0 = self.ee.getPosition()
        self.box_pos0 = self.box.getPosition()

        # define the observation space (see also observation_fct())
        self.observation_space = gym.spaces.Box(-2., +2., shape=(2*n + 6,), dtype=np.float32)

        # define the action space
        action_scale = 5
        action_dim = (3 if 'ee' in self.controlMode else n)
        self.action_space = gym.spaces.Box(-action_scale, +action_scale, shape=(action_dim,), dtype=np.float32)

        # create the underlying physics simulation (using PhysX from Nvidia)
        self.sim = ry.Simulation(self.C, ry.SimulationEngine.physx, verbose=sim_verbose)

    def __del__(self):
        del self.sim
        del self.C

    def setup_config(self, withArm):
        self.C = ry.Config()
        if withArm:
            self.C.addFile(ry.raiPath('scenarios/pandaSingle.g'))
            self.C.setJointState([.007], ['l_panda_finger_joint1']) #only cosmetics
            #the following makes it a LOT simpler
            self.C.setJointState([.2], ['l_panda_joint1'])
            self.C.setJointState([.0], ['l_panda_joint2'])
            self.C.setJointState([.9], ['l_panda_joint7'])
            gripper = 'l_gripper'
        else:
            self.C.addFile(ry.raiPath('scenarios/ballFinger.g'))
            self.C.addFrame('table', '', 'shape: ssBox, pose: [0 0. .6], size: [2.1, 2.1, .1, .02], color: [.3, .3, .3], friction: .5')
            self.C.getFrame('jointZ').setJoint(ry.JT.hingeZ, limits=[-3,3])
            self.C.getFrame('base').setPosition([.0, .0,1.])
            self.C.setJointState([0.,.3,0.])
            gripper = 'finger'

        box = self.C.addFrame('box') \
            .setShape(ry.ST.ssBox, size=[.1,.1,.1,.005]) .setColor([1,.5,0]) \
            .setPosition([.06,.35,.7]) \
            .setMass(.1) \
            .setAttributes({'friction': .5}) \
            .setContact(1) \
            .setPosition([.3,.2,.7]) # this is the initial position of the box - change manually to make harder

        self.C.addFrame('plate', gripper) \
            .setShape(ry.ST.ssBox, size=[.02,.2,.36,.005]) .setColor([.5,1,0]) \
            .setRelativePosition([0,0,-.16]) \
            .setAttributes({'friction': .1}) \
            .setContact(-1)
        
        self.C.addFrame('gripper', 'plate') \
            .setRelativePosition([0,0,-.15]) \
            .setRelativeQuaternion([1,0,0,1]) \
            .setShape(ry.ST.marker, [.1])

        self.C.addFrame('target') \
            .setShape(ry.ST.marker, size=[.1]) .setColor([0,1,0]) \
            .setPosition([.3,.3,.7])

    def step(self, action):
        '''essential to implement gym.Env'''
        if self.controlMode == 'eeDelta':
            self.step_eeDelta(action)
        elif self.controlMode == 'qDelta':
            self.step_qDelta(action)
        elif self.controlMode == 'qPos':
            self.step_qPos(action)
        else:
            raise Exception(f'unknown control mode: {self.controlMode}')

        self.time += self.tau_env
        
        observation = self.observation_fct()
        reward = self.reward_fct()
        terminated = False
        truncated = (self.time >= self.time_limit) # terminated and truncated difference is super important
        info = {"no": "additional info"}
        return observation, reward, terminated, truncated, info
    
    def observation_fct(self):
        '''observations are joint pos/vel, and box pos/vel'''
        X, q, Xdot, qdot = self.sim.getState()
        box_pos = X[self.box.ID, :3]
        box_vel = Xdot[self.box.ID, 0]
        observation = np.concatenate((q, qdot, box_pos, box_vel), axis=0, dtype=np.float32)
        return observation

    def reward_fct(self):
        '''two reward terms: box-at-goal reward (peaked) and touch object reward (negDistance)'''

        goalDist, _ = self.C.eval(ry.FS.positionDiff, ["box", "target"])
        r_goal = np.exp(-np.linalg.norm(goalDist)/.1)

        touchDist, _ = self.C.eval(ry.FS.negDistance, ["plate", "box"])
        touchDist = np.maximum(0., -touchDist[0])
        r_touch = np.exp(-touchDist/.1)

        r = r_goal + .1*r_touch

        # if self.viewSteps:
            # print(f'rewards goal: {r_goal} touch: {r_touch} total: {r}')

        return r

    def step_qPos(self, q_ref):
        '''sending positions in joint space; where deltas have semantic of velocity'''
        sim_steps = int(self.tau_env/self.tau_sim)
        for s in range(sim_steps):
            self.sim.step(q_ref, self.tau_sim, ry.ControlMode.position)
            if self.viewSteps:
                self.C.view(False)
                time.sleep(0.1*self.tau_sim)

    def step_qDelta(self, delta):
        '''sending deltas in joint space; where deltas have semantic of velocity'''
        # the sim needs to run at least with 100Hz (to properly simulate collisions)
        # that's why we have multiple sim steps for each env step
        sim_steps = int(self.tau_env/self.tau_sim)
        for s in range(sim_steps):
            #the following converts the deltas to position references for the underlying sim PD controller
            q = self.sim.get_q()
            self.sim.step(q + self.tau_sim * delta, self.tau_sim, ry.ControlMode.position)
            if self.viewSteps:
                self.C.view(False)
                time.sleep(0.1*self.tau_sim)

    def step_eeDelta(self, delta):
        '''sending deltas in endeffector space - using just 4 Newton steps (via KOMO) to approx. solve the IK'''
        sim_steps = int(self.tau_env/self.tau_sim)
        delta[-1] *= 5.

        dx, dy, dtheta = self.tau_sim * delta

        komo = ry.KOMO()
        komo.setConfig(self.C, False)
        komo.setTiming(1, 1, 1., 0)

        for s in range(sim_steps):
            # the ee position target
            pos = self.ee.getPosition() + np.array([dx, dy, 0.0])
            pos[2] = self.ee_p0[2] #constrain z to stay constant

            # the ee quaternion target (adding via Lie group log/exp)
            quat = self.ee.getQuaternion()
            w = ry.Quaternion().set(quat).getLog()
            w = [0., 0., w[2]+dtheta] #constrain rotation about x/y to stay zero
            quat = ry.Quaternion().setExp(w).asArr()

            komo.clearObjectives()
            komo.addControlObjective([], 0, 1e-1)
            komo.addObjective([], ry.FS.position, ['plate'], ry.OT.sos, [1e2], pos)
            komo.addObjective([], ry.FS.quaternion, ['plate'], ry.OT.sos, [1e2], quat)
            sol = ry.NLP_Solver(komo.nlp())
            sol.setOptions(stopInners=4, damping=1e-4, verbose=0)
            ret = sol.solve()
            # komo.view(True, f'sol{s}')

            self.sim.step(komo.getPath()[0], self.tau_sim, ry.ControlMode.position)
            if self.viewSteps:
                self.C.view()
                time.sleep(self.tau_sim)

    def reset(self, seed=None, options=None):
        '''essential to implement gym.Env'''
        if seed is not None:
            super().reset(seed=int(seed))
            np.random.seed(seed)

        if self.random_reset:
            # resetting the box position to a random initial position -- makes it MUCH harder
            self.box_pos0 = np.array([.0,-.1,.7]) + .7 * np.random.rand(3)
            self.box_pos0[2]=.7

        self.X0[self.box.ID, :3] = self.box_pos0

        # resetting the sim
        self.time = 0.
        self.sim.resetTime()
        self.sim.setState(self.X0, self.q0)
        self.sim.resetSplineRef()

        observation = self.observation_fct()
        info = {"no": "additional info"}
        return observation, info
        
    def rollout(self, pi, dataset=None, view=True):
        '''helper to play and view a policy'''
        self.viewSteps=view #triggers display within the low level step functions
        obs, info = self.reset()
        t = 0
        R = 0
        while True:
            action = pi(obs, t)
            next_obs, reward, terminated, truncated, info = self.step(action)
            if dataset is not None:
                dataset['observations'].append(obs)
                dataset['actions'].append(action)
                dataset['next_observations'].append(next_obs)
                dataset['rewards'].append(np.array([reward]))
                if terminated or truncated:
                    dataset['terminals'].append(np.array([1.]))
                else:
                    dataset['terminals'].append(np.array([0.]))
            obs = next_obs
            R += reward
            t += 1
            if view:
                print("reward: ", reward)
            if terminated or truncated:
                break
        if view:
            print('total return:', R)
            # self.C.view(True)
        self.viewSteps=False #disables display within the low level step functions

    def render(self):
        '''also part of the env.Gym'''
        self.C.view(False, f'RoboticGym time {self.time} / {self.time_limit}')
        if self.render_mode == "rgb_array":
            return self.C.view_getRgb()