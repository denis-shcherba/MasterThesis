from scripts.dataset.komo_dataset import *
from scripts.utils import *
from env import *
from scripts.archiv.train_v_field import *
import env.komo.Lift as Lift

import numpy as np
import robotic as ry
import time 

class field_augmentation():
    data_path = 'komo_data/Lift/eeDelta_global_2_200.npy'
    def __init__(self, config, dataset:field_augment_dataset):
        self.config = config
        self.raw_dataset_path = config.train.dataset
        self.raw_dataset = list( np.load(self.raw_dataset_path, allow_pickle=True))
        self.dataset = dataset
        self.data_len = len(self.dataset)

        self.env = LiftEnv(time_limit=5., withArm=True, controlMode='eeDelta_global', sim_verbose=0)
        
    
    def traj_ref(self, traj):
        self.env.C.getFrame('box').setPosition(traj[:, 0, :3])
        self.env.C.getFrame('box').setQuaternion(ry.Quaternion().setExp(traj[:,0, 6:9].flatten()).asArr())

        komo = ry.KOMO()
        komo.setConfig(self.env.C, False)
        komo.setTiming(1, 1, 1., 0)
        for i in range(traj.shape[1]):
            pos = traj[0, i, 9:12]
            quat = ry.Quaternion().setExp(traj[0, i, 12:-1]).asArr()

            komo.clearObjectives()
            komo.addControlObjective([], 0, 1e-1)
            komo.addObjective([], ry.FS.position, ['l_gripper'], ry.OT.sos, [1e2], pos)
            komo.addObjective([], ry.FS.quaternion, ['l_gripper'], ry.OT.sos, [1e2], quat)
            sol = ry.NLP_Solver(komo.nlp())
            sol.setOptions(stopInners=4, damping=1e-4, verbose=0)
            ret = sol.solve()
            path = komo.getPath()[0]
            self.env.C.setJointState(path)
            self.env.C.view(False, f'reference trajectory')
            time.sleep(0.001)



    def augment_data(self, episodes=100, verbose=0, append=False):

        dataset = []

        for i in range(0, (self.dataset.episode_start.shape[0]), 10):
            x = torch.linspace(-0.3,0.3,3)
            y = torch.linspace(0, 0.2, 3)
            z = torch.linspace(0.675, 0.85, 3)
            x,y,z = torch.meshgrid(x,y,z)
            start_points = torch.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

            # episode_start = self.raw_dataset[-1]['episode_start'][0] + len(self.raw_dataset[-1]['actions'])
            episode_start = 0
            count=0
            for start_point in range(start_points.shape[0]):
                # empty demonstration
                demo = { 'observations': {}, 'actions': [], 'next_observations':{}, 'rewards':[], 'terminals':[],
                    'episode_seed':[], 'episode_start':[], 'phase': [] }
                

                # trajectory from dataset
                # idx = np.random.randint(0, self.data_len)
                # datasample = self.dataset[idx]
                # action_seq = datasample['action'].unsqueeze(0)
                # obs = datasample['obs'].reshape(1, -1)
                # traj = datasample['traj'].unsqueeze(0)
                # traj_phase2 = self.dataset.traj_rest
                # action_phase2 = self.dataset.action_rest
                # just check the first phas
                # if datasample['phase'] == 1:
                #     continue

                # go through all trajectories
                start, end = self.dataset.episode_start[i], self.dataset.episode_end[i]
                action_seq = self.dataset.action[start: end]
                obs = self.dataset.obs[start].reshape(1,-1)
                traj = self.dataset.obs[start: end]
                phase = self.dataset.phase[start: end]
                mask = (phase == 0).flatten()
                action_phase2 = action_seq[~mask]
                traj_phase2 = traj[~mask]
                action_seq = torch.from_numpy(action_seq[mask]).float().unsqueeze(0)
                traj = torch.from_numpy(traj[mask]).float().unsqueeze(0)
                obs = torch.from_numpy(obs).float().reshape(1,-1)
                    
                self.env.reset()

                # set initial state
                start_point = start_points[start_point, :]   #(1,3)
                obs[0, 9:12 ] = start_point
                new_joint_angle = solve_IK(self.env.C, obs[0,9:-1])

                if verbose > 0:
                # reference trajectory
                    self.traj_ref(traj)
                    self.env.C.view(True, 'state check')

                X, q, Xdot, qdot = self.env.sim.getState()
                box = self.env.C.getFrame('box')
                self.env.box_pos0 = obs[:,:3]
                self.env.box_rot0 = ry.Quaternion().setExp(obs[:,6:9].flatten()).asArr()
                self.env.q = new_joint_angle
                self.env.random_reset = False
                self.env.reset()
                # X[box.ID, :3] = obs[:, :3]
                # self.env.sim.setState(X, new_joint_angle)

                done = False
                self.counter = -1
                self.phase = 0
                def pi(obs_dict, t):

                    if self.phase == 1:
                        demo['phase'].append(1)
                        self.counter += 1

                        if self.counter >= len(action_phase2):
                            return action_phase2[-1]
                        
                        action_phase2[self.counter, -1] = 0.015
                        if self.env.sim.getGripperWidth('l_gripper')>0.025:
                            action = np.zeros([7])
                            action[-1] = 0.02
                            return action
                        
                        return action_phase2[self.counter]
                    
                    demo['phase'].append(0)
                    state = []
                    for key in self.config.env.obs_keys:
                            if key == 'gripper':
                                state = np.append(state, np.array(obs_dict['joint_angle'])[-1])
                                break
                            state = np.append(state, np.array(obs_dict[key]))
                    query_point = torch.from_numpy(state).float()
                    query_point =  query_point.reshape(1, -1)
                    query_point_robot = query_point[:, 9:]
                    query_point_trans = query_point[:, 9:12]
                    traj_robot = traj[:,:, 9:]
                    B, L, obs_dim_robot = traj.shape

                    # search the nearest point
                    p0 = traj_robot[:, :-1, :3]    # (B, L-1, 3)
                    p1 = traj_robot[:, 1:, :3]
                    v_tangent = p1 - p0            # (B, L-1, 3)
                    v_converge = query_point_trans[:, None, :] - p0
                    v_norm_sq = v_tangent.square().sum(-1, keepdim=True) + 1e-6          # (B, L-1, 1)
                    t = (v_converge * v_tangent).sum(-1, keepdim= True) / v_norm_sq  # (B, L-1, 1)
                    t = t.clamp(0,1)
                    x_proj = p0 + t * v_tangent # (B, L-1, 3)
                    dist = (query_point_trans[:, None, :] - x_proj).square().sum(-1, keepdim=True)
                    nearest_idx = dist.argmin(dim=1) 


                    # trajectory velocity shape (B, L, Dim)
                    x_proj_nearest = x_proj[torch.arange(B), nearest_idx] # (B, 3)
                    traj_trans_velocity = action_seq[torch.arange(B), nearest_idx, :3]
                    traj_rot_velocity = action_seq[torch.arange(B), nearest_idx, 3:-1] 
                    traj_g_velocity = action_seq[torch.arange(B), nearest_idx, -1] 

                    # backwards
                    v_back_nearest_t = query_point_robot[torch.arange(B), :3] - x_proj_nearest
                    progress = nearest_idx / L
                    backwards_steps = 25*progress
                    backwards_steps.to(torch.int)

                    nearest_idx = np.clip(nearest_idx-backwards_steps, 0, L).to(torch.int)
                    x_proj_nearest = x_proj[torch.arange(B), nearest_idx] # (B, 3)

                    # velocity field converge to trajectory
                    R_nearest = pytorch3d.transforms.axis_angle_to_matrix(traj_robot[torch.arange(B), nearest_idx, 3:-1])
                    R_to_traj = R_nearest.matmul(pytorch3d.transforms.axis_angle_to_matrix(query_point_robot[:, 3:-1]).transpose(-1,-2))
                    v_to_traj_rot = pytorch3d.transforms.matrix_to_axis_angle(R_to_traj)
                    v_to_traj_t = (query_point_robot[torch.arange(B), :3] - x_proj_nearest)
                    v_to_traj_g = (query_point_robot[torch.arange(B), -1] - traj_robot[torch.arange(B), nearest_idx, -1]).unsqueeze(1)
                    v_conv = torch.concatenate([v_to_traj_t, v_to_traj_rot, v_to_traj_g], dim=-1).reshape(-1, 7)

                    # add the backwards term
                    alpha = 1 / (1+5*torch.norm(v_back_nearest_t, dim=-1, keepdim=True))
                    # alpha = 1.
                    v_to_traj_t = (1-alpha) * v_to_traj_t + alpha * v_back_nearest_t

                    # total field TODO: 
                    
                
                    beta_t = 1/(1 + 10* torch.norm(v_to_traj_t, dim=-1, keepdim=True))
                    beta_r = 1/(1 + 10* torch.norm(v_to_traj_rot, dim=-1, keepdim=True))
                    v_rot_total = beta_r* traj_rot_velocity + (1-beta_r) * v_to_traj_rot
                    v_trans_total = beta_t*traj_trans_velocity - (1-beta_t) * v_to_traj_t 

                    v_norm_mean = torch.max(torch.norm(action_seq, dim=-1),dim=-1)
                    scale =   v_norm_mean.values / torch.norm(v_trans_total, dim=-1)
                    scale = torch.clamp(scale, 0, 1)
                    v_trans_total = v_trans_total * scale
                
                    velocity = torch.cat([v_trans_total, v_rot_total, traj_g_velocity.reshape(-1, 1, 1)], dim=-1)
                    
                    if torch.abs(query_point[:, 9:12] - traj[:,-1,:3]).max()<1e-2 and self.phase==0:
                        self.phase = 1
                    
                    return velocity.numpy().flatten()
                    
                self.env.rollout(pi, demo, view=verbose>0)

                demo['episode_start'].append(episode_start)

                if self.env.is_success():
                    dataset.append(demo)
                    episode_start += len(demo['actions'])
                    count = count+1
                    print(count, end='\r', flush=True)


        if append is True:
            newdata = self.raw_dataset + dataset
        np.save('Dataset/komo_data/Lift/ee_global_grasp_DA_step10.npy', newdata, allow_pickle=True)

        
if __name__ == '__main__':
    config = load_config('config/komo/lift_low_dim_v_field.yaml')
    dataset = field_augment_dataset(
                config.train.dataset,
                obs_keys=config.env.obs_keys,
                action_key=config.env.control_mode,
                normalize_action=config.train.normalize_action,
                normalize_obs=config.train.normalize_obs
            )

    v_aug = field_augmentation(config, dataset)

    v_aug.augment_data(episodes=100, verbose=0, append=True)