import numpy as np
import random
from collections import deque
import torch



class BatchData:
    def __init__(self, gamma):

        self.gamma = gamma

        self.states = []
        self.actions = []
        self.raw_actions = []
        self.logprobs = []
        self.rewards = []
        self.next_states = []
        self.is_terminal = []
        self.values = []

    def add(self, transition):

        state, action, reward, next_state, done, action_raw, logprobs, value = transition

        self.states.append(state)
        self.actions.append(action)
        self.raw_actions.append(action_raw)
        self.logprobs.append(logprobs)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.is_terminal.append(done)
        self.values.append(value)

    def sample(self):

        s_t = np.stack([x['state'] for x in self.states], 0) if 'state' in self.states[0].keys() else None
        img_t = np.stack([x['image'] for x in self.states], 0) if 'image' in self.states[0].keys() else None
        depth_t = np.stack([x['depth'] for x in self.states], 0) if 'depth' in self.states[0].keys() else None
        a_t = np.stack(self.actions, 0)
        raw_a_t = np.stack(self.raw_actions, 0)
        r_t = np.stack(self.rewards, 0)
        s_t1 = np.stack([x['state'] for x in self.next_states], 0) if 'state' in self.next_states[0].keys() else None
        img_t1 = np.stack([x['image'] for x in self.next_states], 0) if 'image' in self.next_states[0].keys() else None
        depth_t1 = np.stack([x['depth'] for x in self.next_states], 0) if 'depth' in self.next_states[0].keys() else None
        done_t = np.stack(self.is_terminal, 0)
        logprobs = np.stack(self.logprobs, 0)
        values = np.stack(self.values, 0)

        return (np.array(s_t, dtype=np.float32) if s_t is not None else None,
                np.array(img_t, dtype=np.uint8) if img_t is not None else None,
                np.array(depth_t, dtype=np.uint8) if depth_t is not None else None,
                np.array(a_t, dtype=np.float32),
                np.array(raw_a_t, dtype=np.float32),
                np.array(r_t, dtype=np.float32),
                np.array(s_t1, dtype=np.float32) if s_t1 is not None else None,
                np.array(img_t1, dtype=np.uint8) if img_t1 is not None else None,
                np.array(depth_t1, dtype=np.uint8) if depth_t1 is not None else None,
                np.array(done_t, dtype=np.float32),
                np.array(logprobs, dtype=np.float32),
                np.array(values, dtype=np.float32))

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.raw_actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.is_terminal.clear()
        self.values.clear()

    def __len__(self):
        return len(self.actions) * self.actions[0].shape[0]


def compute_gae_adv(rewards, is_terminals, values, gamma, lamb):

    assert len(rewards) == len(is_terminals)

    deltas = rewards + gamma * (1 - is_terminals) * values[1:] - values[:-1]
    advantages = np.zeros_like(rewards)
    gae = np.zeros(rewards.shape[-1])
    for t in reversed(range(rewards.shape[0])):
        gae = deltas[t] + gamma * lamb * (1 - is_terminals[t]) * gae
        advantages[t] = gae
    return advantages, advantages+values[:-1]


def calc_rtg(rewards, is_terminals, gamma, last_next_value):

    assert len(rewards) == len(is_terminals)
    rtgs = []
    discounted_reward = np.zeros(rewards.shape[-1])
    for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
        # if is_terminal:
        #     discounted_reward = 0
        discounted_reward *= (1 - is_terminal)
        discounted_reward = reward + gamma * discounted_reward
        rtgs.insert(0, discounted_reward)

    return np.stack(rtgs, 0) #np.concatenate(rtgs, 0)


class ReplayBuffer:
    def __init__(self, max_size=1000000, all_dims=None, device='cpu'):

        self.N = max_size
        self.ptr = 0; self.size = 0
        self.device = device
        self.s       = torch.from_numpy(np.empty((self.N, all_dims['s']), np.float32)) if 's' in all_dims else None
        self.s1      = torch.from_numpy(np.empty((self.N, all_dims['s']), np.float32)) if 's' in all_dims else None
        self.o       = torch.from_numpy(np.empty((self.N, all_dims['time']*3, all_dims['o'], all_dims['o']), np.uint8)) if 'o' in all_dims else None
        self.o1      = torch.from_numpy(np.empty((self.N, all_dims['time']*3, all_dims['o'], all_dims['o']), np.uint8)) if 'o' in all_dims else None
        self.d       = torch.from_numpy(np.empty((self.N, all_dims['time']*1, all_dims['d'], all_dims['d']), np.uint8)) if 'd' in all_dims else None
        self.d1      = torch.from_numpy(np.empty((self.N, all_dims['time']*1, all_dims['d'], all_dims['d']), np.uint8)) if 'd' in all_dims else None
        self.pc      = torch.from_numpy(np.empty((self.N, all_dims['time'], all_dims['p'], 3), np.float32)) if 'p' in all_dims else None
        self.pc1     = torch.from_numpy(np.empty((self.N, all_dims['time'], all_dims['p'], 3), np.float32)) if 'p' in all_dims else None
        self.pc_rgb  = torch.from_numpy(np.empty((self.N, all_dims['time'], all_dims['p'], 3), np.uint8)) if 'p' in all_dims else None
        self.pc_rgb1 = torch.from_numpy(np.empty((self.N, all_dims['time'], all_dims['p'], 3), np.uint8)) if 'p' in all_dims else None
        self.sound   = torch.from_numpy(np.empty((self.N, all_dims['sound']), np.float32)) if 'sound' in all_dims else None
        self.sound1 = torch.from_numpy(np.empty((self.N, all_dims['sound']), np.float32)) if 'sound' in all_dims else None
        self.a       = torch.from_numpy(np.empty((self.N, all_dims['a']), np.float32)) if 'a' in all_dims else None
        self.r       = torch.from_numpy(np.empty((self.N, 1), np.float32))
        self.done    = torch.from_numpy(np.empty((self.N, 1), np.float32))

    def __len__(self):
        return self.size #len(self.buffer)

    def add(self, transition):

        obs, action, reward, next_obs, done = transition[:5]

        if action.ndim == 1:
            if 'state' in obs.keys():
                self.s[self.ptr] = obs['state']
                self.s1[self.ptr] = next_obs['state']
            if 'image' in obs.keys():
                self.o[self.ptr] = obs['image'] * 255 if obs['image'].max() < 2.0 else obs['image']
                self.o1[self.ptr] = next_obs['image'] * 255 if next_obs['image'].max() < 2.0 else next_obs['image']
            if 'depth' in obs.keys():
                self.d[self.ptr] = obs['depth'] * 255 if obs['depth'].max() < 2.0 else obs['depth']
                self.d1[self.ptr] = next_obs['depth'] * 255 if next_obs['depth'].max() < 2.0 else next_obs['depth']
            if 'pointcloud' in obs.keys():
                self.pc[self.ptr] = obs['pointcloud']['pc']
                self.pc1[self.ptr] = next_obs['pointcloud']['pc']
                self.pc_rgb[self.ptr] = obs['pointcloud']['pc_rgb'] * 255 if obs['pointcloud']['pc_rgb'].max() < 2.0 else obs['pointcloud']['pc_rgb']
                self.pc_rgb1[self.ptr] = next_obs['pointcloud']['pc_rgb'] * 255 if next_obs['pointcloud']['pc_rgb'].max() < 2.0 else next_obs['pointcloud']['pc_rgb']
            if 'sound' in obs.keys():
                self.sound[self.ptr] = obs['sound']
                self.sound1[self.ptr] = next_obs['sound']

            self.a[self.ptr] = torch.from_numpy(action)
            self.r[self.ptr] = torch.from_numpy(reward[None]).float().view(1)
            self.done[self.ptr] = torch.from_numpy(done[None]).float().view(1)

            self.ptr = (self.ptr + 1) % self.N
            if self.size < self.N:
                self.size += 1

        else:

            for i in range(action.shape[0]):

                if 'state' in obs.keys():
                    self.s[self.ptr] = obs['state'][i]
                    self.s1[self.ptr] = next_obs['state'][i]
                if 'image' in obs.keys():
                    self.o[self.ptr] = obs['image'][i]*255 if obs['image'][i].max() < 2.0 else obs['image'][i]
                    self.o1[self.ptr] = next_obs['image'][i]*255 if next_obs['image'][i].max() < 2.0 else next_obs['image'][i]
                if 'depth' in obs.keys():
                    self.d[self.ptr] = obs['depth'][i]*255 if obs['depth'][i].max() < 2.0 else obs['depth'][i]
                    self.d1[self.ptr] = next_obs['depth'][i]*255 if next_obs['depth'][i].max() < 2.0 else next_obs['depth'][i]
                if 'pointcloud' in obs.keys():
                    self.pc[self.ptr] = obs['pointcloud']['pc'][i]
                    self.pc1[self.ptr] = next_obs['pointcloud']['pc'][i]
                    self.pc_rgb[self.ptr] = obs['pointcloud']['pc_rgb'][i]*255 if obs['pointcloud']['pc_rgb'][i].max() < 2.0 else obs['pointcloud']['pc_rgb'][i]
                    self.pc_rgb1[self.ptr] = next_obs['pointcloud']['pc_rgb'][i]*255 if next_obs['pointcloud']['pc_rgb'][i].max() < 2.0 else next_obs['pointcloud']['pc_rgb'][i]
                if 'sound' in obs.keys():
                    self.sound[self.ptr] = obs['sound'][i]
                    self.sound1[self.ptr] = next_obs['sound'][i]

                self.a[self.ptr]    = torch.from_numpy(action[i])
                self.r[self.ptr]    = torch.from_numpy(reward[i:i+1]).float().view(1)
                self.done[self.ptr] = torch.from_numpy(done[i:i+1]).float().view(1)

                self.ptr = (self.ptr + 1) % self.N
                if self.size < self.N:
                    self.size += 1

    def sample(self, batch_size=64):

        idx = np.random.randint(0, self.size, size=batch_size, dtype=np.int64)

        return (self.s[idx].to(self.device) if self.s is not None else None,
                self.o[idx].to(self.device).float().div_(255) if self.o is not None else None,
                self.d[idx].to(self.device).float().div_(255) if self.d is not None else None,
                self.pc[idx].to(self.device) if self.pc is not None else None,
                self.pc_rgb[idx].to(self.device).float().div_(255) if self.pc_rgb is not None else None,
                self.sound[idx].to(self.device) if self.sound is not None else None,
                self.a[idx].to(self.device),
                self.r[idx].to(self.device),
                self.s1[idx].to(self.device) if self.s1 is not None else None,
                self.o1[idx].to(self.device).float().div_(255) if self.o1 is not None else None,
                self.d1[idx].to(self.device).float().div_(255) if self.d1 is not None else None,
                self.pc1[idx].to(self.device) if self.pc1 is not None else None,
                self.pc_rgb1[idx].to(self.device).float().div_(255) if self.pc_rgb1 is not None else None,
                self.sound1[idx].to(self.device) if self.sound is not None else None,
                self.done[idx].to(self.device),
                None)


def soft_update(nets, nets_target, tau=0.005):
    with torch.no_grad():
        if type(nets) == torch.nn.ModuleList:
            for net_i, net_target_i in zip(nets, nets_target):
                for param, target_param in zip(net_i.parameters(), net_target_i.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        else:
            for param, target_param in zip(nets.parameters(), nets_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def hard_update(nets, nets_target):
    with torch.no_grad():
        if type(nets) == torch.nn.ModuleList:
            for net_i, net_target_i in zip(nets, nets_target):
                net_target_i.load_state_dict(net_i.state_dict())
        else:
            nets_target.load_state_dict(nets.state_dict())





