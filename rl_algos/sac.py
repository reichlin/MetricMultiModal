import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from architectures.main_architectures import Actor, Critic
from .rl_utils import ReplayBuffer, hard_update, soft_update
from .rl_base import RL_ALGO

import time



class SAC(RL_ALGO):

    def __init__(self, preprocessor, all_dims, configs_rl, device, lambda_repr=1.0):
        super(SAC, self).__init__(preprocessor, all_dims, configs_rl, device)

        self.preprocessor = preprocessor
        self.rl_algo_name = 'sac'

        self.gamma = configs_rl['sac']['gamma']
        self.tau = configs_rl['sac']['tau']
        self.min_memory = configs_rl['sac']['min_memory']
        size_buffer = configs_rl['sac']['size_buffer']
        init_alpha = configs_rl['sac']['init_alpha']
        # target_entropy = configs_rl['sac']['target_entropy']

        # self.target_entropy_init = configs_rl['sac']['target_entropy_init']
        # self.target_entropy_min = configs_rl['sac']['target_entropy_min']

        self.update_epochs = configs_rl['sac']['update_epochs']
        self.frq_actor_update = configs_rl['sac']['frq_actor_update']
        self.reward_scaling = configs_rl['sac']['reward_scaling']
        self.gradient_clipping_max = configs_rl['sac']['gradient_clipping_max']
        n_ensemble_critics = configs_rl['sac']['n_ensemble_critics']
        self.batch_size = configs_rl['sac']['batch_size']

        self.device = device

        self.replay_buffer = ReplayBuffer(size_buffer, all_dims, device=device)

        self.lambda_repr = lambda_repr
        # if opt_buffer is not None:
        #     for transition in opt_buffer:
        #         self.replay_buffer.buffer.append(transition)

        self.policy = Actor(all_dims['z'], all_dims['a'], configs_rl['architecture'])
        self.critic1 = nn.ModuleList([Critic(all_dims['s'], all_dims['a'], configs_rl['architecture']) for _ in range(n_ensemble_critics)])
        self.critic2 = nn.ModuleList([Critic(all_dims['s'], all_dims['a'], configs_rl['architecture']) for _ in range(n_ensemble_critics)])
        self.critic1_target = nn.ModuleList([Critic(all_dims['s'], all_dims['a'], configs_rl['architecture']) for _ in range(n_ensemble_critics)])
        self.critic2_target = nn.ModuleList([Critic(all_dims['s'], all_dims['a'], configs_rl['architecture']) for _ in range(n_ensemble_critics)])

        hard_update(self.critic1, self.critic1_target)
        hard_update(self.critic2, self.critic2_target)

        assert init_alpha > 1e-6, "init alpha needs to be positive"
        self.log_alpha = nn.Parameter(torch.log(torch.tensor([init_alpha], dtype=torch.float)), requires_grad=True)

        self.alpha_opt = optim.Adam([self.log_alpha], lr=configs_rl['sac']['lr_alpha'])
        self.actor_and_alpha_optimizer = optim.Adam(list(self.policy.parameters()) +
                                                    #[self.log_alpha] +
                                                    list(self.preprocessor.parameters()), lr=configs_rl['sac']['lr_actor'])
        self.critic_optimizer = optim.Adam(list(self.critic1.parameters())+list(self.critic2.parameters()), lr=configs_rl['sac']['lr_critic'])

        self.target_entropy = - all_dims['a'] #/ 2 #self.target_entropy_init

        self.mse = nn.MSELoss()

        self.critic_update_count = 0

    # def load_opt_buffer(self):
    #     for transition in self.opt_buffer:
    #         self.replay_buffer.add(transition)
        # return
        # if self.opt_buffer is not None:
        #     for transition in self.opt_buffer:
        #         self.replay_buffer.buffer.append(transition)


    def get_critics_loss(self, batch_data):

        z, next_z, states, actions, rewards, next_states, dones = batch_data

        if self.reward_scaling:
            with torch.no_grad():
                reward_scale = 1 / (1 + torch.std(rewards))  # Adaptive scaling
                rewards = rewards * reward_scale

        with torch.no_grad():

            next_actions, next_other_info = self.policy.sample(next_z)
            next_log_probs = next_other_info[0]

            q1_next = torch.stack([q(next_states, next_actions) for q in self.critic1_target], 0).mean(0)
            q2_next = torch.stack([q(next_states, next_actions) for q in self.critic2_target], 0).mean(0)
            q_next = torch.min(q1_next, q2_next) - self.log_alpha.exp().detach() * next_log_probs

            q_target = rewards + self.gamma * (1 - dones) * q_next


        q1 = torch.stack([q(states, actions) for q in self.critic1], 0)
        critic1_loss = self.mse(q1, q_target.expand_as(q1)).mean()
        q2 = torch.stack([q(states, actions) for q in self.critic2], 0)
        critic2_loss = self.mse(q2, q_target.expand_as(q2)).mean()

        self.critic_update_count += 1

        return critic1_loss, critic2_loss

    def get_actor_and_alpha_loss(self, batch_data):

        z, next_z, states, actions, rewards, next_states, dones = batch_data

        new_actions, other_info = self.policy.sample(z)
        log_probs = other_info[0]

        q1_pi = torch.stack([q(states, new_actions) for q in self.critic1], 0).mean(0)
        q2_pi = torch.stack([q(states, new_actions) for q in self.critic2], 0).mean(0)

        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.log_alpha.exp().detach() * log_probs - q_pi).mean()

        # alpha_loss = (-self.log_alpha.exp() * (log_probs + self.target_entropy).detach()).mean()
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        return actor_loss, alpha_loss

    def update(self):#, batch_data, additional_loss=None):

        if len(self.replay_buffer) < self.min_memory:
            return None

        for _ in range(2):
            (states,
             images,
             depths,
             pointclouds,
             pc_rgb,
             sound,
             actions,
             rewards,
             next_states,
             next_images,
             next_depths,
             next_pointclouds,
             next_pc_rgb,
             next_sound,
             dones,
             logprobs) = self.replay_buffer.sample(self.batch_size)

            obs = {"state": states,
                   "image": images,
                   "depth": depths,
                   "pointcloud": {"pc": pointclouds, "pc_rgb": pc_rgb},
                   "sound": sound,
                   }

            next_obs = {"state": next_states,
                        "image": next_images,
                        "depth": next_depths,
                        "pointcloud": {"pc": next_pointclouds, "pc_rgb": next_pc_rgb},
                        "sound": next_sound,
                        }

            # obs = self.preprocessor.preprocess({"state": states,
            #                                     "image": images,
            #                                     "depth": depths,
            #                                     "pointcloud": pointclouds,
            #                                     "pc_rgb": pc_rgb})
            # next_obs = self.preprocessor.preprocess({"state": next_states,
            #                                          "image": next_images,
            #                                          "depth": next_depths,
            #                                          "pointcloud": next_pointclouds,
            #                                          "pc_rgb": next_pc_rgb})
            # states = torch.from_numpy(states).float().to(self.device)
            # next_states = torch.from_numpy(next_states).float().to(self.device)
            # actions = torch.from_numpy(actions).float().to(self.device)
            # rewards = torch.from_numpy(rewards).float().to(self.device)
            # dones = torch.from_numpy(dones).float().to(self.device)

            loss_representation, z, next_z = self.preprocessor.get_loss(obs, actions, next_obs)

            batch_data = (z.detach(), next_z.detach(), states, actions, rewards, next_states, dones)

            critic1_loss, critic2_loss = self.get_critics_loss(batch_data)
            critic_loss = critic1_loss + critic2_loss

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.gradient_clipping_max > 0:
                torch.nn.utils.clip_grad_norm_(list(self.critic1.parameters())+list(self.critic2.parameters()), self.gradient_clipping_max)
            self.critic_optimizer.step()

            actor_loss = None
            if self.critic_update_count % 2 == 0:

                batch_data = (z, next_z, states, actions, rewards, next_states, dones)

                actor_loss, alpha_loss = self.get_actor_and_alpha_loss(batch_data)

                actor_loss = actor_loss + self.lambda_repr * loss_representation # 0.1

                self.actor_and_alpha_optimizer.zero_grad()
                (actor_loss).backward()  # +alpha_loss
                self.actor_and_alpha_optimizer.step()

                self.alpha_opt.zero_grad()
                (alpha_loss).backward()
                self.alpha_opt.step()

            with torch.no_grad():
                self.log_alpha.clamp_(-10, 2)

        self.cleanup()

        logs = {
            "critic1_loss": critic1_loss.detach().cpu().item(),
            "critic2_loss": critic2_loss.detach().cpu().item(),
        }
        if actor_loss is not None:
            logs["repr_loss"] = loss_representation.detach().cpu().item()
            logs["actor_loss"] = actor_loss.detach().cpu().item()
            logs["alpha_loss"] = alpha_loss.detach().cpu().item()

        return logs

    def cleanup(self):
        soft_update(self.critic1, self.critic1_target, self.tau)
        soft_update(self.critic2, self.critic2_target, self.tau)
        self.preprocessor.cleanup()












