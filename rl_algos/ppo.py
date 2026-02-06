import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from architectures.main_architectures import ActorPPO, Critic
from .rl_utils import BatchData, hard_update, soft_update, calc_rtg, compute_gae_adv
from .rl_base import RL_ALGO


class PPO(RL_ALGO):

    def __init__(self, preprocessor, all_dims, configs_rl, device):
        super(PPO, self).__init__(preprocessor, all_dims, configs_rl, device)

        self.preprocessor = preprocessor
        self.rl_algo_name = 'ppo'

        self.device = device

        self.gamma = configs_rl['ppo']['gamma']
        self.lamb = configs_rl['ppo']['lambda']
        self.K_epochs = configs_rl['ppo']['K_epochs']
        self.norm_A = configs_rl['ppo']['norm_A']
        self.eps_clip = configs_rl['ppo']['eps_clip']
        self.c1 = configs_rl['ppo']['c1']
        self.c2 = configs_rl['ppo']['c2']
        self.c2_schedule = configs_rl['ppo']['c2_schedule']
        self.clip_grads = configs_rl['ppo']['clip_grads']
        learn_var = configs_rl['ppo']['learn_var']
        init_var = configs_rl['ppo']['init_var']
        self.min_memory = configs_rl['ppo']['min_memory']
        self.batch_size = configs_rl['ppo']['batch_size']

        self.replay_buffer = BatchData(self.gamma)

        self.policy = ActorPPO(all_dims['z'], all_dims['a'], configs_rl['architecture'], learn_var, init_var)
        self.critic = Critic(all_dims['s'], action_dim=0, architecture_params=configs_rl['architecture'])
        self.old_policy = ActorPPO(all_dims['z'], all_dims['a'], configs_rl['architecture'], learn_var, init_var)

        hard_update(self.policy, self.old_policy)

        # self.policy_optimizer = optim.Adam(list(self.policy.parameters()) +
        #                                    list(self.preprocessor.parameters()), lr=configs_rl['ppo']['lr_actor'], eps=1e-5) # + list(self.critic.parameters())
        # self.critic_optimizer = optim.Adam(list(self.critic.parameters()), lr=configs_rl['ppo']['lr_critic'], eps=1e-5)

        self.preprocessor_optimizer = optim.Adam(list(self.preprocessor.parameters()), lr=configs_rl['ppo']['lr_actor'], eps=1e-5)
        self.policy_optimizer = optim.Adam(list(self.policy.parameters()), lr=configs_rl['ppo']['lr_actor'], eps=1e-5) # + list(self.critic.parameters())
        self.critic_optimizer = optim.Adam(list(self.critic.parameters()), lr=configs_rl['ppo']['lr_critic'], eps=1e-5)

        self.mse = nn.MSELoss()

        self.critic_update_count = 0


    def get_action(self, z, test=False, state=None):
        a, other_info = self.policy.sample(z, test=test)
        if state is not None:
            s = torch.from_numpy(state).float().to(self.device)
            v = self.critic(s).detach().cpu().numpy()
            return a, other_info + (v,)
        return a, other_info

    def update(self):

        (states,
         images,
         depths,
         actions,
         raw_actions,
         rewards,
         next_states,
         next_images,
         next_depths,
         dones,
         logprobs,
         values) = self.replay_buffer.sample()

        with torch.no_grad():

            last_next_states = torch.from_numpy(next_states[-1]).float().to(self.device)
            last_next_value = self.critic(last_next_states).detach().cpu().numpy()
            values = np.concatenate([values, np.expand_dims(last_next_value, 0)], 0).squeeze()
            Adv_tot, rtgs_tot = compute_gae_adv(rewards, dones, values, self.gamma, self.lamb)
            Adv_tot = torch.from_numpy(Adv_tot).float().to(self.device).reshape((-1, *Adv_tot.shape[2:]))
            rtgs_tot = torch.from_numpy(rtgs_tot).float().to(self.device).reshape((-1, *rtgs_tot.shape[2:]))

        for i in range(self.K_epochs):

            all_idx = np.arange(len(self.replay_buffer))
            np.random.shuffle(all_idx)

            for i_s in range(0, len(self.replay_buffer), self.batch_size):

                # rnd_idx = np.random.choice(all_idx, self.batch_size)
                rnd_idx = all_idx[i_s:(i_s+self.batch_size)]

                Adv = Adv_tot[rnd_idx]
                rtgs = rtgs_tot[rnd_idx]

                st = torch.from_numpy(states).float().to(self.device).reshape((-1, *states.shape[2:]))[rnd_idx]

                obs_dict, next_obs_dict = {}, {}
                if states is not None:
                    obs_dict["state"] = states.reshape((-1, *states.shape[2:]))[rnd_idx]
                    next_obs_dict["state"] = next_states.reshape((-1, *next_states.shape[2:]))[rnd_idx]
                if images is not None:
                    obs_dict["image"] = images.reshape((-1, *images.shape[2:]))[rnd_idx]
                    next_obs_dict["image"] = next_images.reshape((-1, *next_images.shape[2:]))[rnd_idx]
                if depths is not None:
                    obs_dict["depth"] = depths.reshape((-1, *depths.shape[2:]))[rnd_idx]
                    next_obs_dict["depth"] = next_depths.reshape((-1, *next_depths.shape[2:]))[rnd_idx]
                # TODO: add pointclouds


                obs = self.preprocessor.preprocess(obs_dict)
                next_obs = self.preprocessor.preprocess(next_obs_dict)

                old_actions = torch.from_numpy(actions).float().to(self.device).reshape((-1, *actions.shape[2:]))[rnd_idx]
                old_raw_actions = torch.from_numpy(raw_actions).float().to(self.device).reshape((-1, *raw_actions.shape[2:]))[rnd_idx]
                old_logprobs = torch.from_numpy(logprobs).float().to(self.device).reshape((-1, *logprobs.shape[2:]))[rnd_idx]

                loss_representation, z, next_z = self.preprocessor.get_loss(obs, old_actions, next_obs)
                actor_loss, v_loss, entropy_loss = self.get_rl_losses(z.detach(), old_raw_actions, old_logprobs, Adv, rtgs, st)

                # TODO: this is not really correct? it still changes z?

                critic_loss = self.c1 * v_loss
                policy_loss = actor_loss - self.c2 * entropy_loss  # loss_representation +

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                if self.clip_grads:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                if self.clip_grads:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.policy_optimizer.step()

        self.preprocessor_optimizer.zero_grad()
        loss_representation.backward()
        self.preprocessor_optimizer.step()

        self.cleanup()

        logs = {
            "repr_loss": loss_representation.detach().cpu().item(),
            "critic_loss": v_loss.detach().cpu().item(),
            "entropy_loss": entropy_loss.detach().cpu().item(),
            "actor_loss": actor_loss.detach().cpu().item()
        }

        return logs

    def get_rl_losses(self, z, old_raw_actions, old_logprobs, Adv, rtgs, st):

        logprobs, dist_entropy = self.policy.evaluate(z, old_raw_actions)
        values = self.critic(st).squeeze(-1)

        # Importance ratio
        ratios = torch.exp(logprobs - old_logprobs.detach()).squeeze()  # new probs over old probs

        if self.norm_A:
            A_norm = (Adv - Adv.mean()) / (Adv.std() + 1e-8)
        else:
            A_norm = Adv

        surr1 = ratios * A_norm
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * A_norm
        actor_loss = torch.mean(-torch.min(surr1, surr2))  # minus to maximize

        return actor_loss, self.mse(rtgs, values), torch.mean(dist_entropy)



    def cleanup(self):
        # Replace old policy with new policy
        hard_update(self.policy, self.old_policy)
        self.c2 *= self.c2_schedule
        self.replay_buffer.clear()
        self.preprocessor.cleanup()










