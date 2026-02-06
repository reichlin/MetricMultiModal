import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from PIL import Image
import yaml
import gymnasium as gym
import cv2
from envs.noises import *
from envs.stochastic_pendulum import make_stochastic_pendulum
from preprocessors.mmm import MMM
from rl_algos.sac import SAC

from utils.loops import test


stoch = 5.0

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

with open('configs/rl.yml', 'r') as file:
    configs_rl = yaml.safe_load(file)

all_noises = [gaussian_noise, salt_and_pepper_noise, patches_noise, puzzle_noise, sensor_failure, texture_noise, hallucination_noise]
env = make_stochastic_pendulum(render_mode=["image", "sound"], noises=all_noises, p_noise=0.0, stoch=stoch)()

state_dim = env.observation_space.spaces['state'].shape[0]
action_dim = env.action_space.shape[0]
img_dim = env.observation_space['image'].shape[-1] if 'image' in env._render_mode else 0
action_bounds = np.stack([env.action_space.low, env.action_space.high], 0)
max_T = env.env.spec.max_episode_steps

configs_rl['architecture']['action_bounds'] = torch.from_numpy(action_bounds).float().to(device)

preprocessor = MMM(state_dim, img_dim, 2, action_dim, ["image", "sound"], configs_rl, device).to(device)
all_dims = {
    's': state_dim,
    'o': img_dim,
    'sound': 2,
    'a': action_dim,
    'z': preprocessor.z_dim,
    'time': env._n_frames,
}
agent = SAC(preprocessor, all_dims, configs_rl, device).to(device)

file_name = 'StochasticPendulum' + "_" + str(3) + "_" + "MMM" + "_train" + "_seed=" + str(seed) + "_stoch=" + str(stoch)
agent.load("pendulum", file_name)

''' TEST NOISE ROBUSTNESS PERFORMANCES '''
# print()
#
# all_seeds = [0, 1, 2]
# all_stoch = [0.0, 0.5, 5.0, 50.0]
# all_p_noise = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
# noise_f = gaussian_noise
# all_r = np.zeros((len(all_stoch), len(all_p_noise), len(all_seeds)))
# for s_i, seed in enumerate(all_seeds):
#     for i, stoch in enumerate(all_stoch):
#         file_name = 'StochasticPendulum' + "_" + str(3) + "_" + "MMM" + "_train" + "_seed=" + str(seed) + "_stoch=" + str(stoch)
#         agent.load("pendulum", file_name)
#         for j, p_noise in enumerate(all_p_noise):
#             print(stoch, p_noise)
#             env_test = make_stochastic_pendulum(render_mode=["image", "sound"], noises=[noise_f], p_noise=p_noise, stoch=stoch)()
#             r, _ = test(agent, env_test, 15, max_T, False)
#             all_r[i, j, s_i] = r
# print()
#
# for i, stoch in enumerate(all_stoch):
#     mu = np.mean(all_r[i, :, :], -1)
#     std = np.std(all_r[i, :, :], -1)
#     plt.plot(all_p_noise, mu, label='stoch='+str(stoch))
#     plt.fill_between(all_p_noise, mu-std, mu+std, alpha=0.2)
# plt.legend()
# plt.show()

print()

''' VISUALIZE LEARNED METRIC SPACE '''
all_sounds = []
all_z_images, all_z_sounds, all_z_images_next, all_z_sounds_next, all_z_bar, all_z_bar_next, all_z_hat_next = [], [], [], [], [], [], []
for theta in tqdm(np.linspace(-np.pi, np.pi, 100)):
    for theta_dot in np.linspace(-8.0, 8, 100):

        env.reset()
        env.env.unwrapped.state = np.array([theta, theta_dot])

        raw_obs = np.array([np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)

        # for i in range(3):
        #     obs, _, _, _, _ = env.step(env.action_space.sample())
        # img = obs['image']
        sound = env._make_sound(raw_obs) #obs['sound']
        a = env.action_space.sample()
        obs_next, _, _, _, _ = env.step(a)
        # img_next = obs_next['image']
        sound_next = obs_next['sound']

        all_sounds.append(sound)

        # z_image = agent.preprocessor.phi['image'](torch.from_numpy(img).float().div_(255).to(device).unsqueeze(0))
        z_sound = agent.preprocessor.phi['sound'](torch.from_numpy(sound).float().to(device).unsqueeze(0))
        # z_image_next = agent.preprocessor.phi['image'](torch.from_numpy(img_next).float().div_(255).to(device).unsqueeze(0))
        z_sound_next = agent.preprocessor.phi['sound'](torch.from_numpy(sound_next).float().to(device).unsqueeze(0))
        # z_bar = 0.5 * z_image + 0.5 * z_sound
        # z_bar_next = 0.5 * z_image_next + 0.5 * z_sound_next
        z_hat_next = agent.preprocessor.phi['transition'](torch.cat([z_sound, torch.tensor([a]).float().to(device)], dim=-1))

        # all_z_images.append(z_image.detach().cpu())
        all_z_sounds.append(z_sound.detach().cpu())
        # all_z_images_next.append(z_image_next.detach().cpu())
        all_z_sounds_next.append(z_sound_next.detach().cpu())
        # all_z_bar.append(z_bar.detach().cpu())
        # all_z_bar_next.append(z_bar_next.detach().cpu())
        all_z_hat_next.append(z_hat_next.detach().cpu())

all_sounds = np.stack(all_sounds, 0)

# all_z_images = torch.cat(all_z_images, 0).detach().cpu().numpy()
all_z_sounds = torch.cat(all_z_sounds, 0).detach().cpu().numpy()
# all_z_images_next = torch.cat(all_z_images_next, 0).detach().cpu().numpy()
all_z_sounds_next = torch.cat(all_z_sounds_next, 0).detach().cpu().numpy()
# all_z_bar = torch.cat(all_z_bar, 0).detach().cpu().numpy()
# all_z_bar_next = torch.cat(all_z_bar_next, 0).detach().cpu().numpy()
all_z_hat_next = torch.cat(all_z_hat_next, 0).detach().cpu().numpy()

print()

for stoch in [0.0, 0.5, 5.0, 50.0]:
    env = make_stochastic_pendulum(render_mode=["image", "sound"], noises=all_noises, p_noise=0.0, stoch=stoch)()
    env.reset()
    env.env.unwrapped.state = np.array([0.0, 0.0])
    raw_obs = np.array([np.cos(0.0), np.sin(0.0), 0.0], dtype=np.float32)
    # for i in range(3):
    #     obs, _, _, _, _ = env.step(env.action_space.sample())
    # img = obs['image']
    sound = env._make_sound(raw_obs) #obs['sound']
    th, thdot = env.env.unwrapped.state
    all_next_sounds = []
    for i in range(1000):
        a = env.action_space.sample()
        obs_next, _, _, _, _ = env.step(a)
        all_next_sounds.append(obs_next['sound'])
        env.reset()
        env.env.unwrapped.state = np.array([th, thdot])
    neigh_z_next_sounds = agent.preprocessor.phi['sound'](torch.from_numpy(np.stack(all_next_sounds, 0)).float().to(device)).detach().cpu().numpy()

    plt.scatter(all_sounds[:, 0], all_sounds[:, 1], s=1, alpha=0.1, c='tab:blue')
    plt.scatter(all_sounds[5050:5051, 0], all_sounds[5050:5051, 1], s=10, alpha=1, c='tab:orange')
    neigh_next_sounds = np.stack(all_next_sounds, 0)
    plt.scatter(neigh_next_sounds[:, 0], neigh_next_sounds[:, 1], s=1, alpha=1.0, c='tab:green')
    plt.title("stoch = " + str(stoch))
    plt.show()
    plt.close()

    # plt.scatter(all_z_sounds_next[:, 0], all_z_sounds_next[:, 1], s=1, alpha=0.1, c='tab:blue')
    # plt.scatter(all_z_sounds_next[5050:5051, 0], all_z_sounds_next[5050:5051, 1], s=10, alpha=1, c='tab:orange')
    # plt.scatter(neigh_z_next_sounds[:, 0], neigh_z_next_sounds[:, 1], s=1, alpha=1.0, c='tab:green')
    # plt.title("stoch = " + str(stoch))
    # plt.show()
    # plt.close()

print()

# test_obs, _ = env_test.reset()
# for t in range(max_T):
#     if enable_render and i == tot_evals - 1:
#         all_test_imgs.append(np.transpose(test_obs['image'][-3:], (1, 2, 0)))
#     with torch.no_grad():
#         test_z = agent.get_representation(test_obs, past_state_action=test_past_state_action, phase="test")
#         test_action, _ = agent.get_action(test_z, test=True, state=None)
#         test_past_state_action = {
#             'z': test_z.detach(),
#             'a': test_action.detach()
#         }
#     test_action = test_action.cpu().numpy()[0]
#     test_next_obs, test_reward, test_terminated, test_truncated, info = env_test.step(test_action)
#     test_done = test_terminated + test_truncated
#     test_obs = test_next_obs
#     tot_test_reward += test_reward
#     if test_done:
#         break






















