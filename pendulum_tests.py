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
from preprocessors.concatenation import ConCat
from rl_algos.sac import SAC

from sklearn.manifold import Isomap

from utils.loops import test

def get_space(env, agent, n_samples=100):

    all_states = []
    all_sounds = []
    all_z_sounds = []
    for theta in tqdm(np.linspace(-np.pi, np.pi, n_samples)):
        for theta_dot in np.linspace(-8.0, 8, n_samples):
            _, _ = env.reset()
            env.env.unwrapped.state = np.array([theta, theta_dot])

            raw_obs = np.array([np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)

            sound = env._make_sound(raw_obs)
            z_sound = agent.preprocessor.phi['sound'](torch.from_numpy(sound).float().to(device).unsqueeze(0))

            all_states.append(np.array([theta, theta_dot], dtype=np.float32))
            all_sounds.append(sound)
            all_z_sounds.append(z_sound.detach().cpu().numpy())

    all_states = np.stack(all_states, 0)
    all_sounds = np.stack(all_sounds, 0)
    all_z_sounds = np.concatenate(all_z_sounds, 0)
    return all_states, all_sounds, all_z_sounds


def get_next_dist(i_th, i_th_dot, env, agent, all_sounds, n_i=1000, n_samples=100):

    # i_th, i_th_dot = 25, 50
    theta = np.linspace(-np.pi, np.pi, n_samples)[i_th]
    theta_dot = np.linspace(-8.0, 8, n_samples)[i_th_dot]
    a = env.action_space.sample()

    state_current = np.array([theta, theta_dot])
    sound_current = all_sounds[i_th * n_samples + i_th_dot]
    z_sound_current = agent.preprocessor.phi['sound'](torch.from_numpy(sound_current).float().to(device).unsqueeze(0))
    pred_next_z_sound = agent.preprocessor.phi['transition'](torch.cat([z_sound_current, torch.tensor([a]).float().to(device)], dim=-1))

    dist_next_states = []
    dist_next_sounds = []
    dist_next_z_sounds = []
    for i in range(n_i):
        _, _ = env.reset()
        env.env.unwrapped.state = np.array([theta, theta_dot])
        obs_next, _, _, _, _ = env.step(a)
        dist_next_states.append(env.env.unwrapped.state)
        dist_next_sounds.append(obs_next['sound'])
        dist_next_z_sounds.append(agent.preprocessor.phi['sound'](torch.from_numpy(obs_next['sound']).float().to(device).unsqueeze(0)).detach().cpu().numpy())
    dist_next_states = np.stack(dist_next_states, 0)
    dist_next_sounds = np.stack(dist_next_sounds, 0)
    dist_next_z_sounds = np.concatenate(dist_next_z_sounds, 0)

    z_sound_current = z_sound_current.detach().cpu().numpy()[0]
    pred_next_z_sound = pred_next_z_sound.detach().cpu().numpy()[0]

    return state_current, dist_next_states, dist_next_sounds, dist_next_z_sounds, z_sound_current, pred_next_z_sound, theta, theta_dot


def plot_S_Z(all_states, theta, theta_dot, dist_next_states, all_z_sounds_proj, z_sound_current_proj, dist_next_z_sounds_proj, pred_next_z_sound_proj, stoch, seed):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter(all_states[:, 0], all_states[:, 1], s=1, alpha=0.1, c='tab:blue')
    axes[0].scatter(theta, theta_dot, s=10, c='tab:orange')
    axes[0].scatter(dist_next_states[:, 0], dist_next_states[:, 1], s=1, alpha=0.3, c='tab:green')
    axes[0].set_title("State")

    axes[1].scatter(all_z_sounds_proj[:, 0], all_z_sounds_proj[:, 1], s=1, alpha=0.1, c='tab:blue')
    axes[1].scatter(z_sound_current_proj[:, 0], z_sound_current_proj[:, 1], s=10, c='tab:orange')
    axes[1].scatter(dist_next_z_sounds_proj[:, 0], dist_next_z_sounds_proj[:, 1], s=10, alpha=0.3, c='tab:green')
    axes[1].scatter(pred_next_z_sound_proj[:, 0], pred_next_z_sound_proj[:, 1], s=10, alpha=0.5, c='tab:red')
    axes[1].set_title("Z")

    fig.suptitle("stoch = " + str(stoch) + " seed = " + str(seed), fontsize=14)
    plt.tight_layout()
    return fig



def plot_performances(agent, agent_concat, tot_evals=15):

    all_seeds = [0, 1, 2]
    # all_stoch = [0.0, 0.01, 0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
    all_stoch = [0.0, 5.0, 10.0]
    all_p_noise = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    # all_noises = [gaussian_noise, salt_and_pepper_noise, patches_noise, puzzle_noise, sensor_failure, texture_noise, hallucination_noise]
    all_noises = [gaussian_noise, sensor_failure, hallucination_noise]

    all_r = np.zeros((len(all_noises), len(all_stoch), len(all_p_noise), len(all_seeds)))
    all_r_concat = np.zeros((len(all_noises), len(all_stoch), len(all_p_noise), len(all_seeds)))
    for n_i, noise_f in enumerate(all_noises):
        for s_i, seed in enumerate(all_seeds):
            for i, stoch in enumerate(all_stoch):

                print(noise_f, seed, stoch)

                file_name = 'StochasticPendulum' + "_" + str(3) + "_" + "ConCat" + "_train" + "_seed=" + str(seed) + "_stoch=" + str(stoch)
                agent_concat.load("pendulum", file_name)
                agent_concat.eval()
                agent_concat.preprocessor.eval()
                for j, p_noise in enumerate(all_p_noise):
                    env_test = make_stochastic_pendulum(render_mode=["image", "sound"], noises=[noise_f], p_noise=p_noise, stoch=stoch)()
                    max_T = env.env.spec.max_episode_steps
                    r, _ = test(agent_concat, env_test, tot_evals, max_T, False)
                    all_r_concat[n_i, i, j, s_i] = r

                file_name = 'StochasticPendulum' + "_" + str(3) + "_" + "MMM" + "_train" + "_seed=" + str(seed) + "_stoch=" + str(stoch)
                agent.load("pendulum", file_name)
                agent.eval()
                agent.preprocessor.eval()
                for j, p_noise in enumerate(all_p_noise):
                    env_test = make_stochastic_pendulum(render_mode=["image", "sound"], noises=[noise_f], p_noise=p_noise, stoch=stoch)()
                    max_T = env.env.spec.max_episode_steps
                    r, _ = test(agent, env_test, tot_evals, max_T, False)
                    all_r[n_i, i, j, s_i] = r

    # for j, noise_name in enumerate(['gaussian', 'salt_and_pepper', 'patches', 'puzzle', 'sensor_failure', 'texture',' hallucination']):
    for j, noise_name in enumerate(['gaussian', 'sensor_failure', ' hallucination']):
        for i, stoch in enumerate(all_stoch):

            mu = np.mean(all_r[j, i, :, :], -1)
            std = np.std(all_r[j, i, :, :], -1)
            plt.plot(all_p_noise, mu, label='MMM stoch='+str(stoch))
            plt.fill_between(all_p_noise, mu-std, mu+std, alpha=0.2)

            mu = np.mean(all_r_concat[j, i], -1)
            std = np.std(all_r_concat[j, i], -1)
            plt.plot(all_p_noise, mu, label='Concat stoch='+str(stoch))
            plt.fill_between(all_p_noise, mu - std, mu + std, alpha=0.2)

        plt.legend()
        plt.title(noise_name)
        plt.show()
        plt.close()

    return all_r

def plot_ablation(agent, tot_evals=15):

    all_p_noise = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    all_noises = [gaussian_noise, sensor_failure, hallucination_noise]

    all_r = np.zeros((len(all_noises), len(all_p_noise), 3))
    for n_i, noise_f in enumerate(all_noises):
        for j, p_noise in enumerate(all_p_noise):

            print(noise_f, p_noise)

            env_test = make_stochastic_pendulum(render_mode=["image", "sound"], noises=[noise_f], p_noise=p_noise, stoch=0.0)()

            file_name = 'StochasticPendulum' + "_" + str(3) + "_" + "MMM" + "_train" + "_seed=" + str(0) + "_stoch=" + str(0.0)
            agent.load("pendulum", file_name)
            agent.eval()
            max_T = env.env.spec.max_episode_steps
            r, _ = test(agent, env_test, tot_evals, max_T, False)
            all_r[n_i, j, 0] = r

            file_name = 'StochasticPendulum' + "_" + str(3) + "_" + "MMM" + "_train" + "_seed=" + str(0) + "_stoch=" + str(0.0) + "_ablation_no_L_inv"
            agent.load("pendulum", file_name)
            agent.eval()
            max_T = env.env.spec.max_episode_steps
            r, _ = test(agent, env_test, tot_evals, max_T, False)
            all_r[n_i, j, 1] = r

            file_name = 'StochasticPendulum' + "_" + str(3) + "_" + "MMM" + "_train" + "_seed=" + str(0) + "_stoch=" + str(0.0) + "_ablation_no_L_contr3"
            agent.load("pendulum", file_name)
            agent.eval()
            max_T = env.env.spec.max_episode_steps
            r, _ = test(agent, env_test, tot_evals, max_T, False)
            all_r[n_i, j, 2] = r

    for j, noise_name in enumerate(['gaussian', 'sensor_failure', ' hallucination']):
        plt.plot(all_p_noise, all_r[j, :, 0], label='MMM')
        plt.plot(all_p_noise, all_r[j, :, 1], label='no_L_inv')
        plt.plot(all_p_noise, all_r[j, :, 2], label='no_L_contr')
        plt.legend()
        plt.show()
        plt.close()

    return


''' MAIN '''

z_dim = 64 #2
mods = ["image", "sound"] #["sound"] #["image", "sound"]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

with open('configs/rl.yml', 'r') as file:
    configs_rl = yaml.safe_load(file)

all_noises = [gaussian_noise, salt_and_pepper_noise, patches_noise, puzzle_noise, sensor_failure, texture_noise, hallucination_noise]
env = make_stochastic_pendulum(render_mode=["image", "sound"], noises=all_noises, p_noise=0.0, stoch=0.0)()

state_dim = env.observation_space.spaces['state'].shape[0]
action_dim = env.action_space.shape[0]
img_dim = env.observation_space['image'].shape[-1] if 'image' in env._render_mode else 0
sound_dim = env.observation_space['sound'].shape[-1] if 'sound' in env._render_mode else 0
action_bounds = np.stack([env.action_space.low, env.action_space.high], 0)
max_T = env.env.spec.max_episode_steps

configs_rl['architecture']['action_bounds'] = torch.from_numpy(action_bounds).float().to(device)

preprocessor = MMM(state_dim, img_dim, sound_dim, z_dim, action_dim, mods, configs_rl, device).to(device)
all_dims = {
    's': state_dim,
    'o': img_dim,
    'sound': sound_dim,
    'a': action_dim,
    'z': preprocessor.z_dim,
    'time': env._n_frames,
}
agent = SAC(preprocessor, all_dims, configs_rl, device).to(device)

preprocessor_concat = ConCat(state_dim, img_dim, sound_dim, 64, action_dim, ["image", "sound"], configs_rl, device).to(device)
all_dims_concat = {
    's': state_dim,
    'o': img_dim,
    'sound': sound_dim,
    'a': action_dim,
    'z': preprocessor_concat.z_dim,
    'time': env._n_frames,
}
agent_concat = SAC(preprocessor_concat, all_dims_concat, configs_rl, device).to(device)

# all_r = plot_ablation(agent, tot_evals=20)
# print()

# all_r = plot_performances(agent, agent_concat, tot_evals=50)
# print()

all_seeds = [0, 1, 2]
all_stoch = [0.0, 5.0, 10.0]
all_p_noise = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
all_noises = [gaussian_noise, sensor_failure, hallucination_noise]

all_r = np.load('./saved_assets/pendulum/all_r.npy')
all_r_mmm = all_r[0]
all_r_concat = all_r[1]

for i, stoch in enumerate(all_stoch):
    shade = 0.5 + 0.5 * stoch / 10
    color_mmm = plt.cm.Blues(shade)
    color_concat = plt.cm.Oranges(shade)

    mu = np.mean(all_r_mmm[i], -1)
    std = np.std(all_r_mmm[i], -1)
    plt.plot(all_p_noise, mu, label='MMM stoch=' + str(stoch), color=color_mmm)
    plt.fill_between(all_p_noise, mu - std, mu + std, alpha=0.2, color=color_mmm)

    mu = np.mean(all_r_concat[i], -1)
    std = np.std(all_r_concat[i], -1)
    plt.plot(all_p_noise, mu, label='Concat stoch=' + str(stoch), color=color_concat)
    plt.fill_between(all_p_noise, mu - std, mu + std, alpha=0.2, color=color_concat)

plt.legend(loc='lower center',
           bbox_to_anchor=(0.5, -0.35),  # tweak vertical offset if needed
           ncol=2)
plt.title('hallucination')
plt.tight_layout()
plt.show()
plt.close()


###################################################################################################################################

env_test = make_stochastic_pendulum(render_mode=["image", "sound"], noises=[], p_noise=0.0, stoch=0.0)()

file_name1 = 'StochasticPendulum' + "_" + str(3) + "_" + "MMM" + "_train" + "_seed=" + str(0) + "_stoch=" + str(0.0)
preprocessor_mmm = MMM(state_dim, img_dim, sound_dim, z_dim, action_dim, mods, configs_rl, device).to(device)
agent_mmm = SAC(preprocessor_mmm, all_dims, configs_rl, device).to(device)
agent_mmm.load("pendulum", file_name1)
agent_mmm.eval()

file_name2 = 'StochasticPendulum' + "_" + str(3) + "_" + "MMM" + "_train" + "_seed=" + str(0) + "_stoch=" + str(0.0) + "_ablation_no_L_inv"
preprocessor_no_inv = MMM(state_dim, img_dim, sound_dim, z_dim, action_dim, mods, configs_rl, device).to(device)
agent_no_inv = SAC(preprocessor_no_inv, all_dims, configs_rl, device).to(device)
agent_no_inv.load("pendulum", file_name2)
agent_no_inv.eval()

file_name3 = 'StochasticPendulum' + "_" + str(3) + "_" + "MMM" + "_train" + "_seed=" + str(0) + "_stoch=" + str(0.0) + "_ablation_no_L_contr3"
preprocessor_no_contr = MMM(state_dim, img_dim, sound_dim, z_dim, action_dim, mods, configs_rl, device).to(device)
agent_no_contr = SAC(preprocessor_no_contr, all_dims, configs_rl, device).to(device)
agent_no_contr.load("pendulum", file_name3)
agent_no_contr.eval()

num_samples = 100

all_states = np.zeros((num_samples, num_samples, 2))
all_z_sounds = np.zeros((num_samples, num_samples, 3, z_dim))
all_z_imgs = np.zeros((num_samples, num_samples, 3, z_dim))
for i, theta in tqdm(enumerate(np.linspace(-np.pi, np.pi, num_samples))):
    for j, theta_dot in enumerate(np.linspace(-8.0, 8, num_samples)):
        _, _ = env.reset()
        env.env.unwrapped.state = np.array([theta, theta_dot])
        for _ in range(3):
            obs_next, _, _, _, _ = env.step(env.action_space.sample())
        all_states[i, j] = env.env.unwrapped.state
        obs_sound = torch.from_numpy(obs_next['sound']).float().to(device).unsqueeze(0)
        obs_img = torch.from_numpy(obs_next['image']).float().to(device).div_(255).unsqueeze(0)
        all_z_sounds[i, j, 0] = agent_mmm.preprocessor.phi['sound'](obs_sound).detach().cpu().numpy()
        all_z_sounds[i, j, 1] = agent_no_inv.preprocessor.phi['sound'](obs_sound).detach().cpu().numpy()
        all_z_sounds[i, j, 2] = agent_no_contr.preprocessor.phi['sound'](obs_sound).detach().cpu().numpy()
        all_z_imgs[i, j, 0] = agent_mmm.preprocessor.phi['image'](obs_img).detach().cpu().numpy()
        all_z_imgs[i, j, 1] = agent_no_inv.preprocessor.phi['image'](obs_img).detach().cpu().numpy()
        all_z_imgs[i, j, 2] = agent_no_contr.preprocessor.phi['image'](obs_img).detach().cpu().numpy()


import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

embedding_mmm = Isomap(n_components=2, n_neighbors=15, metric="euclidean")
all_z_mmm = embedding_mmm.fit_transform(np.concatenate([np.reshape(all_z_sounds[:,:,0], (num_samples*num_samples, 64)),
                                                        np.reshape(all_z_imgs[:,:,0], (num_samples*num_samples, 64))], 0))

embedding_no_inv = Isomap(n_components=2, n_neighbors=15, metric="euclidean")
all_z_no_inv = embedding_no_inv.fit_transform(np.concatenate([np.reshape(all_z_sounds[:,:,1], (num_samples*num_samples, 64)),
                                                              np.reshape(all_z_imgs[:,:,1], (num_samples*num_samples, 64))], 0))

embedding_no_contr = Isomap(n_components=2, n_neighbors=15, metric="euclidean")
all_z_no_contr = embedding_no_contr.fit_transform(np.concatenate([np.reshape(all_z_sounds[:,:,2], (num_samples*num_samples, 64)),
                                                                  np.reshape(all_z_imgs[:,:,2], (num_samples*num_samples, 64))], 0))

z_min = np.min(np.concatenate([all_z_mmm, all_z_no_inv, all_z_no_contr], 0), 0)
z_max = np.max(np.concatenate([all_z_mmm, all_z_no_inv, all_z_no_contr], 0), 0)
label_mod = np.concatenate([np.zeros(num_samples*num_samples), np.ones(num_samples*num_samples)], 0)
flat_states = np.reshape(all_states, (num_samples*num_samples, 2))
label_theta = np.concatenate([flat_states[:, 0], flat_states[:, 0]], 0)
label_thetadot = np.concatenate([flat_states[:, 1], flat_states[:, 1]], 0)

print()

fig, axes = plt.subplots(1, 3, figsize=(10, 4))
axes[0].scatter(all_z_mmm[:(num_samples*num_samples), 0],
                all_z_mmm[:(num_samples*num_samples), 1],
                s=1, alpha=0.3, c=label_theta[:(num_samples*num_samples)], cmap='Blues', rasterized=True)
axes[0].scatter(all_z_mmm[(num_samples*num_samples):, 0],
                all_z_mmm[(num_samples*num_samples):, 1],
                s=1, alpha=0.3, c=label_theta[(num_samples*num_samples):], cmap='Oranges', rasterized=True)
axes[0].set_xlim(z_min[0], z_max[0])
axes[0].set_ylim(z_min[1], z_max[1])
axes[0].set_title("MetricMM")
axes[1].scatter(all_z_no_inv[:(num_samples*num_samples), 0],
                all_z_no_inv[:(num_samples*num_samples), 1],
                s=1, alpha=0.3, c=label_theta[:(num_samples*num_samples)], cmap='Blues', rasterized=True)
axes[1].scatter(all_z_no_inv[(num_samples*num_samples):, 0],
                all_z_no_inv[(num_samples*num_samples):, 1],
                s=1, alpha=0.3, c=label_theta[(num_samples*num_samples):], cmap='Oranges', rasterized=True)
axes[1].set_xlim(z_min[0], z_max[0])
axes[1].set_ylim(z_min[1], z_max[1])
axes[1].set_title("MetricMM no L_inv")
axes[2].scatter(all_z_no_contr[:(num_samples*num_samples), 0],
                all_z_no_contr[:(num_samples*num_samples), 1],
                s=1, alpha=0.3, c=label_theta[:(num_samples*num_samples)], cmap='Blues', rasterized=True)
axes[2].scatter(all_z_no_contr[(num_samples*num_samples):, 0],
                all_z_no_contr[(num_samples*num_samples):, 1],
                s=1, alpha=0.3, c=label_theta[(num_samples*num_samples):], cmap='Oranges', rasterized=True)
axes[2].set_xlim(z_min[0], z_max[0])
axes[2].set_ylim(z_min[1], z_max[1])
axes[2].set_title("MetricMM no L_contr")
fig.suptitle("Stochastic Pendulum Latent Representation", fontsize=14)
plt.tight_layout()
plt.savefig("./saved_assets/pendulum/ablation_latent3.pdf", dpi=600)
plt.close()


###################################################################################################################################


for stoch in [10.0]:
    for seed in [0]:

        n_samples = 100
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)


        all_noises = [gaussian_noise, salt_and_pepper_noise, patches_noise, puzzle_noise, sensor_failure, texture_noise, hallucination_noise]
        env = make_stochastic_pendulum(render_mode=["sound"], noises=all_noises, p_noise=0.0, stoch=stoch)()

        # file_name = 'StochasticPendulum' + "_" + str(3) + "_" + "MMM" + "_train" + "_seed=" + str(seed) + "_stoch=" + str(stoch)
        # file_name = 'StochasticPendulum' + "_" + str(2) + "_" + "MMM" + "_train" + "_seed=" + str(seed) + "_stoch=" + str(20.0) + "viz_multiple_receivers_2_z_dim=64"
        file_name = 'StochasticPendulum' + "_" + str(3) + "_" + "MMM" + "_train" + "_seed=" + str(seed) + "_stoch=" + str(stoch)
        # file_name = 'StochasticPendulum' + "_" + str(2) + "_" + "MMM" + "_train" + "_seed=" + str(seed) + "_stoch=" + str(stoch) + '_viz_g0'

        agent.load("pendulum", file_name)
        agent.eval()
        agent.preprocessor.eval()


        ''' VISUALIZE LEARNED METRIC SPACE '''

        (all_states,
         all_sounds,
         all_z_sounds) = get_space(env,
                                   agent,
                                   n_samples=100)

        i_th, i_th_dot = np.random.randint(0, n_samples), np.random.randint(0, n_samples)
        (state_current,
         dist_next_states,
         dist_next_sounds,
         dist_next_z_sounds,
         z_sound_current,
         pred_next_z_sound,
         theta,
         theta_dot) = get_next_dist(i_th,
                                    i_th_dot,
                                    env,
                                    agent,
                                    all_sounds,
                                    n_i=1000,
                                    n_samples=n_samples)

        # idx = np.nonzero(np.abs(all_states[:, 0] - np.pi / 2) < 0.1)[0]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(all_states[:, 0], all_states[:, 1], s=1, alpha=0.1, c='tab:blue')
        axes[0].scatter(theta, theta_dot, s=10, c='tab:orange')
        axes[0].scatter(dist_next_states[:, 0], dist_next_states[:, 1], s=1, alpha=0.3, c='tab:green')
        axes[0].set_title("State")
        axes[1].scatter(all_z_sounds[:, 0], all_z_sounds[:, 1], s=1, alpha=0.1, c='tab:blue')
        # axes[1].scatter(all_z_sounds[idx, 0], all_z_sounds[idx, 1], s=1, alpha=0.5, c='tab:orange')
        axes[1].scatter(z_sound_current[0], z_sound_current[1], s=10, c='tab:orange')
        axes[1].scatter(dist_next_z_sounds[:, 0], dist_next_z_sounds[:, 1], s=10, alpha=0.3, c='tab:green')
        axes[1].scatter(pred_next_z_sound[0], pred_next_z_sound[1], s=10, alpha=0.5, c='tab:red')
        axes[1].set_title("Z")
        fig.suptitle("stoch = " + str(stoch) + " seed = " + str(seed), fontsize=14)
        plt.tight_layout()
        plt.show()


        embedding = Isomap(n_components=2, radius=2.0, n_neighbors=None)
        all_z_sounds_proj = embedding.fit_transform(all_z_sounds)
        z_sound_current_proj = embedding.transform(np.expand_dims(z_sound_current, 0))
        dist_next_z_sounds_proj = embedding.transform(dist_next_z_sounds)
        pred_next_z_sound_proj = embedding.transform(np.expand_dims(pred_next_z_sound, 0))
        print()

        fig = plot_S_Z(all_states, theta, theta_dot, dist_next_states, all_z_sounds_proj, z_sound_current_proj, dist_next_z_sounds_proj, pred_next_z_sound_proj, stoch, seed)
        plt.show()
        plt.close()
        print()


# # fig, axes = plt.subplots(1, 2, figsize=(10, 4))
# fig = plt.figure()
# axes = [fig.add_subplot(1, 2, 1),
#         fig.add_subplot(1, 2, 2, projection='3d')]
#
# axes[0].scatter(all_sounds[:, 0], all_sounds[:, 3], s=1, alpha=0.1, c='tab:blue')
# axes[0].scatter(sound_current[0], sound_current[3], s=10, c='tab:orange')
# axes[0].scatter(dist_next_sounds[:, 0], dist_next_sounds[:, 3], s=1, alpha=0.3, c='tab:green')
# axes[0].set_title("Sound")
#
# axes[1].scatter(all_z_sounds[:, 0], all_z_sounds[:, 1], all_z_sounds[:, 2], s=1, alpha=0.1, c='tab:blue')
# axes[1].scatter(z_sound_current[0], z_sound_current[1], z_sound_current[2], s=10, c='tab:orange')
# axes[1].scatter(dist_next_z_sounds[:, 0], dist_next_z_sounds[:, 1], dist_next_z_sounds[:, 2], s=10, alpha=0.3, c='tab:green')
# axes[1].scatter(pred_next_z_sound[0], pred_next_z_sound[1], pred_next_z_sound[2], s=10, alpha=0.5, c='tab:red')
# axes[1].set_title("Z")
#
# fig.suptitle("stoch = " + str(stoch) + " seed = " + str(seed), fontsize=14)
#
# plt.tight_layout()
# plt.show()
# plt.close()


# env.reset()
# env.env.unwrapped.state = np.array([theta, theta_dot])
# img = env.render()
# plt.imshow(img)
# plt.show()















