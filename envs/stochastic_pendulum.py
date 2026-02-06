import os
import gymnasium as gym
from gymnasium.spaces import Dict, Box
import numpy as np
from PIL import Image
from pathlib import Path
import yaml
try:
    from .noises import *
except ImportError:
    from noises import *

try:
    import cv2
except:
    pass

import pygame
from pygame import gfxdraw

COMPATIBILITY_NOISES = {
    "image": [gaussian_noise, salt_and_pepper_noise, patches_noise, puzzle_noise, sensor_failure, texture_noise, hallucination_noise],
    "sound": [gaussian_noise, salt_and_pepper_noise, sensor_failure, hallucination_noise]
}


def make_stochastic_pendulum(render_mode=None, noises=None, p_noise=None, stoch=5.0, img_size=84):
    return lambda: StochasticPendulum(gym.make("Pendulum-v1", render_mode="rgb_array" if render_mode != ["state"] else None),
                                      render_mode,
                                      noises=noises,
                                      p_noise=p_noise,
                                      stoch=stoch,
                                      img_size=img_size)


def modified_doppler_effect(freq, obs_pos, obs_vel, obs_speed, src_pos, src_vel, src_speed, sound_vel):
    if not np.all(src_vel == 0):
        src_vel = src_vel / np.linalg.norm(src_vel)
    if not np.all(obs_vel == 0):
        obs_vel = obs_vel / np.linalg.norm(obs_vel)

    src_to_obs = obs_pos - src_pos
    obs_to_src = src_pos - obs_pos
    if not np.all(src_to_obs == 0):
        src_to_obs = src_to_obs / np.linalg.norm(src_to_obs)
    if not np.all(obs_to_src == 0):
        obs_to_src = obs_to_src / np.linalg.norm(obs_to_src)

    src_radial_vel = src_speed * src_vel.dot(src_to_obs)
    obs_radial_vel = obs_speed * obs_vel.dot(obs_to_src)

    fp = ((sound_vel + obs_radial_vel) / (sound_vel - src_radial_vel)) * freq

    return fp


def inverse_square_law_observer_receiver(obs_pos, src_pos, K=1.0, eps=0.0):
    distance = np.linalg.norm(obs_pos - src_pos)
    return K * 1.0 / (distance**2 + eps)


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class StochasticPendulum(gym.Wrapper):

    def __init__(self, env, render_mode, noises=None, p_noise=0.5, stoch=5.0, img_size=84):

        super().__init__(env)

        try:
            with open('configs/noises.yml', 'r') as file:
                self.configs_noises = yaml.safe_load(file)
                self.configs_noises = self.configs_noises['pendulum']
                parent_folder = "./"
        except FileNotFoundError:
            with open('../configs/noises.yml', 'r') as file:
                self.configs_noises = yaml.safe_load(file)
                self.configs_noises = self.configs_noises['pendulum']
                parent_folder = "../"

        self._render_mode = render_mode
        self._noises = noises
        self._p_noise = p_noise
        self._img_size = img_size

        self.std_stoch = stoch

        self._n_frames = 3

        if render_mode != ["state"]:
            if Path(parent_folder+"bg_images/all_imgs.npy").exists():
                self._all_bg_imgs = {'bg_imgs': np.load(parent_folder+"bg_images/all_imgs.npy", allow_pickle=True)}
            else:
                all_bg_imgs = []
                for file in os.listdir(parent_folder+"bg_images/"):
                    if file[-3:] != "npy":
                        img = np.array(Image.open(parent_folder+"bg_images/" + file).convert("RGB"))
                        img = cv2.resize(img, (self._img_size, self._img_size))
                        all_bg_imgs.append(self._process_img(img))
                all_bg_imgs = np.stack(all_bg_imgs, 0)
                np.save(parent_folder+"bg_images/all_imgs.npy", all_bg_imgs)
                self._all_bg_imgs = {'bg_imgs': np.load(parent_folder+"bg_images/all_imgs.npy", allow_pickle=True)}
        else:
            self._all_bg_imgs = None

        self._init_obs = None

        obs_spaces = {}
        obs_spaces["state"] = self.observation_space
        obs_spaces["raw_state"] = self.observation_space
        if "image" in self._render_mode:
            obs_spaces["image"] = Box(low=0, high=255, shape=(3*self._n_frames, self._img_size, self._img_size), dtype=np.uint8)#np.float32)
        if "sound" in self._render_mode:
            obs_spaces["sound"] = Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self._mean_obs = np.zeros(obs_spaces["state"].shape)
        self._var_obs = np.ones(obs_spaces["state"].shape)

        self.observation_space = Dict(obs_spaces)

        self._last_images = None

    def reset(self, **kwargs):

        raw_obs, info = self.env.reset(**kwargs)

        obs = raw_obs.copy()
        modalities = {'raw_state': raw_obs, 'state': obs}

        if "image" in self._render_mode:
            img = self.render_custom(self.env.unwrapped.state[0])
            # img = cv2.resize(self.render(), (self._img_size, self._img_size))
            modalities['image'] = np.tile(self._process_img(img), (self._n_frames, 1, 1))
            self._last_images = modalities['image'].copy()
        if "sound" in self._render_mode:
            modalities['sound'] = self._make_sound(raw_obs)

        self._init_obs = modalities.copy()

        self.env._elapsed_steps = 0

        return modalities, info

    def step(self, action):

        th, thdot = self.env.unwrapped.state
        g = self.env.unwrapped.g
        m = self.env.unwrapped.m
        l = self.env.unwrapped.l
        dt = self.env.unwrapped.dt
        u = np.clip(action, -self.env.unwrapped.max_torque, self.env.unwrapped.max_torque)[0]
        self.env.unwrapped.last_u = u
        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

        acc_det = 3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * u
        acc_noise = self.env.unwrapped.np_random.normal(0.0, self.std_stoch)
        # acc_noise = self.env.unwrapped.np_random.normal(0.0, acc_noise_std) * np.sqrt(dt)
        newthdot = thdot + (acc_det + acc_noise) * dt
        newthdot = np.clip(newthdot, -self.env.unwrapped.max_speed, self.env.unwrapped.max_speed)
        newth = th + newthdot * dt
        # newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * u) * dt
        # newthdot = np.clip(newthdot, -self.env.unwrapped.max_speed, self.env.unwrapped.max_speed)
        # newth = th + newthdot * dt

        self.env.unwrapped.state = np.array([newth, newthdot])

        self.env._elapsed_steps += 1

        raw_obs = np.array([np.cos(newth), np.sin(newth), newthdot], dtype=np.float32)
        reward = -costs
        terminated = False
        truncated = self.env._elapsed_steps >= self.env._max_episode_steps
        info = {}

        obs = raw_obs.copy()
        modalities = {'raw_state': raw_obs, 'state': obs}

        if "image" in self._render_mode:
            img = self.render_custom(self.env.unwrapped.state[0])
            # img = cv2.resize(self.render(), (self._img_size, self._img_size))
            modalities['image'] = np.concatenate([self._last_images[3:], self._process_img(img)], 0)
            self._last_images = modalities['image'].copy()
        if "sound" in self._render_mode:
            modalities['sound'] = self._make_sound(raw_obs)

        modalities = self._add_noise(modalities) if self._noises is not None else modalities

        return modalities, reward, terminated, truncated, info

    def render_custom(self, theta):
        screen_dim = self._img_size

        pygame.init()
        self.screen = pygame.Surface((screen_dim, screen_dim))
        self.surf = pygame.Surface((screen_dim, screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 1.2  # 2.2
        scale = screen_dim / (bound * 2)
        offset = screen_dim // 2

        rod_length = 1 * scale
        rod_width = 0.1 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(theta + np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(
            self.surf, offset, offset, int(rod_width / 2), (204, 77, 77)
        )

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(theta + np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )
        gfxdraw.filled_circle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )
        # drawing axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        img = np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )
        return img

    def _add_noise(self, modalities):

        if np.random.rand() < self._p_noise:
            noise_chosen = False
            while not noise_chosen:
                mode = np.random.choice(list(modalities.keys()))
                if mode in COMPATIBILITY_NOISES.keys():
                    available_noises = [noise for noise in self._noises if noise in COMPATIBILITY_NOISES[mode]]
                    n_noises = len(available_noises)
                    if n_noises > 0:
                        noise_chosen = True
            noise = available_noises[np.random.choice(np.arange(n_noises))]
            modalities[mode] = noise(modalities, mode, self._all_bg_imgs, self._init_obs, self.configs_noises)

        return modalities

    def _process_img(self, img):
        return np.transpose(img, (2, 0, 1))# / 255

    def _make_sound(self, obs):
        x, y, thdot = obs
        abs_src_vel = np.abs(thdot * 1)
        src_vel = np.array([-y, x])
        src_vel = (src_vel / np.linalg.norm(src_vel)) * np.sign(thdot) * abs_src_vel
        src_pos = np.array([x, y])
        original_frequency = 440.
        sound_vel = 20.

        frequencies = np.array([modified_doppler_effect(
            original_frequency,
            obs_pos=pos,
            obs_vel=np.zeros(2),
            obs_speed=0.0,
            src_pos=src_pos,
            src_vel=src_vel,
            src_speed=np.linalg.norm(src_vel),
            sound_vel=sound_vel) / 450 for pos in [np.array([2.2, -2.2]),
                                                   np.array([2.2, 2.2]),
                                                   np.array([-2.2, 0.0])]])
        amplitudes = np.array([inverse_square_law_observer_receiver(
            obs_pos=pos,
            src_pos=src_pos) for pos in [np.array([2.2, -2.2]),
                                         np.array([2.2, 2.2]),
                                         np.array([-2.2, 0.0])]])
        return np.hstack([frequencies, amplitudes])
        # pos = np.array([2.2, -2.2])
        # frequency = modified_doppler_effect(
        #     original_frequency,
        #     obs_pos=pos,
        #     obs_vel=np.zeros(2),
        #     obs_speed=0.0,
        #     src_pos=src_pos,
        #     src_vel=src_vel,
        #     src_speed=np.linalg.norm(src_vel),
        #     sound_vel=sound_vel) / 450
        # amplitude = inverse_square_law_observer_receiver(
        #     obs_pos=pos,
        #     src_pos=src_pos)
        # return np.array([frequency, amplitude])

    def _update_obs_stats(self, batch_obs):
        batch_mu, batch_var = batch_obs.mean(0), batch_obs.var(0)
        self._mean_obs = 0.99 * self._mean_obs + (1 - 0.99) * batch_mu
        self._var_obs = 0.99 * self._var_obs + (1 - 0.99) * batch_var

    def set_obs_stats(self, mean, var):
        self._mean_obs[:] = mean      # [:] keeps shared dtype/shape
        self._var_obs[:] = var



if __name__ == "__main__":

    import matplotlib.pyplot as plt

    render_mode = ["image", "sound"]
    noises = [salt_and_pepper_noise] # [gaussian_noise, salt_and_pepper_noise, patches_noise, puzzle_noise, sensor_failure, texture_noise, hallucination_noise]
    p_noise = 1.0

    all_noises = [gaussian_noise, salt_and_pepper_noise, patches_noise, puzzle_noise, sensor_failure, texture_noise, hallucination_noise]
    all_noise_names = ["gaussian", "salt_pepper", "patches", "puzzle", "failure", "texture", "hallucination"]

    env = make_stochastic_pendulum(render_mode=render_mode, noises=all_noises, p_noise=1.0)()
    state, _ = env.reset()
    modalities, reward, terminated, truncated, info = env.step(env.action_space.sample())

    plt.figure()
    plt.imshow(modalities['image'][:3].transpose((1, 2, 0)))
    plt.tight_layout()
    plt.savefig("../figs/no_noise.pdf")
    plt.close()

    exit()

    for noise, noise_name in zip(all_noises, all_noise_names):

        noises = [noise]

        env = make_sid_env(env_name, "image", noises=noises, p_noise=p_noise)()
        state, _ = env.reset()
        modalities, reward, terminated, truncated, info = env.step(env.action_space.sample())
        plt.figure()
        plt.imshow(modalities['image'][:3].transpose((1, 2, 0)))
        plt.tight_layout()
        plt.savefig("../figs/" + noise_name + ".pdf")
        plt.close()


