import os
import gymnasium as gym
from gymnasium.spaces import Dict, Box
import numpy as np
from PIL import Image
from pathlib import Path
import yaml
try:
    from .noises import *
    from .fetch_wrappers import DepthObsWrapper
except ImportError:
    from noises import *
    from fetch_wrappers import DepthObsWrapper

try:
    import cv2
except:
    pass

COMPATIBILITY_NOISES = {
    #"state": [gaussian_noise, salt_and_pepper_noise, sensor_failure, hallucination_noise],
    "image": [gaussian_noise, salt_and_pepper_noise, patches_noise, puzzle_noise, sensor_failure, texture_noise, hallucination_noise],
    "depth": [gaussian_noise, salt_and_pepper_noise, patches_noise, puzzle_noise, sensor_failure, hallucination_noise]
}


def make_sid_env(env_name=None, render_mode=None, noises=None, p_noise=None, img_size=84):
    # if render_mode != ["state"]:
    return lambda: NoisyEnvSID(DepthObsWrapper(gym.make(env_name, render_mode="rgb_array" if render_mode != ["state"] else None, width=img_size, height=img_size),  # rgbd_tuple
                                               width=img_size,
                                               height=img_size,
                                               camera_name=None),
                               render_mode,
                               noises=noises,
                               p_noise=p_noise,
                               img_size=84)
    # else:
    #     return lambda: NoisyEnvSID(gym.make(env_name, render_mode=None),
    #                                render_mode,
    #                                noises=noises,
    #                                p_noise=p_noise,
    #                                img_size=84)


class NoisyEnvSID(gym.Wrapper):

    def __init__(self, env, render_mode, noises=None, p_noise=0.5, img_size=84):

        super().__init__(env)

        try:
            with open('configs/noises.yml', 'r') as file:
                self.configs_noises = yaml.safe_load(file)
                self.configs_noises = self.configs_noises['mujoco']
                parent_folder = "./"
        except FileNotFoundError:
            with open('../configs/noises.yml', 'r') as file:
                self.configs_noises = yaml.safe_load(file)
                self.configs_noises = self.configs_noises['mujoco']
                parent_folder = "../"

        self._render_mode = render_mode
        self._noises = noises
        self._p_noise = p_noise
        self._img_size = img_size

        self._n_frames = 3

        if render_mode != ["state"]:
            if Path(parent_folder+"bg_images/all_imgs.npy").exists():
                self._all_bg_imgs = {'bg_imgs': np.load(parent_folder+"bg_images/all_imgs.npy", allow_pickle=True)}
            else:
                all_bg_imgs = []
                for file in os.listdir(parent_folder+"bg_images/"):
                    if file[-3:] != "npy":
                        img = np.array(Image.open(parent_folder+"bg_images/" + file).convert("RGB"))
                        img = cv2.resize(img, (img_size, img_size))
                        all_bg_imgs.append(self._process_img(img))
                all_bg_imgs = np.stack(all_bg_imgs, 0)
                np.save(parent_folder+"bg_images/all_imgs.npy", all_bg_imgs)
                self._all_bg_imgs = {'bg_imgs': np.load(parent_folder+"bg_images/all_imgs.npy", allow_pickle=True)}
        else:
            self._all_bg_imgs = None

        self._init_obs = None

        obs_spaces = {}
        S = self.observation_space.spaces['observation'].shape[0]
        obs_spaces["state"] = Box(low=-np.inf, high=np.inf, shape=(S,), dtype=np.float32)
        obs_spaces["raw_state"] = Box(low=-np.inf, high=np.inf, shape=(S,), dtype=np.float32)
        if "image" in self._render_mode:
            obs_spaces["image"] = Box(low=0, high=255, shape=(3*self._n_frames, img_size, img_size), dtype=np.uint8)#np.float32)
        if "depth" in self._render_mode:
            obs_spaces["depth"] = Box(low=0, high=255, shape=(1*self._n_frames, img_size, img_size), dtype=np.uint8)#np.float32)

        # self._mean_obs = np.zeros(self.observation_space.shape)
        # self._var_obs = np.ones(self.observation_space.shape)
        self._mean_obs = np.zeros(obs_spaces["state"].shape)
        self._var_obs = np.ones(obs_spaces["state"].shape)

        self.observation_space = Dict(obs_spaces)

        self._last_images = None
        self._last_depths = None

    def reset(self, **kwargs):

        raw_obs, info = self.env.reset(**kwargs)
        # obs = self._normalize_obs(raw_obs)
        #
        # modalities = {'raw_state': raw_obs, 'state': obs}
        # if self._render_mode != "state":
        #     img, depth = self.render()
        #     if self._render_mode == "image" or self._render_mode == "all":
        #         modalities['image'] = np.tile(self._process_img(img), (self._n_frames, 1, 1))
        #         self._last_images = modalities['image'].copy()
        #     if self._render_mode == "depth" or self._render_mode == "all":
        #         modalities['depth'] = np.tile(self._process_depth(depth), (self._n_frames, 1, 1))
        #         self._last_depths = modalities['depth'].copy()
        if type(raw_obs) is dict:
            obs = self._normalize_obs(raw_obs['observation'])
            modalities = {'raw_state': raw_obs['observation'], 'state': obs}
        else:
            obs = self._normalize_obs(raw_obs)
            modalities = {'raw_state': raw_obs, 'state': obs}


        if "image" in self._render_mode:
            modalities['image'] = np.tile(self._process_img(raw_obs['rgb']), (self._n_frames, 1, 1))
            self._last_images = modalities['image'].copy()
        if "depth" in self._render_mode:
            modalities['depth'] = np.tile(self._process_depth(raw_obs['depth']), (self._n_frames, 1, 1))
            self._last_depths = modalities['depth'].copy()

        self._init_obs = modalities.copy()

        return modalities, info

    def step(self, action):

        raw_obs, reward, terminated, truncated, info = self.env.step(action)
        # obs = self._normalize_obs(raw_obs)
        #
        # modalities = {'raw_state': raw_obs, 'state': obs}
        # if self._render_mode != "state":
        #     img, depth = self.render()
        #     self._all_bg_imgs['depth'] = depth
        #     if self._render_mode == "image" or self._render_mode == "all":
        #         modalities['image'] = np.concatenate([self._last_images[3:], self._process_img(img)], 0)
        #         self._last_images = modalities['image'].copy()
        #     if self._render_mode == "depth" or self._render_mode == "all":
        #         modalities['depth'] = np.concatenate([self._last_depths[1:], self._process_depth(depth)], 0)
        #         self._last_depths = modalities['depth'].copy()
        if type(raw_obs) is dict:
            obs = self._normalize_obs(raw_obs['observation'])
            modalities = {'raw_state': raw_obs['observation'], 'state': obs}
        else:
            obs = self._normalize_obs(raw_obs)
            modalities = {'raw_state': raw_obs, 'state': obs}

        if "image" in self._render_mode:
            modalities['image'] = np.concatenate([self._last_images[3:], self._process_img(raw_obs['rgb'])], 0)
            self._last_images = modalities['image'].copy()
        if "depth" in self._render_mode:
            self._all_bg_imgs['depth'] = raw_obs['depth']
            modalities['depth'] = np.concatenate([self._last_depths[1:], self._process_depth(raw_obs['depth'])], 0)
            self._last_depths = modalities['depth'].copy()

        modalities = self._add_noise(modalities) if self._noises is not None else modalities

        return modalities, reward, terminated, truncated, info

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

    def _process_depth(self, depth):
        return (np.expand_dims(((depth-0.95)/0.05), 0)*255).astype(np.uint8)

    def _normalize_obs(self, state):
        return (state - self._mean_obs) / np.sqrt(self._var_obs + 1e-8)

    def _update_obs_stats(self, batch_obs):
        batch_mu, batch_var = batch_obs.mean(0), batch_obs.var(0)
        self._mean_obs = 0.99 * self._mean_obs + (1 - 0.99) * batch_mu
        self._var_obs = 0.99 * self._var_obs + (1 - 0.99) * batch_var

    def set_obs_stats(self, mean, var):
        self._mean_obs[:] = mean      # [:] keeps shared dtype/shape
        self._var_obs[:] = var



if __name__ == "__main__":

    import matplotlib.pyplot as plt

    env_name = "Ant-v5"
    modalities = "all"
    noises = [salt_and_pepper_noise] # [gaussian_noise, salt_and_pepper_noise, patches_noise, puzzle_noise, sensor_failure, texture_noise, hallucination_noise]
    p_noise = 1.0

    all_noises = [gaussian_noise, salt_and_pepper_noise, patches_noise, puzzle_noise, sensor_failure, texture_noise, hallucination_noise]
    all_noise_names = ["gaussian", "salt_pepper", "patches", "puzzle", "failure", "texture", "hallucination"]

    env = make_sid_env(env_name, "image", noises=noises, p_noise=0.0)()
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


