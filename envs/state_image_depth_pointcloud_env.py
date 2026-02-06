import os
import gymnasium as gym
import gymnasium_robotics as gr
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
from .fetch_wrappers import DepthObsWrapper, PointCloudObsWrapper
from gymnasium.spaces import Dict, Box, Sequence
import numpy as np
from PIL import Image
from pathlib import Path
import yaml
from .noises import *

try:
    import cv2
except:
    pass

COMPATIBILITY_NOISES = {
    #"state": [gaussian_noise, salt_and_pepper_noise, sensor_failure, hallucination_noise],
    "image":      [gaussian_noise, salt_and_pepper_noise, patches_noise, puzzle_noise, sensor_failure, texture_noise, hallucination_noise],
    "depth":      [gaussian_noise, salt_and_pepper_noise, patches_noise, puzzle_noise, sensor_failure,                hallucination_noise],
    "pointcloud": [gaussian_noise, salt_and_pepper_noise,                              sensor_failure,                hallucination_noise]
}

DEFAULT_CAMERA_CONFIG = {
    "distance": 1.5, #2.5,
    "azimuth": 132.0,
    "elevation": -14.0,
    "lookat": np.array([1.3, 0.75, 0.55]),
}

MAX_PC_N = 180

def make_sidp_env(env_name=None, render_mode=None, noises=None, p_noise=None, img_size=128, noised_mods=1, sparse_reward=False, sticky=1):
    return lambda: NoisyEnvSIDP(PointCloudObsWrapper(DepthObsWrapper(gym.make(env_name,
                                                                              render_mode="rgb_array" if render_mode != ["state"] else None,
                                                                              width=img_size,
                                                                              height=img_size,
                                                                              default_camera_config=DEFAULT_CAMERA_CONFIG),
                                                                     width=img_size,
                                                                     height=img_size,
                                                                     camera_name=None),
                                                     camera_name="external_camera_0",
                                                     stride=2, #2,
                                                     world_frame=True,
                                                     near_clip=0.4, #0.03,
                                                     far_clip=0.68,), #5.0,),
                                render_mode,
                                noises=noises,
                                p_noise=p_noise,
                                img_size=img_size,
                                noised_mods=noised_mods,
                                sparse_reward=sparse_reward,
                                sticky=sticky)


class NoisyEnvSIDP(gym.Wrapper):

    def __init__(self, env, render_mode, noises=None, p_noise=0.5, img_size=128, noised_mods=1, sparse_reward=False, sticky=1):

        super().__init__(env)

        with open('configs/noises.yml', 'r') as file:
            self.configs_noises = yaml.safe_load(file)
            self.configs_noises = self.configs_noises['fetch']

        self._render_mode = render_mode
        self._noises = noises
        self._p_noise = p_noise
        self._img_size = img_size

        self._noised_mods = noised_mods
        self._sparse_reward = sparse_reward

        self._n_frames = 3

        self.sticky = sticky
        self.counter_sticky = 0
        self.last_noise_mod = None

        if render_mode != ["state"]:
            if Path("./bg_images/all_imgs_128.npy").exists():
                self._all_bg_imgs = {'bg_imgs': np.load("./bg_images/all_imgs_128.npy", allow_pickle=True)}
            else:
                all_bg_imgs = []
                for file in os.listdir("./bg_images/"):
                    if file[-3:] != "npy":
                        img = np.array(Image.open("./bg_images/" + file).convert("RGB"))
                        img = cv2.resize(img, (img_size, img_size))
                        all_bg_imgs.append(self._process_img(img))
                all_bg_imgs = np.stack(all_bg_imgs, 0)
                np.save("./bg_images/all_imgs_128.npy", all_bg_imgs)
                self._all_bg_imgs = {'bg_imgs': np.load("./bg_images/all_imgs_128.npy", allow_pickle=True)}
        else:
            self._all_bg_imgs = None

        self._init_obs = None

        obs_spaces = {}
        # if self._render_mode == "state" or self._render_mode == 'all':
        S = self.observation_space.spaces['observation'].shape[0]
        AG = self.observation_space.spaces['achieved_goal'].shape[0]
        DG = self.observation_space.spaces['desired_goal'].shape[0]
        obs_spaces["state"] = Box(low=-np.inf, high=np.inf, shape=(S + AG + DG,), dtype=np.float32)
        obs_spaces["raw_state"] = Box(low=-np.inf, high=np.inf, shape=(S + AG + DG,), dtype=np.float32)
        if "image" in self._render_mode:
            obs_spaces["image"] = Box(low=0, high=255, shape=(3 * self._n_frames, img_size, img_size), dtype=np.uint8)
        if "depth" in self._render_mode:
            obs_spaces["depth"] = Box(low=0, high=255, shape=(1 * self._n_frames, img_size, img_size), dtype=np.uint8)
        if "pointcloud" in self._render_mode:
            # obs_spaces["pointcloud"] = Sequence(space=Sequence(space=Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)))
            # obs_spaces["pc_rgb"] = Sequence(space=Sequence(Box(low=0, high=255, shape=(3,), dtype=np.uint8)))
            obs_spaces["pointcloud"] = space=Box(low=-np.inf, high=np.inf, shape=(self._n_frames, MAX_PC_N, 3), dtype=np.float32)
            obs_spaces["pc_rgb"] = Box(low=0, high=255, shape=(self._n_frames, MAX_PC_N, 3), dtype=np.uint8)

        self._mean_obs = None #np.zeros(obs_spaces["state"].shape)
        self._var_obs = None #np.ones(obs_spaces["state"].shape)

        self.observation_space = Dict(obs_spaces)

        self._last_images = None
        self._last_depths = None
        self._last_pc = None
        self._last_pc_rgb = None

    def reset(self, **kwargs):

        raw_obs, info = self.env.reset(**kwargs)
        state = raw_obs['observation'] #self._normalize_obs(raw_obs['observation'])

        self.init_cube_goal_d = np.linalg.norm(raw_obs['desired_goal'] - raw_obs['achieved_goal'], axis=-1)

        modalities = {'raw_state': np.concatenate([raw_obs['observation'],
                                                   raw_obs['achieved_goal'],
                                                   raw_obs['desired_goal']]),
                      'state': np.concatenate([state,
                                               raw_obs['achieved_goal'],
                                               raw_obs['desired_goal']])}

        if "image" in self._render_mode:
            modalities['image'] = np.tile(self._process_img(raw_obs['rgb']), (self._n_frames, 1, 1))
            self._last_images = modalities['image'].copy()
        if "depth" in self._render_mode:
            modalities['depth'] = np.tile(self._process_depth(raw_obs['depth'], raw_obs['rgb']), (self._n_frames, 1, 1))
            self._last_depths = modalities['depth'].copy()
        if "pointcloud" in self._render_mode:
            pc, pc_rgb = self._process_pointcloud(raw_obs['pointcloud'], raw_obs['pc_rgb'])
            modalities['pointcloud'] = np.tile(pc, (self._n_frames, 1, 1))
            modalities['pc_rgb'] = np.tile(pc_rgb, (self._n_frames, 1, 1))
            # modalities['pointcloud'] = [pc]*self._n_frames
            # modalities['pc_rgb'] = [pc_rgb]*self._n_frames
            self._last_pc = modalities['pointcloud'].copy()
            self._last_pc_rgb = modalities['pc_rgb'].copy()

        self._init_obs = modalities.copy()

        return modalities, info

    def step(self, action):

        raw_obs, reward, terminated, truncated, info = self.env.step(action)
        state = raw_obs['observation'] #self._normalize_obs(raw_obs['observation'])

        reward = self.reward_shaping(raw_obs)

        modalities = {'raw_state': np.concatenate([raw_obs['observation'],
                                                   raw_obs['achieved_goal'],
                                                   raw_obs['desired_goal']]),
                      'state': np.concatenate([state,
                                               raw_obs['achieved_goal'],
                                               raw_obs['desired_goal']])}

        if "image" in self._render_mode:
            modalities['image'] = np.concatenate([self._last_images[3:], self._process_img(raw_obs['rgb'])], 0)
            self._last_images = modalities['image'].copy()
        if "depth" in self._render_mode:
            self._all_bg_imgs['depth'] = raw_obs['depth']
            modalities['depth'] = np.concatenate([self._last_depths[1:], self._process_depth(raw_obs['depth'], raw_obs['rgb'])], 0)
            self._last_depths = modalities['depth'].copy()
        if "pointcloud" in self._render_mode:
            pc, pc_rgb = self._process_pointcloud(raw_obs['pointcloud'], raw_obs['pc_rgb'])
            modalities['pointcloud'] = np.concatenate([self._last_pc[1:], np.expand_dims(pc, 0)], 0)
            modalities['pc_rgb'] = np.concatenate([self._last_pc_rgb[1:], np.expand_dims(pc_rgb, 0)], 0)
            # modalities['pointcloud'] = self._last_pc[1:] + [pc]
            # modalities['pc_rgb'] = self._last_pc_rgb[1:] + [pc_rgb]
            self._last_pc = modalities['pointcloud'].copy()
            self._last_pc_rgb = modalities['pc_rgb'].copy()

        modalities = self._add_noise(modalities) if self._noises is not None else modalities

        return modalities, reward, terminated, truncated, info

    def _add_noise(self, modalities):

        if self.counter_sticky > 0:
            if self.counter_sticky == self.sticky:
                self.counter_sticky = 0
            else:
                modalities[self.last_noise_mod[0]] = self.last_noise_mod[1](modalities, self.last_noise_mod[0], self._all_bg_imgs, self._init_obs, self.configs_noises)

        if np.random.rand() < self._p_noise:
            noisified_mods = []
            for _ in range(self._noised_mods):
                noise_chosen = False
                while not noise_chosen:
                    mode = np.random.choice(list(modalities.keys()))
                    if mode in COMPATIBILITY_NOISES.keys() and mode not in noisified_mods:
                        available_noises = [noise for noise in self._noises if noise in COMPATIBILITY_NOISES[mode]]
                        n_noises = len(available_noises)
                        if n_noises > 0:
                            noise_chosen = True
                noise = available_noises[np.random.choice(np.arange(n_noises))]
                if mode == 'pointcloud' and noise == hallucination_noise:
                    modalities['pc_rgb'] = noise(modalities, 'pc_rgb', self._all_bg_imgs, self._init_obs, self.configs_noises)
                modalities[mode] = noise(modalities, mode, self._all_bg_imgs, self._init_obs, self.configs_noises)

                noisified_mods.append(mode)

                self.counter_sticky = 1
                self.last_noise_mod = [mode, noise]

        return modalities

    def _process_img(self, img):
        return np.transpose(img, (2, 0, 1))# / 255

    def _process_depth(self, depth, img):
        depth_norm = np.clip(depth, a_min=0.96, a_max=0.99)
        depth_norm = ((depth_norm - 0.96) / (0.99 - 0.96))
        goal_mask = (img[:, :, 0] > 200) * (img[:, :, 1:] < 50).prod(-1)
        obj_mask = (img[:, :, :] < 50).prod(-1)
        depth_norm_aug = (depth_norm * (1 - obj_mask) + 1 * goal_mask)
        depth_norm_aug = (np.clip(depth_norm_aug, a_min=0, a_max=1)*255).astype(np.uint8)
        return np.expand_dims(depth_norm_aug, 0)

    def _process_pointcloud(self, pc, pc_rgb):
        red_mask = (pc_rgb[:, 0] > 200) * (pc_rgb[:, 1:] < 70).prod(-1) * 1.0
        black_mask = (pc_rgb[:, :] < 80).prod(-1) * 1.0
        blue_mask = (pc_rgb[:, 2] > 200) * (pc_rgb[:, :2] < 200).prod(-1) * 1.0
        idx = np.nonzero(red_mask + black_mask + blue_mask)[0]
        if idx.shape[0] > MAX_PC_N:
            idx = idx[:MAX_PC_N]
        pc = np.pad(pc[idx] - np.array([[2.4, 0.6, 1.6]]), pad_width=((0, MAX_PC_N - idx.shape[0]), (0, 0)), mode='constant', constant_values=0)
        pc_rgb = np.pad(pc_rgb[idx], pad_width=((0, MAX_PC_N - idx.shape[0]), (0, 0)), mode='constant', constant_values=0)
        return pc, pc_rgb

        # return pc[idx] - np.array([[2.4, 0.6, 1.6]]), pc_rgb[idx]

    # def _process_pointcloud_rgb(self, pc_rgb):
    #     return pc_rgb

    # def _normalize_obs(self, state):
    #     return (state - self._mean_obs) / np.sqrt(self._var_obs + 1e-8)
    #
    def _update_obs_stats(self, batch_obs):
        return

    def set_obs_stats(self, mean, var):
        return

    def reward_shaping(self, obs):

        p_ee = obs['observation'][:3]
        p_cube = obs['achieved_goal']
        p_goal = obs['desired_goal']

        if self._sparse_reward:
            return (np.linalg.norm(obs['achieved_goal']-obs['desired_goal']) < 0.05) * 1.0

        if self.spec.id == 'FetchReachDense-v4':
            return - np.linalg.norm(p_goal - p_ee, axis=-1)
        elif self.spec.id == 'FetchPushDense-v4':
            interaction_point = p_cube + ((p_cube - p_goal) / np.linalg.norm(p_cube - p_goal, axis=-1, keepdims=True)) * 0.06
        elif self.spec.id == 'FetchPickAndPlaceDense-v4':
            interaction_point = p_cube
        elif self.spec.id == 'FetchSlideDense-v4':
            interaction_point = p_cube

        # d_p_t  = np.linalg.norm(interaction_point - self._old_p_ee, axis=-1)
        # d_p_t1 = np.linalg.norm(interaction_point - p_ee, axis=-1)
        # neg_delta_d_p = - (d_p_t1 - d_p_t)
        #
        # d_g_t  = np.linalg.norm(p_goal - self._p_cube, axis=-1)
        # d_g_t1 = np.linalg.norm(p_goal - p_cube, axis=-1)
        # neg_delta_d_g = - (d_g_t1 - d_g_t)

        neg_d_p = - np.linalg.norm(interaction_point - p_ee, axis=-1)
        neg_delta_d_g = - (np.linalg.norm(p_goal - p_cube, axis=-1) - self.init_cube_goal_d)

        # self._old_p_ee = p_ee
        # self._p_cube = p_cube
        # self._p_goal = p_goal

        return neg_delta_d_g + 0.1 * neg_d_p #10.0 * neg_delta_d_g + 1.0 * neg_delta_d_p




























