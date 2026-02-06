import os

os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")

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

from utils.logger import Logger
from envs.noises import *
from envs.state_image_depth_env import make_sid_env

from preprocessors.linear_comb import LinearComb
from preprocessors.concatenation import ConCat
from preprocessors.curl import Curl
from preprocessors.mmm import MMM
from preprocessors.gmc import GMC
from preprocessors.amdf import AMDF
from preprocessors.coral import CORAL

from rl_algos.sac import SAC
from rl_algos.ppo import PPO

from utils.loops import train

import time


CONVERGED_MODELS = {
    "Ant-v5":               {"LinearComb": [], "ConCat": [], "Curl": [], "MMM": [], "GMC": [], "AMDF": [], "CORAL": []},
    "HalfCheetah-v5":       {"LinearComb": [], "ConCat": [], "Curl": [], "MMM": [], "GMC": [], "AMDF": [], "CORAL": []},
    "Hopper-v5":            {"LinearComb": [], "ConCat": [], "Curl": [], "MMM": [], "GMC": [], "AMDF": [], "CORAL": []},
    "Humanoid-v5":          {"LinearComb": [], "ConCat": [], "Curl": [], "MMM": [], "GMC": [], "AMDF": [], "CORAL": []},
    "Walker2d-v5":          {"LinearComb": [], "ConCat": [], "Curl": [], "MMM": [], "GMC": [], "AMDF": [], "CORAL": []},
    "InvertedPendulum-v5":  {"LinearComb": [], "ConCat": [], "Curl": [], "MMM": [], "GMC": [], "AMDF": [], "CORAL": []},
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    ''' MANAGEMENT '''
    parser.add_argument('--reload', default=0, type=int)
    parser.add_argument('--save_model', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    ''' REPR '''
    parser.add_argument('--algo', default=1, type=int)  # 0: 'linear_comb', 1: 'concat', 2: 'curl', 3: 'our', 4: 'gmc', 5: 'a-mdf', 6: 'coral'
    parser.add_argument('--z_dim', default=64, type=int)
    ''' RL '''
    parser.add_argument('--rl_algo', default=0, type=int)  # 0: sac, 1: ppo
    ''' ENVS '''
    parser.add_argument('--env_id', default=0, type=int)  # from 0 to 5 included
    parser.add_argument('--noise_level', default=0.0, type=float)
    parser.add_argument('--render', default=1, type=int)
    parser.add_argument('--modalities', default=3, type=int)  # 0: 'state', 1: 'image', 2: 'depth', 3: 'all'
    parser.add_argument('--epochs', default=1_000_000, type=int)

    args = parser.parse_args()

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    enable_render = args.render == 1
    save_model = args.save_model == 1
    epochs = args.epochs

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    env_encodings = {
        0: {"name": "Ant-v5", "max_R": 6000},  # [-1, 1], T = 1000, max_R = 6000
        1: {"name": "HalfCheetah-v5", "max_R": 10000},  # [-1, 1], T = 1000, max_R = 10000
        2: {"name": "Hopper-v5", "max_R": 3500},  # [-1, 1], T = 1000, max_R = 3500
        3: {"name": "Humanoid-v5", "max_R": 6500},  # [-0.4, 0.4], T = 1000, max_R = 6500
        4: {"name": "Walker2d-v5", "max_R": 5500},  # [-1, 1], T = 1000, max_R = 5500
        5: {"name": "InvertedPendulum-v5", "max_R": 1000},  # [-3, 3], T = 1000, max_R = 1000
    }
    env_name = env_encodings[args.env_id]["name"]
    env_max_R = env_encodings[args.env_id]["max_R"]

    algo_encodings = {
        0: "LinearComb",
        1: "ConCat",
        2: "Curl",
        3: "MMM",
        4: "GMC",
        5: "AMDF",
        6: "CORAL"
    }
    algo_name = algo_encodings[args.algo]

    name_exp = ''
    for k, v in args.__dict__.items():
        if k == 'algo':
            name_exp += algo_name + "_"
        else:
            name_exp += str(k) + "=" + str(v) + "_"

    ####################################################### ENV DEF #######################################################

    all_modalities = {0: ["state"],
                      1: ["image"],
                      2: ["depth"],
                      3: ["image", "depth"]}
    modalities = all_modalities[args.modalities]
    noises = [gaussian_noise]
    noises_encoding = {
        gaussian_noise: "0",
        salt_and_pepper_noise: "1",
        patches_noise: "2",
        puzzle_noise: "3",
        sensor_failure: "4",
        texture_noise: "5",
        hallucination_noise: "6",
    }
    p_noise = args.noise_level #0.0

    with open('configs/rl.yml', 'r') as file:
        configs_rl = yaml.safe_load(file)

    envs = SyncVectorEnv([make_sid_env(env_name, modalities, noises=noises, p_noise=p_noise) for _ in range(configs_rl['n_workers'])])
    if enable_render:
        env_test = make_sid_env(env_name, ["image", "depth"], noises=noises, p_noise=0.0)()
    else:
        env_test = make_sid_env(env_name, ["state"], noises=noises, p_noise=0.0)()

    state_dim = env_test.observation_space.spaces['state'].shape[0]
    action_dim = env_test.action_space.shape[0]
    img_dim = env_test.observation_space['image'].shape[-1] if 'image' in env_test._render_mode else 0
    action_bounds = np.stack([env_test.action_space.low, env_test.action_space.high], 0)
    max_T = env_test.env.spec.max_episode_steps

    configs_rl['architecture']['action_bounds'] = torch.from_numpy(action_bounds).float().to(device)

    ####################################################### MODEL DEF #######################################################

    rl_algo_name = 'sac' if args.rl_algo == 0 else 'ppo'

    frq_training = configs_rl[rl_algo_name]['frq_training']


    if args.algo == 0:
        preprocessor = LinearComb(state_dim, img_dim, args.z_dim, action_dim, modalities, configs_rl, device).to(device)
    elif args.algo == 1:
        preprocessor = ConCat(state_dim, img_dim, args.z_dim, action_dim, modalities, configs_rl, device).to(device)
    elif args.algo == 2:
        preprocessor = Curl(state_dim, img_dim, args.z_dim, action_dim, modalities, configs_rl, device).to(device)
    elif args.algo == 3:
        preprocessor = MMM(state_dim, img_dim, args.z_dim, action_dim, modalities, configs_rl, device).to(device)
    elif args.algo == 4:
        preprocessor = GMC(state_dim, img_dim, args.z_dim, action_dim, modalities, configs_rl, device).to(device)
    elif args.algo == 5:
        preprocessor = AMDF(state_dim, img_dim, args.z_dim, action_dim, modalities, configs_rl, device).to(device)
    elif args.algo == 6:
        preprocessor = CORAL(state_dim, img_dim, args.z_dim, action_dim, modalities, configs_rl, device).to(device)
    else:
        preprocessor = None
        algo_name = ""
        print("not implemented yet")
        exit()

    all_dims = {
        's': state_dim,
        'o': img_dim,
        'd': img_dim,
        'a': action_dim,
        'z': preprocessor.z_dim,
        'time': env_test._n_frames,
    }

    if rl_algo_name == 'sac':
        agent = SAC(preprocessor, all_dims, configs_rl, device).to(device)
    else:
        agent = PPO(preprocessor, all_dims, configs_rl, device).to(device)

    ####################################################### TRAIN #######################################################

    file_name = env_name + "_" + str(args.modalities) + "_" + algo_name + "_train" + "_seed=" + str(seed)

    if args.reload == 1:
        agent.load("mujoco", file_name)
    elif save_model and (seed in CONVERGED_MODELS[env_name][algo_name]):
        if os.path.isfile('./saved_assets/mujoco/saved_models_'+rl_algo_name+'/'+file_name+'.pt'):
            print("Model already saved, exiting!")
            print()
            exit()

    logger = Logger(name_exp, "mujoco", rl_algo_name, "")

    train(envs, env_test, agent, file_name, max_T, device, enable_render, logger, env_max_R, frq_training, save_model, epochs, configs_rl['norm_state_epochs'])

    print()













