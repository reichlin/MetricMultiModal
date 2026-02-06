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

from utils.logger import Logger
from envs.noises import *
from envs.stochastic_pendulum import make_stochastic_pendulum

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




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    ''' MANAGEMENT '''
    parser.add_argument('--reload', default=0, type=int)
    parser.add_argument('--save_model', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    ''' REPR '''
    parser.add_argument('--algo', default=3, type=int)  # 0: 'linear_comb', 1: 'concat', 2: 'curl', 3: 'our', 4: 'gmc', 5: 'a-mdf', 6: 'coral'
    parser.add_argument('--z_dim', default=64, type=int)
    ''' RL '''
    parser.add_argument('--rl_algo', default=0, type=int)  # 0: sac, 1: ppo
    ''' ENVS '''
    parser.add_argument('--noise_level', default=0.0, type=float)
    parser.add_argument('--render', default=1, type=int)
    parser.add_argument('--modalities', default=3, type=int)  # 0: 'state', 1: 'image', 2: 'sound', 3: 'all'
    parser.add_argument('--epochs', default=20_000, type=int)

    parser.add_argument('--stoch', default=0.0, type=float)

    args = parser.parse_args()

    img_size = 84
    #
    #         print(seed, stoch)

    stoch = args.stoch

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    enable_render = args.render == 1
    save_model = args.save_model == 1
    epochs = args.epochs

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    env_name = 'StochasticPendulum'
    env_max_R = 0

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
                      2: ["sound"],
                      3: ["image", "sound"]}
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

    envs = SyncVectorEnv([make_stochastic_pendulum(render_mode=modalities, noises=noises, p_noise=p_noise, stoch=stoch, img_size=img_size) for _ in range(configs_rl['n_workers'])])
    if enable_render:
        env_test = make_stochastic_pendulum(render_mode=["image", "sound"], noises=noises, p_noise=0.0, stoch=stoch, img_size=img_size)()
    else:
        env_test = make_stochastic_pendulum(render_mode=["state"], noises=noises, p_noise=0.0, stoch=stoch, img_size=img_size)()

    state_dim = env_test.observation_space.spaces['state'].shape[0]
    action_dim = env_test.action_space.shape[0]
    img_dim = env_test.observation_space['image'].shape[-1] if 'image' in env_test._render_mode else 0
    sound_dim = env_test.observation_space['sound'].shape[-1] if 'sound' in env_test._render_mode else 0
    action_bounds = np.stack([env_test.action_space.low, env_test.action_space.high], 0)
    max_T = env_test.env.spec.max_episode_steps

    configs_rl['architecture']['action_bounds'] = torch.from_numpy(action_bounds).float().to(device)

    ####################################################### MODEL DEF #######################################################

    rl_algo_name = 'sac' if args.rl_algo == 0 else 'ppo'

    frq_training = configs_rl[rl_algo_name]['frq_training']


    if args.algo == 0:
        preprocessor = LinearComb(state_dim, img_dim, sound_dim, args.z_dim, action_dim, modalities, configs_rl, device).to(device)
    elif args.algo == 1:
        preprocessor = ConCat(state_dim, img_dim, sound_dim, args.z_dim, action_dim, modalities, configs_rl, device).to(device)
    elif args.algo == 2:
        preprocessor = Curl(state_dim, img_dim, sound_dim, args.z_dim, action_dim, modalities, configs_rl, device).to(device)
    elif args.algo == 3:
        preprocessor = MMM(state_dim, img_dim, sound_dim, args.z_dim, action_dim, modalities, configs_rl, device).to(device)
    elif args.algo == 4:
        preprocessor = GMC(state_dim, img_dim, sound_dim, args.z_dim, action_dim, modalities, configs_rl, device).to(device)
    elif args.algo == 5:
        preprocessor = AMDF(state_dim, img_dim, sound_dim, args.z_dim, action_dim, modalities, configs_rl, device).to(device)
    elif args.algo == 6:
        preprocessor = CORAL(state_dim, img_dim, sound_dim, args.z_dim, action_dim, modalities, configs_rl, device).to(device)
    else:
        preprocessor = None
        algo_name = ""
        print("not implemented yet")
        exit()

    all_dims = {
        's': state_dim,
        'o': img_dim,
        'sound': sound_dim,
        'a': action_dim,
        'z': preprocessor.z_dim,
        'time': env_test._n_frames,
    }

    if rl_algo_name == 'sac':
        agent = SAC(preprocessor, all_dims, configs_rl, device).to(device)
    else:
        agent = PPO(preprocessor, all_dims, configs_rl, device).to(device)

    ####################################################### TRAIN #######################################################

    file_name = env_name + "_" + str(args.modalities) + "_" + algo_name + "_train" + "_seed=" + str(seed) + "_stoch=" + str(stoch) + '_ablation_no_L_contr3'

    if args.reload == 1:
        agent.load("pendulum", file_name)
    elif save_model:# and (seed in CONVERGED_MODELS[env_name][algo_name]):
        if os.path.isfile('./saved_assets/pendulum/saved_models_'+rl_algo_name+'/'+file_name+'.pt'):
            print("Model already saved, exiting!")
            print()
            exit()

    logger = Logger(name_exp, "pendulum", rl_algo_name, "stochastic=" + str(stoch) + '_seed=' + str(seed) + '_ablation_no_L_contr3')

    train(envs, env_test, agent, file_name, max_T, device, enable_render, logger, env_max_R, frq_training, save_model, epochs, configs_rl['norm_state_epochs'])

    print()













