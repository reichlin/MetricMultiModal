import os

os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")

import numpy as np
import torch
import argparse
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
import gymnasium_robotics
import yaml
from tqdm import tqdm
import pickle

import matplotlib.pyplot as plt

from utils.logger import Logger
from envs.noises import *
from envs.state_image_depth_pointcloud_env import make_sidp_env

from preprocessors.linear_comb import LinearComb
from preprocessors.concatenation import ConCat
from preprocessors.curl import Curl
from preprocessors.mmm import MMM
from preprocessors.gmc import GMC
from preprocessors.amdf import AMDF
from preprocessors.coral import CORAL

from rl_algos.sac import SAC
from rl_algos.ppo import PPO

from utils.loops import train, collect_buffer



CONVERGED_MODELS = {
    "FetchReachDense-v4":          {"LinearComb": [0, 1, 2], "ConCat": [0, 1, 2], "Curl": [0, 1, 2], "MMM": [0, 1, 2], "GMC": [], "AMDF": [0, 1, 2], "CORAL": []},
    "FetchPushDense-v4":           {"LinearComb": [0, 1, 2], "ConCat": [], "Curl": [0, 1, 2], "MMM": [1], "GMC": [], "AMDF": [0, 2], "CORAL": []},
    "FetchPickAndPlaceDense-v4":   {"LinearComb": [0, 1], "ConCat": [0, 1, 2], "Curl": [0, 1, 2], "MMM": [], "GMC": [], "AMDF": [0, 1], "CORAL": []},
    "FetchSlideDense-v4":          {"LinearComb": [0, 1, 2], "ConCat": [0, 1, 2], "Curl": [0, 1, 2], "MMM": [0, 1, 2], "GMC": [], "AMDF": [], "CORAL": []},
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    ''' MANAGEMENT '''
    parser.add_argument('--reload', default=0, type=int)
    parser.add_argument('--save_model', default=1, type=int)
    parser.add_argument('--seed', default=3, type=int)
    ''' REPR '''
    parser.add_argument('--algo', default=3, type=int)  # 0: 'linear_comb', 1: 'concat', 2: 'curl', 3: 'our', 4: 'gmc', 5: 'a-mdf', 6: 'coral'
    parser.add_argument('--z_dim', default=64, type=int)
    ''' RL '''
    parser.add_argument('--rl_algo', default=0, type=int)  # 0: sac, 1: ppo
    ''' ENVS '''
    parser.add_argument('--env_id', default=1, type=int)  # 0: reach, 1: push, 2: pick, 3: slide
    parser.add_argument('--noise_level', default=0.0, type=float)
    parser.add_argument('--render', default=1, type=int)
    parser.add_argument('--modalities', default=4, type=int)  # 0: 'state', 1: 'image', 2: 'depth', 3: 'pointcloud, 4: 'all'

    parser.add_argument('--save_buffer', default=0, type=int)
    parser.add_argument('--epochs', default=1_000_000, type=int)

    parser.add_argument('--lambda_repr', default=1.0, type=float)

    args = parser.parse_args()

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    enable_render = args.render == 1
    save_model = args.save_model == 1

    save_buffer = args.save_buffer == 1
    epochs = args.epochs
    lambda_repr = args.lambda_repr

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    env_encodings = {
        0: {"name": "FetchReachDense-v4", "name_sparse": "FetchReach-v4", "max_R": 0},
        1: {"name": "FetchPushDense-v4", "name_sparse": "FetchPush-v4", "max_R": 0},
        2: {"name": "FetchPickAndPlaceDense-v4", "name_sparse": "FetchPickAndPlace-v4", "max_R": 0},
        3: {"name": "FetchSlideDense-v4", "name_sparse": "FetchSlide-v4", "max_R": 0},
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
                      3: ["pointcloud"],
                      4: ["image", "depth", "pointcloud"]}
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
    p_noise = args.noise_level

    with open('configs/rl.yml', 'r') as file:
        configs_rl = yaml.safe_load(file)

    envs = SyncVectorEnv([make_sidp_env(env_name, modalities, noises=noises, p_noise=p_noise) for _ in range(configs_rl['n_workers'])])
    env_test = make_sidp_env(env_encodings[args.env_id]["name"], ["image", "depth", "pointcloud"], noises=noises, p_noise=0.0)()

    state_dim = env_test.observation_space.spaces['state'].shape[0]
    action_dim = env_test.action_space.shape[0]
    img_dim = env_test.observation_space['image'].shape[-1] if 'image' in env_test._render_mode else 0
    pc_dim = env_test.observation_space['pointcloud'].shape[1]
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
        'p': pc_dim,
        'a': action_dim,
        'z': preprocessor.z_dim,
        'time': env_test._n_frames,
    }

    if rl_algo_name == 'sac':
        agent = SAC(preprocessor, all_dims, configs_rl, device, lambda_repr=lambda_repr).to(device)
    else:
        agent = PPO(preprocessor, all_dims, configs_rl, device).to(device)

    ####################################################### TRAIN #######################################################

    file_name = env_name + "_" + str(args.modalities) + "_" + algo_name + "_train" + "_seed=" + str(seed)

    if args.reload == 1:
        agent.load("fetch", file_name)
    elif save_model and (seed in CONVERGED_MODELS[env_name][algo_name]):
        if os.path.isfile('./saved_assets/fetch/saved_models_' + rl_algo_name + '/' + file_name + '.pt'):
            print("Model already saved, exiting!")
            print()
            exit()

    logger = Logger(name_exp, "fetch", rl_algo_name, "")

    train(envs, env_test, agent, file_name, max_T, device, enable_render, logger, env_max_R, frq_training, save_model, epochs)

    # if save_buffer:
    #     opt_buffer = collect_buffer(env_test, agent, max_T, device, size=int(1e3))
    #     buffer_file_name = "./saved_assets/fetch/saved_buffers/" + env_name + ".pkl"
    #     with open(buffer_file_name, "wb") as f:
    #         pickle.dump(opt_buffer, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done!")
































