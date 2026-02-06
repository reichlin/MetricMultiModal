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

from envs.noises import *
from envs.state_image_depth_env import make_sid_env
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


def load_model_and_env(sparse_reward=False, sticky=1):

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if sim_type == 0:
        env_encodings = {
            0: {"name": "Ant-v5", "max_R": 6000},  # [-1, 1], T = 1000, max_R = 6000
            1: {"name": "HalfCheetah-v5", "max_R": 10000},  # [-1, 1], T = 1000, max_R = 10000
            2: {"name": "Hopper-v5", "max_R": 3500},  # [-1, 1], T = 1000, max_R = 3500
            3: {"name": "Humanoid-v5", "max_R": 6500},  # [-0.4, 0.4], T = 1000, max_R = 6500
            4: {"name": "Walker2d-v5", "max_R": 5500},  # [-1, 1], T = 1000, max_R = 5500
            7: {"name": "InvertedPendulum-v5", "max_R": 1000},  # [-3, 3], T = 1000, max_R = 1000
        }
        modalities = ["image", "depth"]
    elif sim_type == 1:
        env_encodings = {
            0: {"name": "FetchReachDense-v4", "name_sparse": "FetchReach-v4", "max_R": 0},
            1: {"name": "FetchPushDense-v4", "name_sparse": "FetchPush-v4", "max_R": 0},
            2: {"name": "FetchPickAndPlaceDense-v4", "name_sparse": "FetchPickAndPlace-v4", "max_R": 0},
            3: {"name": "FetchSlideDense-v4", "name_sparse": "FetchSlide-v4", "max_R": 0},
        }
        modalities = ["image", "depth", "pointcloud"]

    env_name = env_encodings[env_id]["name"]
    env_max_R = env_encodings[env_id]["max_R"]

    algo_encodings = {
        0: "LinearComb",
        1: "ConCat",
        2: "Curl",
        3: "MMM",
        4: "GMC",
        5: "AMDF",
        6: "CORAL"
    }
    algo_name = algo_encodings[algo]

    with open('configs/rl.yml', 'r') as file:
        configs_rl = yaml.safe_load(file)

    ####################################################### ENV DEF #######################################################

    if sim_type == 0:
        env = make_sid_env(env_name, modalities, noises=noises, p_noise=p_noise)()
    elif sim_type == 1:
        env = make_sidp_env(env_encodings[env_id]["name"], modalities, noises=noises, p_noise=p_noise, noised_mods=noised_mods, sparse_reward=sparse_reward, sticky=sticky)()


    state_dim = env.observation_space.spaces['state'].shape[0]
    action_dim = env.action_space.shape[0]
    img_dim = env.observation_space['image'].shape[-1]
    if 'pointcloud' in env.observation_space.keys():
        pc_dim = env.observation_space['pointcloud'].shape[1]
    action_bounds = np.stack([env.action_space.low, env.action_space.high], 0)
    max_T = env.env.spec.max_episode_steps

    configs_rl['architecture']['action_bounds'] = torch.from_numpy(action_bounds).float().to(device)

    ####################################################### MODEL DEF #######################################################

    if algo == 0:
        preprocessor = LinearComb(state_dim, img_dim, 6, z_dim, action_dim, modalities, configs_rl, device).to(device)
    elif algo == 1:
        preprocessor = ConCat(state_dim, img_dim, 6, z_dim, action_dim, modalities, configs_rl, device).to(device)
    elif algo == 2:
        preprocessor = Curl(state_dim, img_dim, 6, z_dim, action_dim, modalities, configs_rl, device).to(device)
    elif algo == 3:
        preprocessor = MMM(state_dim, img_dim, 6, z_dim, action_dim, modalities, configs_rl, device).to(device)
    elif algo == 4:
        preprocessor = GMC(state_dim, img_dim, 6, z_dim, action_dim, modalities, configs_rl, device).to(device)
    elif algo == 5:
        preprocessor = AMDF(state_dim, img_dim, 6, z_dim, action_dim, modalities, configs_rl, device).to(device)
    elif algo == 6:
        preprocessor = CORAL(state_dim, img_dim, 6, z_dim, action_dim, modalities, configs_rl, device).to(device)
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
        'time': env._n_frames,
    }
    if "pointcloud" in modalities:
        all_dims['p'] = pc_dim

    if rl_algo_name == 'sac':
        agent = SAC(preprocessor, all_dims, configs_rl, device, lambda_repr=lambda_repr).to(device)
    else:
        agent = PPO(preprocessor, all_dims, configs_rl, device).to(device)

    # file_name = env_name + "_"+str(len(modalities)+1)+"_" + algo_name + "_train" + "_seed=" + str(seed)
    file_name = env_name + "_all_" + algo_name + "_train" + "_seed=" + str(seed)

    if not agent.load(sim_type_name, file_name):
        return None, env, max_T

    return agent, env, max_T


def simulate():

    agent.eval()

    tot_test_reward = 0

    for _ in range(n_tests): #tqdm(range(n_tests)):

        test_past_state_action = None

        # all_test_imgs = []

        test_obs, _ = env.reset()
        for t in range(max_T):
            # all_test_imgs.append(np.transpose(test_obs['image'][-3:], (1, 2, 0)))
            with torch.no_grad():
                test_z = agent.get_representation(test_obs, past_state_action=test_past_state_action, phase="test")
                test_action, _ = agent.get_action(test_z, test=True, state=None)
                test_past_state_action = {
                    'z': test_z.detach(),
                    'a': test_action.detach()
                }
            test_action = test_action.cpu().numpy()[0]
            test_next_obs, test_reward, test_terminated, test_truncated, info = env.step(test_action)
            test_done = test_terminated + test_truncated
            test_obs = test_next_obs
            tot_test_reward += test_reward
            if test_done:
                break

        # imgs = [Image.fromarray(img) for img in all_test_imgs]
        # imgs[0].save("./gifs/test.gif", save_all=True, append_images=imgs[1:], duration=50, loop=0)
        #
        # print()

    tot_test_reward /= n_tests

    return tot_test_reward



z_dim = 64
lambda_repr = 1.0
n_tests = 50
all_noises = [gaussian_noise, salt_and_pepper_noise, patches_noise, puzzle_noise, sensor_failure, texture_noise, hallucination_noise]

############################################### TEST SAVED MODELS ###############################################
if True:
    sim_type = 1
    rl_algo = 0
    sticky = 1
    sim_type_name = "mujoco" if sim_type == 0 else "fetch"
    rl_algo_name = 'sac' if rl_algo == 0 else 'ppo'
    noise = 0
    p_noise = 0.0
    noised_mods = 1
    noises = [all_noises[noise]]

    sparse_reward = False

    env_encodings = {2: 'FetchPickAndPlace', 3: 'FetchSlide'}
    noise_encodings = {0: 'Gaussian', 4: 'Failure', 6: 'Hallucination'}
    algo_encodings = {
        0: "LinearComb",
        1: "ConCat",
        2: "Curl",
        3: "MMM",
        4: "GMC",
        5: "AMDF",
        6: "CORAL"
    }

    # sticky = 1
    # env_id = 3
    # noise = 0
    # algo = 5
    # seed = 0
    #
    # agent, env, max_T = load_model_and_env(sparse_reward, sticky=sticky)
    #
    # all_z_img = []
    # all_z_depth = []
    # all_z_pc = []
    # for _ in tqdm(range(100)):
    #     test_past_state_action = None
    #     test_obs, _ = env.reset()
    #     for t in range(max_T):
    #         with torch.no_grad():
    #             z_agg, (z_pred, z_dict) = agent.get_representation(test_obs, past_state_action=test_past_state_action, phase="train")
    #             all_z_img.append(z_dict['image'].detach().cpu().numpy())
    #             all_z_depth.append(z_dict['depth'].detach().cpu().numpy())
    #             all_z_pc.append(z_dict['pointcloud'].detach().cpu().numpy())
    #             test_action, _ = agent.get_action(z_agg, test=True, state=None)
    #             test_past_state_action = {
    #                 'z': z_agg.detach(),
    #                 'a': test_action.detach()
    #             }
    #         test_action = test_action.cpu().numpy()[0]
    #         test_next_obs, test_reward, test_terminated, test_truncated, info = env.step(test_action)
    #         test_done = test_terminated + test_truncated
    #         test_obs = test_next_obs
    #         if test_done:
    #             break
    #
    # all_z_img = np.concatenate(all_z_img, 0)
    # all_z_depth = np.concatenate(all_z_depth, 0)
    # all_z_pc = np.concatenate(all_z_pc, 0)
    # N = all_z_img.shape[0]
    #
    # from sklearn.manifold import Isomap
    # from sklearn.manifold import TSNE
    #
    # embedding = Isomap(n_components=2, n_neighbors=15, metric="euclidean")
    # proj_z = embedding.fit_transform(np.concatenate([all_z_img, all_z_depth, all_z_pc], 0))
    # proj_z_img = proj_z[:N]
    # proj_z_depth = proj_z[N:2*N]
    # proj_z_pc = proj_z[-N:]
    #
    # print()
    #
    # for p_noise in [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
    #     tmp = np.zeros(3)
    #     for seed in [0, 1, 2]:
    #         agent, env, max_T = load_model_and_env(sparse_reward, sticky=sticky)
    #         agent.preprocessor.use_idw = False
    #         if agent is not None:
    #             test_reward = simulate()
    #             tmp[seed] = test_reward
    #     print(round(tmp.mean(), 2), end=" ")
    #
    # print()


    for sticky in [3, 10]:
        for env_id in [2, 3]:
            for noise in [4]:
                noises = [all_noises[noise]]
                print(sticky, env_id, noise)
                for algo in [0, 1, 2, 3, 4, 5, 6]:
                    print(algo_encodings[algo], end=': ')
                    for p_noise in [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
                        tmp = np.zeros(3)
                        for seed in [0, 1, 2]:
                            agent, env, max_T = load_model_and_env(sparse_reward, sticky=sticky)
                            if agent is not None:
                                test_reward = simulate()
                                tmp[seed] = test_reward
                            env.close()
                        print(str(round(tmp.mean(), 2)) + "+-" + str(round(tmp.std(), 2)), end=" ")
                    print()
                print()


    # for env_id in [2, 3]:
    #     for algo in [0, 1, 2, 3, 4, 5, 6]:
    #         print(env_id, algo, end=" ")
    #         tmp = np.zeros(3)
    #         for seed in [0, 1, 2]:
    #             agent, env, max_T = load_model_and_env(sparse_reward, sticky=sticky)
    #
    #             if agent is not None:
    #                 test_reward = simulate()
    #
    #                 tmp[seed] = test_reward
    #
    #                 print(round(test_reward, 2), end=" ")
    #
    #             env.close()
    #         print(round(tmp.mean(), 2))

    exit()

    # for env_id in [0, 1, 2, 3, 4, 7]:
    #     for algo in [0, 1, 2, 3, 4, 5, 6]:
    #         print(env_id, algo, end=" ")
    #         tmp = np.zeros(3)
    #         for seed in [0, 1, 2]:
    #             agent, env, max_T = load_model_and_env(sparse_reward)
    #
    #             if agent is not None:
    #                 test_reward = simulate()
    #
    #                 tmp[seed] = test_reward
    #
    #                 print(round(test_reward, 2), end=" ")
    #
    #             env.close()
    #         print(round(tmp.mean(), 2))
    #
    # exit()
################################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--sim_type', default=1, type=int)  # 0: mujoco, 1: fetch
parser.add_argument('--rl_algo', default=0, type=int)  # 0: sac, 1: ppo
parser.add_argument('--seed', default=2, type=int)
parser.add_argument('--algo', default=6, type=int)
parser.add_argument('--env_id', default=3, type=int)

# parser.add_argument('--noise', default=2, type=int)
# parser.add_argument('--p', default=0.99, type=float)
# parser.add_argument('--noised_mods', default=2, type=int)


# seed=2_algo=6_env_id=3 DOES NOT EXISTS

args = parser.parse_args()


sim_type = args.sim_type
rl_algo = args.rl_algo
seed = args.seed
env_id = args.env_id
algo = args.algo

# noise = args.noise
# p_noise = args.p
# noised_mods = args.noised_mods


# noised_mods = 1 #2
# for seed in [0]:
#     for env_id in [3]:
#         for algo in [3]:
#             noise = 4
#             p_noise = 0.0
#             for max_a in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
#
#                 noises = [all_noises[noise]]
#                 sim_type_name = "mujoco" if sim_type == 0 else "fetch"
#                 rl_algo_name = 'sac' if rl_algo == 0 else 'ppo'
#
#                 agent, env, max_T = load_model_and_env()
#
#                 if agent is not None:
#
#                     test_reward = simulate()
#
#                     print("z_mmm: ", max_a, test_reward)
#
#                 env.close()
#
#
# exit()

# algo = 3
#
# for env_id in [0, 1, 2, 3]:
#     for seed in [0, 1, 2]:


sim_type_name = "mujoco" if sim_type == 0 else "fetch"
rl_algo_name = 'sac' if rl_algo == 0 else 'ppo'

name_test_file = "./saved_assets/" + sim_type_name + "/saved_test_results_" + rl_algo_name + "/"
name_test_file += 'seed=' + str(seed)
name_test_file += '_algo=' + str(algo)
name_test_file += '_env_id=' + str(env_id)
# name_test_file += '_noise=' + str(noise)
# name_test_file += '_p=' + str(p_noise)
# name_test_file += '_noised_mods=' + str(noised_mods)
name_test_file += ".npy"

all_test_rewards = np.zeros((2, 7, 7))

print("begin")

if not os.path.isfile(name_test_file):

    for m_i, noised_mods in enumerate([1, 2]):
        for n_i, noise in enumerate([0, 1, 2, 3, 4, 5, 6]):
            noises = [all_noises[noise]]
            if noise == 5 and noised_mods == 2:
                continue
            for p_i, p_noise in enumerate([0.99, 0.9, 0.75, 0.5, 0.25, 0.1, 0.01]):

                agent, env, max_T = load_model_and_env()

                if agent is not None:
                    test_reward = simulate()

                    # np.save(name_test_file, np.array([test_reward]))
                    all_test_rewards[m_i, n_i, p_i] = test_reward

                env.close()

                print(noised_mods, noise, p_noise, test_reward)

    np.save(name_test_file, all_test_rewards) #np.array([test_reward]))

    print("saved test rewards")
    # print()

else:

    print("file already exists")






























