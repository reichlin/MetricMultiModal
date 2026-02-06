import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

from rl_algos.sac import SAC
from rl_algos.ppo import PPO

import time
# from torch.profiler import profile, record_function, ProfilerActivity


def train(envs, env_test, agent, file_name, max_T, device, enable_render, logger, env_max_R, frq_training, save_model, epochs, norm_state_epochs=None):

    total_steps = epochs #int(1e7) #int(1e5) #int(1e7)
    start_steps = int(1e3) #int(1e4) if type(agent) == SAC else 0
    frq_testing = int(1e3)
    tot_evals = 15

    episode_reward = 0
    best_test_reward = -np.inf
    best_repr_loss = np.inf
    power_consumption_log = []

    last_obs = []

    past_state_action = None

    obs, _ = envs.reset()
    obs_torch = agent.preprocessor.preprocess(obs)

    for step in tqdm(range(total_steps)):

        # if step % 10000 == 0:
        #     agent.load_opt_buffer()

        agent.eval()

        last_obs.append(obs['raw_state'])

        if step < start_steps:
            actions_np = envs.action_space.sample()
            other_agent_info = ()
        else:
            with torch.no_grad():
                z = agent.get_representation(obs_torch, past_state_action=past_state_action, phase="collect")
                actions, other_agent_info = agent.get_action(z, test=False, state=obs['state'])
            actions_np = actions.cpu().numpy()

            past_state_action = {
                'z': z.detach(),
                'a': actions.detach()
            }
        # actions_np = envs.action_space.sample()
        # other_agent_info = ()

        next_obs, rewards, terminated, truncated, info = envs.step(actions_np)
        next_obs_torch = agent.preprocessor.preprocess(next_obs)

        # dones = terminated
        dones = np.logical_or(terminated, truncated)

        if (np.logical_or(terminated, truncated)).any() and past_state_action is not None and agent.preprocessor.recurrent_model:
            new_past_z = agent.get_representation(next_obs_torch, past_state_action=None, phase="collect")
            mask = torch.from_numpy((dones * 1)).to(agent.device).view(-1, 1)
            past_state_action['z'] = (1 - mask) * past_state_action['z'] + mask * new_past_z

        transition = (obs_torch, actions_np, rewards, next_obs_torch, dones) + other_agent_info
        agent.store_data(transition)

        obs_torch = next_obs_torch

        episode_reward += rewards.mean()

        try:
            power_consumption_log.append(torch.cuda.power_draw(device))
        except:
            power_consumption_log.append(0)

        if step % frq_training == frq_training-1:
            agent.train()
            logs = agent.update()

            if logs is not None:
                for name, metric in logs.items():
                    logger.write(name, metric, int(step / frq_training))

        if step % frq_testing == frq_testing - 1:

            tot_test_reward, all_test_imgs = test(agent, env_test, tot_evals, max_T, enable_render)

            logger.write("Training Reward", episode_reward, step)
            episode_reward = 0

            if tot_test_reward is not None:
                logger.write("Test Reward", tot_test_reward - env_max_R, int(step / frq_testing))
            logger.write("Power Consumption", np.array(power_consumption_log).mean(), int(step / frq_training))

            # if 'repr_loss' in logger.stats.keys() and best_repr_loss >= np.array(logger.stats['repr_loss']).mean():
            #     best_repr_loss = np.array(logger.stats['repr_loss']).mean()
            #     if save_model:
            #         agent.save(logger.sim_name, file_name, env_test)

            logger.display()
            logger.append_reward(tot_test_reward)

            if norm_state_epochs is None or step < norm_state_epochs:
                env_test._update_obs_stats(np.concatenate(last_obs, 0))
                envs.call("set_obs_stats", env_test._mean_obs, env_test._var_obs)

            power_consumption_log = []

            if tot_test_reward is not None and tot_test_reward >= best_test_reward:

                best_test_reward = tot_test_reward

                if save_model:
                    agent.save(logger.sim_name, file_name, env_test)

                # if enable_render:
                #     imgs = [Image.fromarray(img) for img in all_test_imgs]
                #     imgs[0].save("./gifs/" + file_name + ".gif", save_all=True, append_images=imgs[1:], duration=50, loop=0)

            last_obs = []

    print("Training finished!")


def test(agent, env_test, tot_evals, max_T, enable_render):
    agent.eval()

    tot_test_reward = 0

    for i in range(tot_evals):

        all_test_imgs = []

        test_past_state_action = None

        # try:
        test_obs, _ = env_test.reset()
        for t in range(max_T):
            if enable_render and i == tot_evals - 1:
                all_test_imgs.append(np.transpose(test_obs['image'][-3:], (1, 2, 0)))
            with torch.no_grad():
                test_z = agent.get_representation(test_obs, past_state_action=test_past_state_action, phase="test")
                test_action, _ = agent.get_action(test_z, test=True, state=None)
                test_past_state_action = {
                    'z': test_z.detach(),
                    'a': test_action.detach()
                }
            test_action = test_action.cpu().numpy()[0]
            test_next_obs, test_reward, test_terminated, test_truncated, info = env_test.step(test_action)
            test_done = test_terminated + test_truncated
            test_obs = test_next_obs
            tot_test_reward += test_reward
            if test_done:
                break
        # except ValueError:
        #     tot_test_reward = None

    if tot_test_reward is not None:
        tot_test_reward /= tot_evals

    return tot_test_reward, all_test_imgs



def collect_buffer(env, agent, max_T, device, size=int(1e3)):
    agent.eval()

    buffer = []

    for i in tqdm(range(size)):

        obs, _ = env.reset()
        for t in range(max_T):
            with torch.no_grad():
                z = agent.get_representation(obs, past_state_action=None, phase="train")
                action, _ = agent.get_action(z, test=False, state=obs['state'])
            action = action.cpu().numpy()[0]
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated + truncated

            datapoint = [
                obs['state'] if 'state' in obs else None,
                obs['image'] if 'image' in obs else None,
                obs['depth'] if 'depth' in obs else None,
                obs['pointcloud'] if 'pointcloud' in obs else None,
                obs['pc_rgb'] if 'pc_rgb' in obs else None,
                action,
                np.array(reward, dtype=np.float32),
                next_obs['state'] if 'state' in next_obs else None,
                next_obs['image'] if 'image' in next_obs else None,
                next_obs['depth'] if 'depth' in next_obs else None,
                next_obs['pointcloud'] if 'pointcloud' in next_obs else None,
                next_obs['pc_rgb'] if 'pc_rgb' in next_obs else None,
                np.array(done, dtype=np.float32),
            ]

            buffer.append(datapoint)

            obs = next_obs
            if done:
                break

    return buffer






































