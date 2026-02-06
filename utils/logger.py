import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class Logger:

    def __init__(self, name_exp, sim_name, rl_algo_name, tags):

        self.name_exp = name_exp
        self.rl_algo_name = rl_algo_name
        self.sim_name = sim_name
        self.writer = SummaryWriter("logs/" + sim_name + "/" + rl_algo_name + "/" + name_exp + tags)
        self.test_rewards = []
        self.stats = {}
        self.counter = 0

    def write(self, name, val, time):

        if name not in self.stats.keys():
            self.stats[name] = [val]
        else:
            self.stats[name].append(val)

    def display(self):
        for name, vals in self.stats.items():
            self.writer.add_scalar(name, np.mean(np.array(vals)), self.counter)
        self.counter += 1
        self.stats = {}

    def append_reward(self, reward):
        if not os.path.exists('saved_assets/'+self.sim_name+'/saved_rewards_'+self.rl_algo_name):
            os.makedirs('saved_assets/'+self.sim_name+'/saved_rewards_'+self.rl_algo_name)

        self.test_rewards.append(reward)
        np.save('saved_assets/'+self.sim_name+"/saved_rewards_"+self.rl_algo_name+"/"+self.name_exp+".npy",
                np.array(self.test_rewards))





































