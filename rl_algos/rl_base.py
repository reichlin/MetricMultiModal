import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt



class RL_ALGO(nn.Module):

    def __init__(self, preprocessor, all_dims, configs_rl, device):
        super(RL_ALGO, self).__init__()

        self.lambda_repr = 1.0

        self.mean_obs = nn.Parameter(torch.zeros(all_dims['s']), requires_grad=False)
        self.var_obs = nn.Parameter(torch.zeros(all_dims['s']), requires_grad=False)

    def get_representation(self, obs, past_state_action=None, phase='collect'):
        if type(list(obs.values())[0]) != torch.Tensor:
            obs = self.preprocessor.preprocess(obs)
        return self.preprocessor.get_representation(obs, past_state_action=past_state_action, phase=phase)

    def get_action(self, z, test=False, state=None):
        return self.policy.sample(z, test=test)

    def store_data(self, transition):
        self.replay_buffer.add(transition)

    def save(self, sim_name, file_name, test_env):
        if test_env._mean_obs is not None:
            self.mean_obs = nn.Parameter(torch.from_numpy(test_env._mean_obs).float().to(self.device), requires_grad=False)
            self.var_obs = nn.Parameter(torch.from_numpy(test_env._var_obs).float().to(self.device), requires_grad=False)
        if not os.path.exists("saved_assets/"+sim_name+"/saved_models_"+self.rl_algo_name):
            os.makedirs("saved_assets/"+sim_name+"/saved_models_"+self.rl_algo_name)
        if self.lambda_repr != 1.0:
            torch.save(self.state_dict(),
                       "saved_assets/" + sim_name + "/saved_models_" + self.rl_algo_name + "/" + file_name + "_lambda_r=" + str(self.lambda_repr) + ".pt")
        else:
            torch.save(self.state_dict(),
                       "saved_assets/" + sim_name + "/saved_models_" + self.rl_algo_name + "/" + file_name + ".pt")

    def load(self, sim_name, file_name):
        try:
            if self.lambda_repr != 1.0:
                self.load_state_dict(torch.load("saved_assets/" + sim_name + "/saved_models_" + self.rl_algo_name + "/" + file_name + "_lambda_r=" + str(self.lambda_repr) + ".pt",
                                                map_location=self.device))
            else:
                self.load_state_dict(torch.load("saved_assets/" + sim_name + "/saved_models_" + self.rl_algo_name + "/" + file_name + ".pt",
                                                map_location=self.device))
            return True
        except FileNotFoundError:
            print("No saved models found for " + file_name + ".")
            return False











