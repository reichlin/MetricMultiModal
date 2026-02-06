import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, architecture_params):
        super(MLP, self).__init__()

        n_neurons = architecture_params['n_neurons']
        n_hidden_layers = architecture_params['n_layers']
        activation_type = nn.ReLU() if architecture_params['activation_type'] == 0 else nn.Tanh()

        layers = [nn.Linear(input_dim, n_neurons), activation_type]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(n_neurons, n_neurons),
                       activation_type]
        layers += [nn.Linear(n_neurons, output_dim)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CNN(nn.Module):

    def __init__(self, input_channels, output_dim, architecture_params):
        super(CNN, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            # nn.Linear(32 * 17 * 17, output_dim)
            nn.LazyLinear(out_features=output_dim)
        )

    def forward(self, x):
        return self.network(x)


class DCNN(nn.Module):

    def __init__(self, input_dim, output_channels, architecture_params):
        super(DCNN, self).__init__()

        self.linear = nn.Linear(input_dim, 64 * 7 * 7)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_channels, kernel_size=8, stride=4)
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 64, 7, 7)
        return self.decoder(x)


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, architecture_params):
        super(Actor, self).__init__()

        self.net = MLP(state_dim, architecture_params['n_neurons'], architecture_params)
        self.mean_head = nn.Linear(architecture_params['n_neurons'], action_dim)
        self.log_std_head = nn.Linear(architecture_params['n_neurons'], action_dim)

        def init_(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        self.net.apply(init_)
        nn.init.uniform_(self.mean_head.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.log_std_head.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.mean_head.bias); nn.init.zeros_(self.log_std_head.bias)

        self.bound_mean = architecture_params['action_bounds'].mean(0)
        self.bound_scale = architecture_params['action_bounds'][1] - self.bound_mean

    def forward(self, state):
        # x = self.net(state) #F.relu(self.net(state))
        x = F.relu(self.net(state))
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)

        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state, test=False):

        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample() if not test else mean

        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - (2 * (torch.log(torch.tensor(2.0, device=x_t.device)) - x_t - F.softplus(-2.0 * x_t)))
        log_prob = log_prob.sum(dim=-1, keepdim=True)  # NO extra -log|scale|
        # scale to env bounds for execution (but keep log_prob in [-1,1] space)
        scaled_action = action * self.bound_scale + self.bound_mean

        # action = torch.tanh(x_t)
        #
        # log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        # log_prob = log_prob.sum(dim=-1, keepdim=True) - torch.log(self.bound_scale.abs()).sum().unsqueeze(0)
        #
        # scaled_action = action * self.bound_scale + self.bound_mean

        return scaled_action, (log_prob,)


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim=0, architecture_params=None):
        super(Critic, self).__init__()

        if action_dim != 0:
            self.net = MLP(state_dim + action_dim, 1, architecture_params)
        else:
            self.net = MLP(state_dim, 1, architecture_params)

    def forward(self, state, action=None):
        x = torch.cat([state, action], dim=-1) if action is not None else state
        q = self.net(x)
        return q

class ActorPPO(nn.Module):

    def __init__(self, state_dim, action_dim, architecture_params, learn_var, init_var):
        super(ActorPPO, self).__init__()

        self.policy_body = MLP(state_dim, architecture_params['n_neurons'], architecture_params)
        self.mean_head = nn.Linear(architecture_params['n_neurons'], action_dim)
        if learn_var:
            self.log_std = nn.Parameter(torch.ones(1, action_dim) * init_var, requires_grad=True).float()  # -2.0 -1.2
        else:
            self.log_std = nn.Parameter(torch.ones(1, action_dim) * init_var, requires_grad=False).float() # -2.0 -1.2

        self.bound_mean = architecture_params['action_bounds'].mean(0)
        self.bound_scale = architecture_params['action_bounds'][1] - self.bound_mean

        for layer in self.policy_body.network:
            if type(layer) == nn.Linear:
                torch.nn.init.orthogonal_(layer.weight, np.sqrt(2))
        torch.nn.init.orthogonal_(self.mean_head.weight, np.sqrt(2))

    def forward(self, state):

        x = F.relu(self.policy_body(state))
        mean = self.mean_head(x)
        log_std = self.log_std

        log_std = torch.clamp(log_std, min=-10, max=0)
        std = log_std.exp()
        return mean, std

    def sample(self, state, test=False):
        mean, std = self.forward(state)

        normal = torch.distributions.Normal(mean, std)
        action = normal.sample() if not test else mean
        scaled_action = torch.tanh(action) * self.bound_scale + self.bound_mean
        log_prob = (normal.log_prob(action) - torch.log1p(-torch.tanh(action).pow(2) + 1e-6)).sum(-1)

        other_info = (action.detach().cpu().numpy(), log_prob.detach().cpu().numpy())

        return scaled_action, other_info

    def evaluate(self, state, raw_action):

        mean, std = self.forward(state)

        normal = torch.distributions.Normal(mean, std)
        log_prob = (normal.log_prob(raw_action) - torch.log1p(-torch.tanh(raw_action).pow(2) + 1e-6)).sum(-1)

        dist_entropy = normal.entropy().sum(-1)

        return log_prob, dist_entropy













