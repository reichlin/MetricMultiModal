import torch
import torch.nn as nn

from .base import Preprocessor
from architectures.pointnet import PointNet2SetEncoder
from architectures.main_architectures import MLP, CNN


class MMM(Preprocessor):

    def __init__(self, state_dim, img_dim, sound_dim, z_dim, action_dim, modalities, configs_rl, device):
        super(MMM, self).__init__(state_dim, img_dim, z_dim, modalities, device)

        self.recurrent_model = True
        self.z_dim = z_dim
        self.action_dim = action_dim

        self.use_tr_only = False

        phi = {}
        if 'state' in modalities:
            phi['state'] = MLP(state_dim, z_dim, configs_rl['architecture'])
        if 'image' in modalities:
            phi['image'] = CNN(3 * 3, z_dim, configs_rl['architecture'])
        if 'depth' in modalities:
            phi['depth'] = CNN(1 * 3, z_dim, configs_rl['architecture'])
        if 'pointcloud' in modalities:
            phi['pointcloud'] = PointNet2SetEncoder(in_feat_dim=3, z_dim=z_dim, n_frames=3)
        if 'sound' in modalities:
            phi['sound'] = MLP(sound_dim, z_dim, configs_rl['architecture'])

        phi['transition'] = MLP(z_dim + action_dim, z_dim, configs_rl['architecture'])

        self.phi = nn.ModuleDict(phi)

        self.mse = nn.MSELoss()


    def get_representation(self, obs, past_state_action=None, phase="collect"):
        z_dict = {}
        for mode, x in obs.items():
            if obs[mode] is None:
                continue
            if type(obs[mode]) == dict and obs[mode]['pc'] is None:
                continue
            if mode in self.phi.keys():
                z_dict[mode] = self.phi[mode](x)
        z_obs = torch.stack([z_dict[m] for m in sorted(z_dict.keys())], 1)
        z_bar = torch.mean(z_obs, 1)

        if past_state_action is not None:
            z_past = past_state_action['z']
            a_past = past_state_action['a']
        else:
            z_past = z_bar
            a_past = torch.zeros(z_bar.shape[0], self.action_dim).float().to(self.device)

        z_hat = self.phi['transition'](
            torch.cat([z_past, a_past], dim=-1)
        )

        if self.use_tr_only:
            return z_hat

        if phase == 'test':
            d = torch.linalg.norm(z_obs - torch.unsqueeze(z_hat, 1), dim=-1)
            # print(d.mean().detach().cpu().item(), ",")
            d_inv = 1.0 / (d + 1e-6)
            coeff = d_inv / torch.sum(d_inv, dim=-1, keepdim=True)
            # coeff = torch.exp(d_inv*0.4) / torch.sum(torch.exp(d_inv*0.4), dim=-1, keepdim=True)
            z = torch.sum(torch.unsqueeze(coeff, -1) * z_obs, dim=1)

            # if past_state_action is not None:
            #     z = z_hat
            # else:
            #     z = torch.sum(torch.unsqueeze(coeff, -1) * z_obs, dim=1)

            # d = torch.sqrt((z_obs - torch.unsqueeze(z_hat, 1)) ** 2)
            # d_inv = 1.0 / (d + 1e-6)
            # coeff = d_inv / torch.sum(d_inv, dim=1, keepdim=True)
            # z = torch.sum(coeff * z_obs, dim=1)
            return z
        elif phase == 'collect':
            return z_bar

        return z_bar, (z_obs, z_hat)


    def get_loss(self, obs, actions, next_obs):
        z_t, (z_t_obs, z_t_hat) = self.get_representation(obs, past_state_action=None, phase="train")
        past_state_action = {
            'z': z_t,
            'a': actions
        }
        z_t1, (z_t1_obs, z_t1_hat) = self.get_representation(next_obs, past_state_action=past_state_action, phase="train")

        L_pos = torch.mean((torch.linalg.norm(z_t - z_t1, dim=-1) - 1.0) ** 2)
        L_neg = - torch.mean(torch.log(torch.cdist(z_t, z_t) + 1e-6)) - torch.mean(torch.log(torch.cdist(z_t1, z_t1) + 1e-6))
        L_trans = torch.mean(torch.linalg.norm(z_t1_hat - z_t1, dim=-1))
        L_inv = torch.cdist(z_t_obs, z_t_obs).mean() + torch.cdist(z_t1_obs, z_t1_obs).mean()

        mmm_loss = 1.0 * L_pos + 1.0 * L_neg + 1.0 * L_trans + 1.0 * L_inv # 10.0

        return mmm_loss, z_t, z_t1



























