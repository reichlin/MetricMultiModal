import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Preprocessor
from architectures.pointnet import PointNet2SetEncoder
from architectures.main_architectures import MLP, CNN


class AMDF(Preprocessor):

    def __init__(self, state_dim, img_dim, sound_dim, z_dim, action_dim, modalities, configs_rl, device):
        super(AMDF, self).__init__(state_dim, img_dim, z_dim, modalities, device)

        self.recurrent_model = True
        self.z_dim = z_dim

        self.temperature = 0.5

        phi = {}
        n = 0
        if 'state' in modalities:
            phi['state'] = MLP(state_dim, z_dim, configs_rl['architecture'])
            n += 1
        if 'image' in modalities:
            phi['image'] = CNN(3 * 3, z_dim, configs_rl['architecture'])
            n += 1
        if 'depth' in modalities:
            phi['depth'] = CNN(1 * 3, z_dim, configs_rl['architecture'])
            n += 1
        if 'pointcloud' in modalities:
            phi['pointcloud'] = PointNet2SetEncoder(in_feat_dim=3, z_dim=z_dim, n_frames=3)
            n += 1
        if 'sound' in modalities:
            phi['sound'] = MLP(sound_dim, z_dim, configs_rl['architecture'])
            n += 1

        phi['transition'] = MLP(z_dim + action_dim, z_dim, configs_rl['architecture'])
        phi['K'] = nn.Linear(z_dim, z_dim, bias=False)
        phi['V'] = nn.Linear(z_dim, z_dim, bias=False)
        phi['Q'] = nn.Linear(1, 1 * z_dim, bias=False)
        phi['norm'] = nn.LayerNorm(z_dim)

        self.phi = nn.ModuleDict(phi)
        self.phi_inv = MLP(self.z_dim, state_dim, configs_rl['architecture'])

        self.sm = torch.nn.Softmax(dim=1)
        self.mse = nn.MSELoss()

        self.augmentations = {
            'state': self.augment_gaussian,
            'image': self.augment_crop,
            'depth': self.augment_crop,
            'sound': self.augment_gaussian,
            'pointcloud': self.augment_gaussian,
        }

        self.use_idw = False

    def get_representation(self, obs, past_state_action=None, phase="collect"):

        z_dict = {}
        for mode, x in obs.items():
            if mode in self.phi.keys():
                x = self.augmentations[mode](x, test=phase!="train")
                z_dict[mode] = self.phi[mode](x)
        z_obs = torch.stack([z_dict[m] for m in sorted(z_dict.keys())], 1)

        if past_state_action is not None:
            past_z = torch.unsqueeze(past_state_action['z'], 1).to(self.device)
            past_a = torch.unsqueeze(past_state_action['a'], 1).to(self.device)
            z_pred = past_z + self.phi['transition'](torch.cat([past_z, past_a], dim=-1))
        else:
            z_pred = z_obs.mean(dim=1, keepdim=True)

        if self.use_idw:
            # maybe use self.phi['norm']
            all_z = torch.cat([z_pred, z_obs], 1)
            all_z = self.phi['norm'](all_z)
            d = torch.linalg.norm(all_z[0, 0:1] - all_z[0, 1:], dim=-1)
            d_inv = 1.0 / (d + 1e-6)
            coeff = d_inv / torch.sum(d_inv, dim=-1, keepdim=True)
            z = torch.sum(torch.unsqueeze(coeff, -1) * all_z[:, 1:], dim=1)
            return z

        all_z = torch.cat([z_pred, z_obs], 1)

        z = self.phi['norm'](all_z)                 # stable scale
        K = self.phi['K'](z)
        V = self.phi['V'](z)

        q = self.phi['Q'].weight.view(1, self.z_dim)
        q = F.normalize(q, dim=-1)      # (H,D)
        K_n = F.normalize(K, dim=-1)         # (B,N,D)
        logits = (K_n @ q.T) / self.temperature      # (B,N,H)

        weights = F.softmax(logits, dim=1)

        z_agg = torch.einsum('bnh,bnd->bhd', weights, V)     # (B,H,D)
        z_agg = z_agg.mean(dim=1)

        if phase == 'test':
            return z_agg
        elif phase == 'collect':
            return z_agg

        return z_agg, (z_pred, z_dict)

    def get_loss(self, obs, actions, next_obs):

        z_t, _ = self.get_representation(obs, past_state_action=None, phase="train")
        past_state_action = {
            'z': z_t,
            'a': actions
        }

        z_t1, (z_pred, z_obs) = self.get_representation(next_obs, past_state_action=past_state_action, phase="train")

        loss_dec = 0.0
        x_s = next_obs['state'].to(self.device)
        for i, (mode, z_i) in enumerate(z_obs.items()):
            x_hat_i = self.phi_inv(z_i)
            loss_dec += self.mse(x_hat_i, x_s)
        loss_dec /= len(z_obs)

        # e2e
        loss_e2e = self.mse(self.phi_inv(z_t1), x_s)
        loss_tf = self.mse(self.phi_inv(z_pred).sum(1), x_s)

        amdf_loss = loss_dec + loss_e2e + loss_tf + 0.001 * (torch.linalg.norm(z_t, dim=-1).mean() + torch.linalg.norm(z_t1, dim=-1).mean())

        return amdf_loss, z_t, z_t1






























