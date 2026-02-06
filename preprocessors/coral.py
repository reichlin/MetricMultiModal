import torch
import torch.nn as nn

from .base import Preprocessor
from architectures.pointnet import PointNet2SetEncoder
from architectures.main_architectures import MLP, CNN


def info_nce_loss(z, z_i, temperature=0.1):

    sim = torch.mm(z, z_i.t()) / temperature
    diag = torch.arange(z.shape[0], device=z.device)
    pos = sim[diag, diag]
    neg = torch.logsumexp(sim, dim=-1)
    nce = - torch.mean(pos - neg)

    return nce


class CORAL(Preprocessor):

    def __init__(self, state_dim, img_dim, sound_dim, z_dim, action_dim, modalities, configs_rl, device):
        super(CORAL, self).__init__(state_dim, img_dim, z_dim, modalities, device)

        self.action_dim = action_dim

        phi = {}
        phi_inv = {}

        self.contrastive_mods = ['image', 'depth']
        self.reconstruction_mods = ['state'] if 'state' in modalities else []

        n = 0
        if 'state' in modalities:
            phi['state'] = MLP(state_dim, z_dim, configs_rl['architecture'])
            phi_inv['state'] = MLP(z_dim, state_dim, configs_rl['architecture'])
            n += 1
        if 'image' in modalities:
            phi['image'] = CNN(3 * 3, z_dim, configs_rl['architecture'])
            phi_inv['image'] = nn.Linear(z_dim, z_dim, bias=False)
            n += 1
        if 'depth' in modalities:
            phi['depth'] = CNN(1 * 3, z_dim, configs_rl['architecture'])
            phi_inv['depth'] = nn.Linear(z_dim, z_dim, bias=False)
            n += 1
        if 'pointcloud' in modalities:
            phi['pointcloud'] = PointNet2SetEncoder(in_feat_dim=3, z_dim=z_dim, n_frames=3)
            phi_inv['pointcloud'] = nn.Linear(z_dim, z_dim, bias=False)
            n += 1
        if 'sound' in modalities:
            phi['sound'] = MLP(sound_dim, z_dim, configs_rl['architecture'])
            n += 1

        phi['g'] = MLP(z_dim * (n+1) + action_dim, z_dim, configs_rl['architecture'])
        phi['transition'] = MLP(z_dim + action_dim, z_dim, configs_rl['architecture'])
        phi_inv['repr_proj'] = nn.Linear(z_dim, z_dim, bias=False)

        self.phi = nn.ModuleDict(phi)
        self.phi_inv = nn.ModuleDict(phi_inv)

        self.sm = torch.nn.Softmax(dim=1)
        self.mse = nn.MSELoss()

        self.recurrent_model = True

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
        z_mod = torch.stack([z_dict[m] for m in sorted(z_dict.keys())], 1)

        if past_state_action is not None:
            z_past = past_state_action['z'].to(self.device)
            a_past = past_state_action['a'].to(self.device)
        else:
            z_past = z_mod.mean(1)
            a_past = torch.zeros(z_mod.shape[0], self.action_dim).float().to(self.device)

        if self.use_idw:
            z_pred = self.phi['transition'](torch.cat([z_past, a_past], dim=-1))
            d = torch.linalg.norm(z_mod - torch.unsqueeze(z_pred, 1), dim=-1)
            d_inv = 1.0 / (d + 1e-6)
            coeff = d_inv / torch.sum(d_inv, dim=-1, keepdim=True)
            z = torch.sum(torch.unsqueeze(coeff, -1) * z_mod, dim=1)
            return z

        z = self.phi['g'](
            torch.cat([z_mod.view(z_mod.shape[0], -1), z_past, a_past], dim=-1)
        )

        if phase == 'test':
            return z
        elif phase == 'collect':
            return z

        return z, (z_dict)

    def get_loss(self, obs, actions, next_obs):

        z_t, (zt_mod_dict) = self.get_representation(obs, past_state_action=None, phase="train")
        past_state_action = {
            'z': z_t,
            'a': actions
        }
        z_t1, (zt1_mod_dict) = self.get_representation(next_obs, past_state_action=past_state_action, phase="train")

        total_recon_loss = 0.
        total_contr_loss = 0.

        for m in self.reconstruction_mods:
            x_hat = self.phi_inv[m](z_t)
            recon_loss = self.mse(x_hat, obs[m].to(self.device))
            total_recon_loss += recon_loss

        z_t1_hat = self.phi['transition'](
            torch.cat([z_t, actions], dim=-1)
        )
        z_t1_hat_proj = self.phi_inv['repr_proj'](z_t1_hat)

        for m in self.contrastive_mods:
            z_i = zt1_mod_dict[m]
            z_i_proj = self.phi_inv[m](z_i)
            contr_loss = info_nce_loss(z_t1_hat_proj, z_i_proj, temperature=self.temperature)
            total_contr_loss += contr_loss

        coral_loss = total_recon_loss + total_contr_loss  # + reward_loss_avg

        return coral_loss, z_t, z_t1


























