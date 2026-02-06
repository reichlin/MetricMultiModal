import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Preprocessor
from architectures.pointnet import PointNet2SetEncoder
from architectures.main_architectures import MLP, CNN


class GMC(Preprocessor):

    def __init__(self, state_dim, img_dim, sound_dim, z_dim, action_dim, modalities, configs_rl, device):
        super(GMC, self).__init__(state_dim, img_dim, z_dim, modalities, device)

        self.temperature = 0.3

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
        if 'depth' in modalities:
            phi['joint_img'] = CNN(4 * 3, z_dim, configs_rl['architecture'])
        if 'state' in modalities or 'sound' in modalities:
            phi['joint_mlp'] = MLP(z_dim, z_dim, configs_rl['architecture'])
        phi['proj'] = MLP(z_dim, z_dim, configs_rl['architecture'])
        self.phi = nn.ModuleDict(phi)

        self.augmentations = {
            'state': self.augment_gaussian,
            'image': self.augment_crop,
            'depth': self.augment_crop,
            'sound': self.augment_gaussian,
            'pointcloud': self.augment_gaussian,
        }


    def get_representation(self, obs, past_state_action=None, phase="collect"):

        z_dict = {}
        for mode, x in obs.items():
            if mode in self.phi.keys():
                x = self.augmentations[mode](x, test=phase!="train")
                z_dict[mode] = self.phi['proj'](self.phi[mode](x))
        z = torch.stack([z_dict[m] for m in sorted(z_dict.keys())], 0)

        if 'joint_img' in self.phi.keys():
            h = self.phi['joint_img'](torch.cat([obs['image'].to(self.device),
                                                         obs['depth'].to(self.device)], 1))
        else:
            h = self.phi['image'](obs['image'].to(self.device))

        if 'sound' in obs.keys():
            joint_z = self.phi['joint_mlp'](torch.cat([h, obs['sound'].to(self.device)], -1))
        elif 'joint_mlp' in self.phi.keys():
            joint_z = self.phi['joint_mlp'](torch.cat([h,  obs['state'].to(self.device)], -1))
        else:
            joint_z = h
        joint_z = self.phi['proj'](joint_z)

        if phase == 'test':
            return joint_z
        elif phase == 'collect':
            return joint_z

        return z, joint_z

    def get_loss(self, obs, actions, next_obs):

        z, joint_z = self.get_representation(obs, phase="train")
        next_z, next_joint_z = self.get_representation(next_obs, phase="train")

        B = z.shape[1]

        gmc_loss = 0.0
        for z_i in z:
            z_i_norm = F.normalize(z_i, dim=-1)
            joint_z_norm = F.normalize(joint_z, dim=-1)
            sim_matrix_joint_1 = torch.mm(z_i_norm, joint_z_norm.t()) / self.temperature
            sim_matrix_joint_2 = torch.mm(joint_z_norm, z_i_norm.t()) / self.temperature
            gmc_loss += 0.5 * F.cross_entropy(sim_matrix_joint_1, torch.arange(B, device=z_i.device))
            gmc_loss += 0.5 * F.cross_entropy(sim_matrix_joint_2, torch.arange(B, device=z_i.device))

        return gmc_loss, joint_z, next_joint_z



























