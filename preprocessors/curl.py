import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from .base import Preprocessor
from architectures.pointnet import PointNet2SetEncoder
from architectures.main_architectures import MLP, CNN


class Curl(Preprocessor):

    def __init__(self, state_dim, img_dim, sound_dim, z_dim, action_dim, modalities, configs_rl, device):
        super(Curl, self).__init__(state_dim, img_dim, z_dim, modalities, device)

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
        self.phi = nn.ModuleDict(phi)

        self.momentum_phi = copy.deepcopy(self.phi)

        self.augmentations = {
            'state': self.augment_gaussian,
            'image': self.augment_crop,
            'depth': self.augment_crop,
            'sound': self.augment_gaussian,
            'pointcloud': self.augment_gaussian,
        }

    def get_representation(self, obs, past_state_action=None, target=False, phase="collect"):

        phi = self.phi if not target else self.momentum_phi

        z_dict = {}
        for mode, x in obs.items():
            if mode in self.phi.keys():
                x = self.augmentations[mode](x, test=phase!="train")
                z_dict[mode] = phi[mode](x)
        return torch.stack([z_dict[m] for m in sorted(z_dict.keys())], 1).mean(1)

    def get_loss(self, obs, actions, next_obs):

        z_q = self.get_representation(obs, phase="train")
        z_k = self.get_representation(obs, target=True, phase="train").detach()
        next_z_q = self.get_representation(next_obs, phase="train")

        logits = torch.matmul(F.normalize(z_q, dim=1), F.normalize(z_k, dim=1).t()) / self.temperature
        batch_size = z_q.size(0)
        labels = torch.arange(batch_size, dtype=torch.long, device=z_q.device)
        curl_loss = F.cross_entropy(logits, labels)

        return curl_loss, z_q, next_z_q

    def cleanup(self):
        with torch.no_grad():
            m = 0.99
            for param_q, param_k in zip(self.phi.parameters(), self.momentum_phi.parameters()):
                param_k.data = param_k.data * m + param_q.data * (1.0 - m)






