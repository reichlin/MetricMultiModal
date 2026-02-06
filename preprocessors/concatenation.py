import torch
import torch.nn as nn

from .base import Preprocessor
from architectures.pointnet import PointNet2SetEncoder
from architectures.main_architectures import MLP, CNN


class ConCat(Preprocessor):

    def __init__(self, state_dim, img_dim, sound_dim, z_dim, action_dim, modalities, configs_rl, device):

        final_z_dim = z_dim * len(modalities)

        # if modalities == 'all' and not no_state:
        #     final_z_dim = z_dim * (n_mods+1)
        # elif modalities == 'all' and no_state:
        #     final_z_dim = z_dim * n_mods
        # else:
        #     final_z_dim = z_dim
        super(ConCat, self).__init__(state_dim, img_dim, final_z_dim, modalities, device)

        phi = {}
        if 'state' in modalities:
            phi['state'] = MLP(state_dim, z_dim, configs_rl['architecture'])
        if 'image' in modalities:
            phi['image'] = CNN(3*3, z_dim, configs_rl['architecture'])
        if 'depth' in modalities:
            phi['depth'] = CNN(1*3, z_dim, configs_rl['architecture'])
        if 'pointcloud' in modalities:
            phi['pointcloud'] = PointNet2SetEncoder(in_feat_dim=3, z_dim=z_dim, n_frames=3)
        if 'sound' in modalities:
            phi['sound'] = MLP(sound_dim, z_dim, configs_rl['architecture'])

        self.phi = nn.ModuleDict(phi)

    def get_representation(self, obs, past_state_action=None, phase="collect"):
        z_dict = {}
        for mode, x in obs.items():
            if mode in self.phi.keys():
                z_dict[mode] = self.phi[mode](x)
        return torch.cat([z_dict[m] for m in sorted(z_dict.keys())], -1)











