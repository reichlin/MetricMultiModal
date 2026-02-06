import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import torchvision.transforms.v2 as tv2
from torchvision.transforms.functional import convert_image_dtype

import time

class Preprocessor(nn.Module):

    def __init__(self, state_dim, img_dim, z_dim, modalities, device):
        super(Preprocessor, self).__init__()

        self.device = device
        self.modalities = modalities
        self.z_dim = z_dim
        self.img_dim = img_dim

        self.noise_std = 0.1
        self.temperature = 0.2

        self.recurrent_model = False

        self.mean_obs = None
        self.var_obs = None

        self.crop_resize = tv2.RandomResizedCrop(size=(img_dim), scale=(0.84, 0.84))

    def preprocess(self, obs):

        new_obs = {}

        if 'image' in obs.keys() and obs['image'] is not None:
            new_obs['image'] = obs['image'][None] if (obs['image'].ndim % 2 == 1) else obs['image']
            new_obs['image'] = torch.from_numpy(new_obs['image'])
            new_obs['image'] = new_obs['image'].to(self.device).float().div_(255) if self.device is not None else new_obs['image'].float().div_(255)

        if 'depth' in obs.keys() and obs['depth'] is not None:
            new_obs['depth'] = obs['depth'][None] if (obs['depth'].ndim % 2 == 1) else obs['depth']
            new_obs['depth'] = torch.from_numpy(new_obs['depth'])
            new_obs['depth'] = new_obs['depth'].to(self.device).float().div_(255) if self.device is not None else new_obs['depth'].float().div_(255)

        if 'state' in obs.keys() and obs['state'] is not None:
            new_obs['state'] = obs['state'][None] if (obs['state'].ndim % 2 == 1) else obs['state']
            new_obs['state'] = torch.from_numpy(new_obs['state']).float()
            new_obs['state'] = new_obs['state'].to(self.device) if self.device is not None else new_obs['state']

        if 'pointcloud' in obs.keys() and obs['pointcloud'] is not None:
            pc = obs['pointcloud'][None] if (obs['pointcloud'].ndim % 2 == 1) else obs['pointcloud']
            pc_rgb = obs['pc_rgb'][None] if (obs['pc_rgb'].ndim % 2 == 1) else obs['pc_rgb']

            new_obs['pointcloud'] = {"pc": torch.from_numpy(pc).float().to(self.device),
                                     "pc_rgb": torch.from_numpy(pc_rgb).to(self.device).float().div_(255)}

        if 'sound' in obs.keys() and obs['sound'] is not None:
            new_obs['sound'] = obs['sound'][None] if (obs['sound'].ndim % 2 == 1) else obs['sound']
            new_obs['sound'] = torch.from_numpy(new_obs['sound']).float()
            new_obs['sound'] = new_obs['sound'].to(self.device) if self.device is not None else new_obs['sound']

        return new_obs

    def get_representation(self, obs, past_state_action=None, phase="collect"):
        raise NotImplementedError("Not implemented")

    def get_loss(self, obs, actions, next_obs):
        return torch.zeros((), device=self.device), \
               self.get_representation(obs, past_state_action=None, phase="train"), \
               self.get_representation(next_obs, past_state_action=None, phase="train")

    def augment_gaussian(self, x, test=False):
        if type(x) == list or type(x) == dict:
            return {'pc': x['pc'] + torch.randn_like(x['pc']) * self.noise_std if not test else x['pc'], 'pc_rgb': x['pc_rgb']}
        return x + torch.randn_like(x) * self.noise_std if not test else x

    def augment_crop(self, x, test=False):

        if not test:
            resized = self.crop_resize(x)
        else:
            center = tv2.functional.center_crop(x, [int(0.84 * self.img_dim), int(0.84 * self.img_dim)])
            resized = tv2.functional.resize(center, size=[int(self.img_dim), int(self.img_dim)])

        return resized

    def cleanup(self):
        return









