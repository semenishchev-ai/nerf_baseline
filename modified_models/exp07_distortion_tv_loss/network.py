"""
EXP-07: Opacity Regularization + Total Variation Loss
Adds opacity and sparsity regularization to the training loss.
Source: Barron et al., "Mip-NeRF 360" (CVPR 2022) — distortion/opacity loss;
TV regularization — standard (TensoRF, DVGO). Not integrated in torch-ngp training loop.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys

_DIR = os.path.dirname(os.path.abspath(__file__))
_TORCH_NGP = os.path.join(_DIR, '..', '..', 'baseline_choice', 'torch_ngp')
sys.path.insert(0, _TORCH_NGP)
from encoding import get_encoder
from activation import trunc_exp
from nerf.renderer import NeRFRenderer

class NeRFNetwork(NeRFRenderer):
    """Standard NeRFNetwork with opacity/TV regularization hooks.
    The aux_loss is computed inside run_cuda and stored as an attribute
    so the custom Trainer can add it to the main loss.
    """

    def __init__(self,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 encoding_bg="hashgrid",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 bound=1,
                 **kwargs):
        super().__init__(bound, **kwargs)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound)
        self.lambda_opacity = 1e-3
        self.lambda_sparsity = 1e-4
        self.aux_loss = torch.tensor(0.0)
        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim
            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim
            else:
                out_dim = hidden_dim
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))
        self.sigma_net = nn.ModuleList(sigma_net)
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding_dir)
        color_net = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_dir + self.geo_feat_dim
            else:
                in_dim = hidden_dim_color
            if l == num_layers_color - 1:
                out_dim = 3
            else:
                out_dim = hidden_dim_color
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))
        self.color_net = nn.ModuleList(color_net)
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder(encoding_bg, input_dim=2, num_levels=4, log2_hashmap_size=19, desired_resolution=2048)
            bg_net = []
            for l in range(num_layers_bg):
                if l == 0:
                    in_dim = self.in_dim_bg + self.in_dim_dir
                else:
                    in_dim = hidden_dim_bg
                if l == num_layers_bg - 1:
                    out_dim = 3
                else:
                    out_dim = hidden_dim_bg
                bg_net.append(nn.Linear(in_dim, out_dim, bias=False))
            self.bg_net = nn.ModuleList(bg_net)
        else:
            self.bg_net = None

    def forward(self, x, d):
        x_enc = self.encoder(x, bound=self.bound)
        h = x_enc
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]
        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        color = torch.sigmoid(h)
        return sigma, color

    def density(self, x):
        x_enc = self.encoder(x, bound=self.bound)
        h = x_enc
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]
        return {'sigma': sigma, 'geo_feat': geo_feat}

    def background(self, x, d):
        h = self.encoder_bg(x)
        d = self.encoder_dir(d)
        h = torch.cat([d, h], dim=-1)
        for l in range(self.num_layers_bg):
            h = self.bg_net[l](h)
            if l != self.num_layers_bg - 1:
                h = F.relu(h, inplace=True)
        return torch.sigmoid(h)

    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device)
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]
        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        h = torch.sigmoid(h)
        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype)
        else:
            rgbs = h
        return rgbs

    def get_params(self, lr):
        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr},
        ]
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        return params
