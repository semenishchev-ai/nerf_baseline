"""
EXP-09: Progressive Frequency Masking (PFM)
Gradually "fades in" higher-resolution levels of the hash grid during the first 5000 iterations.
Rationale: Helps the model converge on stable coarse geometry before being "distracted" by 
high-frequency details.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys

_DIR = os.path.dirname(os.path.abspath(__file__))
_TORCH_NGP = os.path.join(_DIR, '..', '..', 'baseline_choice', 'torch_ngp')
sys.path.insert(0, _TORCH_NGP)

from encoding import get_encoder
from activation import trunc_exp
from nerf.renderer import NeRFRenderer

class NeRFNetwork(NeRFRenderer):
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
        self.num_levels = self.encoder.num_levels
        self.level_dim = self.encoder.level_dim
        
        # Tracking step for progressive masking
        self.register_buffer('step', torch.tensor(0, dtype=torch.long))
        self.max_step = 5000 # Iterations to fully enable all levels

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

    def _apply_progressive_mask(self, encoded):
        """Gradually enable higher levels."""
        if self.training:
            self.step += 1
        
        # alpha grows from 1 to num_levels
        # at step 0, alpha = 1 (only first level)
        # at step max_step, alpha = num_levels (all levels)
        alpha = 1.0 + (self.num_levels - 1) * min(self.step.item() / self.max_step, 1.0)
        
        # Mask for each level
        # levels: [num_levels]
        levels = torch.arange(self.num_levels, device=encoded.device) + 1
        weights = torch.clamp(alpha - levels + 1, 0, 1)
        
        # Reshape encoded to [B, num_levels, level_dim]
        B = encoded.shape[0]
        encoded = encoded.view(B, self.num_levels, self.level_dim)
        encoded = encoded * weights.view(1, self.num_levels, 1)
        
        return encoded.view(B, -1)

    def forward(self, x, d):
        encoded = self.encoder(x, bound=self.bound)
        encoded = self._apply_progressive_mask(encoded)

        h = encoded
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
        encoded = self.encoder(x, bound=self.bound)
        encoded = self._apply_progressive_mask(encoded)
        
        h = encoded
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
