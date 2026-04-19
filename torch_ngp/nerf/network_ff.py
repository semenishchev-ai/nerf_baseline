import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from ffmlp import FFMLP

from .renderer import NeRFRenderer

class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 bound=1,
                 **kwargs
                 ):
        super().__init__(bound, **kwargs)

                       
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound)

        self.sigma_net = FFMLP(
            input_dim=self.in_dim, 
            output_dim=1 + self.geo_feat_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )

                       
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_color = get_encoder(encoding_dir)
        self.in_dim_color += self.geo_feat_dim + 1                                                               
        
        self.color_net = FFMLP(
            input_dim=self.in_dim_color, 
            output_dim=3,
            hidden_dim=self.hidden_dim_color,
            num_layers=self.num_layers_color,
        )
    
    def forward(self, x, d):
                                       
                                         

               
        x = self.encoder(x, bound=self.bound)
        h = self.sigma_net(x)

                                  
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

                       
        d = self.encoder_dir(d)

                                                     
        p = torch.zeros_like(geo_feat[..., :1])                       
        h = torch.cat([d, geo_feat, p], dim=-1)
        h = self.color_net(h)
        
                                    
        rgb = torch.sigmoid(h)

        return sigma, rgb

    def density(self, x):
                                       

        x = self.encoder(x, bound=self.bound)
        h = self.sigma_net(x)

                                  
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }

                            
    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
                                      
                                                                             

                                                                                                    
                         

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device)         
                                   
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

                                       

                                                                                                                        
                         

        d = self.encoder_dir(d)

        p = torch.zeros_like(geo_feat[..., :1])                       
        h = torch.cat([d, geo_feat, p], dim=-1)

        h = self.color_net(h)
        
                                    
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