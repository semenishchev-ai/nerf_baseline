import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from encoding import get_encoder
from activation import trunc_exp
from nerf.renderer import NeRFRenderer

import raymarching


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 resolution=[128] * 3,
                 sigma_rank=[96] * 3,                                                                                            
                 color_rank=[288] * 3,
                 color_feat_dim=27,
                 num_layers=3,
                 hidden_dim=128,
                 bound=1,
                 **kwargs
                 ):
        super().__init__(bound, **kwargs)

        self.resolution = resolution

                                     
        self.sigma_rank = sigma_rank
        self.color_rank = color_rank
        self.color_feat_dim = color_feat_dim

        self.mat_ids = [[0, 1], [0, 2], [1, 2]]
        self.vec_ids = [2, 1, 0]

        self.sigma_vec = self.init_one_svd(self.sigma_rank, self.resolution)
        self.color_vec = self.init_one_svd(self.color_rank, self.resolution)
        self.basis_mat = nn.Linear(self.color_rank[0], self.color_feat_dim, bias=False)

                                                         
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.encoder, enc_dim = get_encoder('frequency', input_dim=color_feat_dim, multires=2)
        self.encoder_dir, enc_dim_dir = get_encoder('frequency', input_dim=3, multires=2)

        self.in_dim = enc_dim + enc_dim_dir

        color_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = self.hidden_dim
            
            if l == num_layers - 1:
                out_dim = 3      
            else:
                out_dim = self.hidden_dim
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)


    def init_one_svd(self, n_component, resolution, scale=0.2):

        vec = []

        for i in range(len(self.vec_ids)):
            vec_id = self.vec_ids[i]
            vec.append(torch.nn.Parameter(scale * torch.randn((1, n_component[i], resolution[vec_id], 1))))                                            

        return torch.nn.ParameterList(vec)


    def get_sigma_feat(self, x):
                               

        N = x.shape[0]

                    
        vec_coord = torch.stack((x[..., self.vec_ids[0]], x[..., self.vec_ids[1]], x[..., self.vec_ids[2]]))
        vec_coord = torch.stack((torch.zeros_like(vec_coord), vec_coord), dim=-1).view(3, -1, 1, 2)                              

        vec_feat = F.grid_sample(self.sigma_vec[0], vec_coord[[0]], align_corners=True).view(-1, N) * \
                   F.grid_sample(self.sigma_vec[1], vec_coord[[1]], align_corners=True).view(-1, N) * \
                   F.grid_sample(self.sigma_vec[2], vec_coord[[2]], align_corners=True).view(-1, N)         

        sigma_feat = torch.sum(vec_feat, dim=0)

        return sigma_feat


    def get_color_feat(self, x):
                               

        N = x.shape[0]

                    
        vec_coord = torch.stack((x[..., self.vec_ids[0]], x[..., self.vec_ids[1]], x[..., self.vec_ids[2]]))
        vec_coord = torch.stack((torch.zeros_like(vec_coord), vec_coord), dim=-1).view(3, -1, 1, 2)                              

        vec_feat = F.grid_sample(self.color_vec[0], vec_coord[[0]], align_corners=True).view(-1, N) * \
                   F.grid_sample(self.color_vec[1], vec_coord[[1]], align_corners=True).view(-1, N) * \
                   F.grid_sample(self.color_vec[2], vec_coord[[2]], align_corners=True).view(-1, N)         

        color_feat = self.basis_mat(vec_feat.T)                                 

        return color_feat
    
    
    def forward(self, x, d):
                                       
                                         

                                                
        x = 2 * (x - self.aabb_train[:3]) / (self.aabb_train[3:] - self.aabb_train[:3]) - 1

               
        sigma_feat = self.get_sigma_feat(x)
        sigma = trunc_exp(sigma_feat)

             
        color_feat = self.get_color_feat(x)
        enc_color_feat = self.encoder(color_feat)
        enc_d = self.encoder_dir(d)

        h = torch.cat([enc_color_feat, enc_d], dim=-1)
        for l in range(self.num_layers):
            h = self.color_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)
        
                                    
        rgb = torch.sigmoid(h)

        return sigma, rgb


    def density(self, x):
                                       

                                                
        x = 2 * (x - self.aabb_train[:3]) / (self.aabb_train[3:] - self.aabb_train[:3]) - 1

        sigma_feat = self.get_sigma_feat(x)
        sigma = trunc_exp(sigma_feat)

        return {
            'sigma': sigma,
        }

                            
    def color(self, x, d, mask=None, **kwargs):
                                      
                                                                             

                                                
        x = 2 * (x - self.aabb_train[:3]) / (self.aabb_train[3:] - self.aabb_train[:3]) - 1

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device)         
                                   
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]

        color_feat = self.get_color_feat(x)
        color_feat = self.encoder(color_feat)
        d = self.encoder_dir(d)

        h = torch.cat([color_feat, d], dim=-1)
        for l in range(self.num_layers):
            h = self.color_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)
        
                                    
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype)
        else:
            rgbs = h

        return rgbs


                         
    def density_loss(self):
        loss = 0
        for i in range(len(self.sigma_vec)):
            loss = loss + torch.mean(torch.abs(self.sigma_vec[i]))
        return loss
    
                    
    @torch.no_grad()
    def upsample_params(self, vec, resolution):

        for i in range(len(self.vec_ids)):
            vec_id = self.vec_ids[i]
            vec[i] = torch.nn.Parameter(F.interpolate(vec[i].data, size=(resolution[vec_id], 1), mode='bilinear', align_corners=True))


    @torch.no_grad()
    def upsample_model(self, resolution):
        self.upsample_params(self.sigma_vec, resolution)
        self.upsample_params(self.color_vec, resolution)
        self.resolution = resolution

    @torch.no_grad()
    def shrink_model(self):

        half_grid_size = self.bound / self.grid_size
        thresh = min(self.density_thresh, self.mean_density)

                                                                                                       
        valid_grid = self.density_grid[self.cascade - 1] > thresh      
        valid_pos = raymarching.morton3D_invert(torch.nonzero(valid_grid))                                  

                                                                                                  
        valid_pos = (2 * valid_pos / (self.grid_size - 1) - 1) * (self.bound - half_grid_size)                              
        min_pos = valid_pos.amin(0) - half_grid_size      
        max_pos = valid_pos.amax(0) + half_grid_size      

                      
        reso = torch.LongTensor(self.resolution).to(self.aabb_train.device)
        units = (self.aabb_train[3:] - self.aabb_train[:3]) / reso
        tl = (min_pos - self.aabb_train[:3]) / units
        br = (max_pos - self.aabb_train[:3]) / units
        tl = torch.round(tl).long().clamp(min=0)
        br = torch.minimum(torch.round(br).long(), reso)
        
        for i in range(len(self.vec_ids)):
            vec_id = self.vec_ids[i]

            self.sigma_vec[i] = nn.Parameter(self.sigma_vec[i].data[..., tl[vec_id]:br[vec_id], :])
            self.color_vec[i] = nn.Parameter(self.color_vec[i].data[..., tl[vec_id]:br[vec_id], :])
        
        self.aabb_train = torch.cat([min_pos, max_pos], dim=0)      

        print(f'[INFO] shrink slice: {tl.cpu().numpy().tolist()} - {br.cpu().numpy().tolist()}')
        print(f'[INFO] new aabb: {self.aabb_train.cpu().numpy().tolist()}')

                     
    def get_params(self, lr1, lr2):
        return [
            {'params': self.sigma_vec, 'lr': lr1},
            {'params': self.color_vec, 'lr': lr1},
            {'params': self.basis_mat.parameters(), 'lr': lr2},
            {'params': self.color_net.parameters(), 'lr': lr2},
        ]
        