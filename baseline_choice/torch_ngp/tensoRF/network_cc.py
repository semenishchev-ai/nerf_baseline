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
                 degree=4,
                                         
                                         
                                 
                                 
                 rank_vec_density=[64, 64, 64, 64, 64],
                 rank_mat_density=[0, 4, 8, 12, 16],
                 rank_vec=[64, 64, 64, 64, 64],
                 rank_mat=[0, 4, 16, 32, 64],
                 bg_resolution=[512, 512],
                 bg_rank=8,
                 bound=1,
                 **kwargs
                 ):
        super().__init__(bound, **kwargs)

        self.resolution = resolution

        self.degree = degree
        self.encoder_dir, self.enc_dir_dim = get_encoder('sphere_harmonics', degree=self.degree)
        self.out_dim = 3 * self.enc_dir_dim                 

                                            
        self.rank_vec_density = [rank_vec_density]
        self.rank_mat_density = [rank_mat_density]
        self.rank_vec = [rank_vec]
        self.rank_mat = [rank_mat]

                                                  
        assert len(rank_vec) == len(rank_mat) == len(rank_vec_density) == len(rank_mat_density)

        self.K = [len(rank_vec)]

                 
        self.group_vec_density = [np.diff(rank_vec_density, prepend=0)]
        self.group_mat_density = [np.diff(rank_mat_density, prepend=0)]
        self.group_vec = [np.diff(rank_vec, prepend=0)]
        self.group_mat = [np.diff(rank_mat, prepend=0)]

        self.mat_ids = [[0, 1], [0, 2], [1, 2]]
        self.vec_ids = [2, 1, 0]

                         

        self.U_vec_density = nn.ParameterList() 
        self.S_vec_density = nn.ParameterList()

        for k in range(self.K[0]):
            if self.group_vec_density[0][k] > 0:
                for i in range(3):                
                    vec_id = self.vec_ids[i]
                    w = torch.randn(self.group_vec_density[0][k], self.resolution[vec_id]) * 0.2         
                    self.U_vec_density.append(nn.Parameter(w.view(1, self.group_vec_density[0][k], self.resolution[vec_id], 1)))               
                w = torch.ones(1, self.group_vec_density[0][k])
                torch.nn.init.kaiming_normal_(w)
                self.S_vec_density.append(nn.Parameter(w))

        self.U_mat_density = nn.ParameterList() 
        self.S_mat_density = nn.ParameterList()

        
        for k in range(self.K[0]):
            if self.group_mat_density[0][k] > 0:
                for i in range(3):
                    mat_id_0, mat_id_1 = self.mat_ids[i]
                    w = torch.randn(self.group_mat_density[0][k], self.resolution[mat_id_1] * self.resolution[mat_id_0]) * 0.2          
                    self.U_mat_density.append(nn.Parameter(w.view(1, self.group_mat_density[0][k], self.resolution[mat_id_1], self.resolution[mat_id_0])))               
                w = torch.ones(1, self.group_mat_density[0][k])
                torch.nn.init.kaiming_normal_(w)
                self.S_mat_density.append(nn.Parameter(w))

        self.U_vec = nn.ParameterList() 
        self.S_vec = nn.ParameterList()

        for k in range(self.K[0]):
            if self.group_vec[0][k] > 0:
                for i in range(3):                
                    vec_id = self.vec_ids[i]
                    w = torch.randn(self.group_vec[0][k], self.resolution[vec_id]) * 0.2         
                    self.U_vec.append(nn.Parameter(w.view(1, self.group_vec[0][k], self.resolution[vec_id], 1)))               
                w = torch.ones(self.out_dim, self.group_vec[0][k])
                torch.nn.init.kaiming_normal_(w)
                self.S_vec.append(nn.Parameter(w))

        self.U_mat = nn.ParameterList() 
        self.S_mat = nn.ParameterList()

        for k in range(self.K[0]):
            if self.group_mat[0][k] > 0:
                for i in range(3):
                    mat_id_0, mat_id_1 = self.mat_ids[i]
                    w = torch.randn(self.group_mat[0][k], self.resolution[mat_id_1] * self.resolution[mat_id_0]) * 0.2          
                    self.U_mat.append(nn.Parameter(w.view(1, self.group_mat[0][k], self.resolution[mat_id_1], self.resolution[mat_id_0])))               
                w = torch.ones(self.out_dim, self.group_mat[0][k])
                torch.nn.init.kaiming_normal_(w)
                self.S_mat.append(nn.Parameter(w))

              
        self.finalized = False if self.K[0] != 1 else True

                          
        if self.bg_radius > 0:
            
            self.bg_resolution = bg_resolution
            self.bg_rank = bg_rank
            self.bg_mat = nn.Parameter(0.2 * torch.randn((1, bg_rank, bg_resolution[0], bg_resolution[1])))               

            w = torch.ones(self.out_dim, bg_rank)             
            torch.nn.init.kaiming_normal_(w)
            self.bg_S = nn.Parameter(w)


    def compute_features_density(self, x, K=-1, residual=False, oid=0):
                               
                                 

        prefix = x.shape[:-1]
        N = np.prod(prefix)

        vec_coord = torch.stack((x[..., self.vec_ids[0]], x[..., self.vec_ids[1]], x[..., self.vec_ids[2]]))
        vec_coord = torch.stack((torch.zeros_like(vec_coord), vec_coord), dim=-1).view(3, -1, 1, 2)

        mat_coord = torch.stack((x[..., self.mat_ids[0]], x[..., self.mat_ids[1]], x[..., self.mat_ids[2]])).view(3, -1, 1, 2)               

                                  
        if K <= 0:
            K = self.K[oid]
            
                          
        if residual:
            outputs = []

        last_y = None

        offset_vec = oid
        offset_mat = oid

        for k in range(K):

            y = 0

            if self.group_vec_density[oid][k]:
                vec_feat = F.grid_sample(self.U_vec_density[3 * offset_vec + 0], vec_coord[[0]], align_corners=False).view(-1, N) * \
                           F.grid_sample(self.U_vec_density[3 * offset_vec + 1], vec_coord[[1]], align_corners=False).view(-1, N) * \
                           F.grid_sample(self.U_vec_density[3 * offset_vec + 2], vec_coord[[2]], align_corners=False).view(-1, N)         

                y = y + (self.S_vec_density[offset_vec] @ vec_feat)

                offset_vec += 1

            if self.group_mat_density[oid][k]:
                mat_feat = F.grid_sample(self.U_mat_density[3 * offset_mat + 0], mat_coord[[0]], align_corners=False).view(-1, N) * \
                           F.grid_sample(self.U_mat_density[3 * offset_mat + 1], mat_coord[[1]], align_corners=False).view(-1, N) * \
                           F.grid_sample(self.U_mat_density[3 * offset_mat + 2], mat_coord[[2]], align_corners=False).view(-1, N)         

                y = y + (self.S_mat_density[offset_mat] @ mat_feat)               

                offset_mat += 1

            if last_y is not None:
                y = y + last_y

            if residual:
                outputs.append(y)

            last_y = y
        
        if residual:
            outputs = torch.stack(outputs, dim=0).permute(0, 2, 1).contiguous().view(K, *prefix, -1)                                      
        else:
            outputs = last_y.permute(1, 0).contiguous().view(*prefix, -1)                                
        
        return outputs

    def compute_features(self, x, K=-1, residual=False, oid=0):
                               
                                 

        prefix = x.shape[:-1]
        N = np.prod(prefix)

        vec_coord = torch.stack((x[..., self.vec_ids[0]], x[..., self.vec_ids[1]], x[..., self.vec_ids[2]]))
        vec_coord = torch.stack((torch.zeros_like(vec_coord), vec_coord), dim=-1).view(3, -1, 1, 2)

        mat_coord = torch.stack((x[..., self.mat_ids[0]], x[..., self.mat_ids[1]], x[..., self.mat_ids[2]])).view(3, -1, 1, 2)               

                                  
        if K <= 0:
            K = self.K[oid]
            
                          
        if residual:
            outputs = []

        last_y = None

        offset_vec = oid
        offset_mat = oid

        for k in range(K):

            y = 0

            if self.group_vec[oid][k]:
                vec_feat = F.grid_sample(self.U_vec[3 * offset_vec + 0], vec_coord[[0]], align_corners=False).view(-1, N) * \
                           F.grid_sample(self.U_vec[3 * offset_vec + 1], vec_coord[[1]], align_corners=False).view(-1, N) * \
                           F.grid_sample(self.U_vec[3 * offset_vec + 2], vec_coord[[2]], align_corners=False).view(-1, N)         

                y = y + (self.S_vec[offset_vec] @ vec_feat)

                offset_vec += 1

            if self.group_mat[oid][k]:
                mat_feat = F.grid_sample(self.U_mat[3 * offset_mat + 0], mat_coord[[0]], align_corners=False).view(-1, N) * \
                           F.grid_sample(self.U_mat[3 * offset_mat + 1], mat_coord[[1]], align_corners=False).view(-1, N) * \
                           F.grid_sample(self.U_mat[3 * offset_mat + 2], mat_coord[[2]], align_corners=False).view(-1, N)         

                y = y + (self.S_mat[offset_mat] @ mat_feat)               

                offset_mat += 1

            if last_y is not None:
                y = y + last_y

            if residual:
                outputs.append(y)

            last_y = y
        
        if residual:
            outputs = torch.stack(outputs, dim=0).permute(0, 2, 1).contiguous().view(K, *prefix, -1)                                      
        else:
            outputs = last_y.permute(1, 0).contiguous().view(*prefix, -1)                                
        
        return outputs


    def normalize_coord(self, x, oid=0):
        
        if oid == 0:
            aabb = self.aabb_train
        else:
            tr = getattr(self, f'T_{oid}')                               
            x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)          
            x = (x @ tr.T)[:, :3]                    

            aabb = getattr(self, f'aabb_{oid}')

        return 2 * (x - aabb[:3]) / (aabb[3:] - aabb[:3]) - 1                  
            

    def normalize_dir(self, d, oid=0):
        if oid != 0:
            tr = getattr(self, f'R_{oid}')                         
            d = d @ tr.T
        return d

    
    def forward(self, x, d, K=-1):
                                       
                                         

        N = x.shape[0]

                       
        if len(self.K) == 1:

            x_model = self.normalize_coord(x)
            feats_density = self.compute_features_density(x_model, K, residual=self.training)            
            sigma = trunc_exp(feats_density).squeeze(-1)         

            enc_d = self.encoder_dir(d)         

            h = self.compute_features(x_model, K, residual=self.training)             
            h = h.view(K, N, 3, self.degree ** 2)               
            h = (h * enc_d.unsqueeze(1)).sum(-1)            

            rgb = torch.sigmoid(h)             

            return sigma, rgb

                                                                                       
        else:
            
            sigma_list = []
            h_list = []

            sigma_all = 0
            rgb_all = 0


            for oid in range(1, len(self.K)):
                x_model = self.normalize_coord(x, oid=oid)

                feats_density = self.compute_features_density(x_model, -1, residual=False, oid=oid)         

                sigma = trunc_exp(feats_density).squeeze(-1)      
                sigma_list.append(sigma.detach().clone())

                sigma_all += sigma

                d_model = self.normalize_dir(d, oid=oid)
                enc_d = self.encoder_dir(d_model)         

                h = self.compute_features(x_model, -1, residual=False, oid=oid)          
                h = h.view(N, 3, self.degree ** 2)
                h = (h * enc_d.unsqueeze(1)).sum(-1)         

                h_list.append(h)


            ws = torch.stack(sigma_list, dim=0)         
            ws = F.softmax(ws, dim=0)

            for oid in range(1, len(self.K)):
                rgb_all += h_list[oid - 1] * ws[oid - 1].unsqueeze(-1)

            rgb_all = torch.sigmoid(rgb_all)

            return sigma_all, rgb_all


    def density(self, x, K=-1):
                                       

        if len(self.K) == 1:
        
            x_model = self.normalize_coord(x)
            feats_density = self.compute_features_density(x_model, K, residual=False)              
            sigma = trunc_exp(feats_density).squeeze(-1)      

            return {
                'sigma': sigma,
            }

        else:

            sigma_all = 0
            for oid in range(1, len(self.K)):
                x_model = self.normalize_coord(x, oid=oid)
                feats_density = self.compute_features_density(x_model, -1, residual=False, oid=oid)         
                sigma = trunc_exp(feats_density).squeeze(-1)      
                sigma_all += sigma

            return {
                'sigma': sigma_all,
            }


    def background(self, x, d):
                              

        N = x.shape[0]

        h = F.grid_sample(self.bg_mat, x.view(1, N, 1, 2), align_corners=False).view(-1, N)         
        h = (self.bg_S @ h).T.contiguous()                      
        enc_d = self.encoder_dir(d)

        h = h.view(N, 3, -1)
        h = (h * enc_d.unsqueeze(1)).sum(-1)         
        
                                    
        rgb = torch.sigmoid(h)

        return rgb


                         
    def density_loss(self):
        loss = 0
        for i in range(len(self.U_vec_density)):
            loss = loss + torch.mean(torch.abs(self.U_vec_density[i]))
        for i in range(len(self.U_mat_density)):
            loss = loss + torch.mean(torch.abs(self.U_mat_density[i]))
        return loss
    

                    
    @torch.no_grad()
    def upsample_model(self, resolution):

        for i in range(len(self.U_vec_density)):
            vec_id = self.vec_ids[i % 3]
            self.U_vec_density[i] = nn.Parameter(F.interpolate(self.U_vec_density[i].data, size=(resolution[vec_id], 1), mode='bilinear', align_corners=False))

        for i in range(len(self.U_mat_density)):
            mat_id_0, mat_id_1 = self.mat_ids[i % 3]
            self.U_mat_density[i] = nn.Parameter(F.interpolate(self.U_mat_density[i].data, size=(resolution[mat_id_1], resolution[mat_id_0]), mode='bilinear', align_corners=False))

        for i in range(len(self.U_vec)):
            vec_id = self.vec_ids[i % 3]
            self.U_vec[i] = nn.Parameter(F.interpolate(self.U_vec[i].data, size=(resolution[vec_id], 1), mode='bilinear', align_corners=False))

        for i in range(len(self.U_mat)):
            mat_id_0, mat_id_1 = self.mat_ids[i % 3]
            self.U_mat[i] = nn.Parameter(F.interpolate(self.U_mat[i].data, size=(resolution[mat_id_1], resolution[mat_id_0]), mode='bilinear', align_corners=False))

        self.resolution = resolution

        print(f'[INFO] upsampled to {resolution}')

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
        
        for i in range(len(self.U_vec_density)):
            vec_id = self.vec_ids[i % 3]
            self.U_vec_density[i] = nn.Parameter(self.U_vec_density[i].data[..., tl[vec_id]:br[vec_id], :])
        
        for i in range(len(self.U_mat_density)):
            mat_id_0, mat_id_1 = self.mat_ids[i % 3]
            self.U_mat_density[i] = nn.Parameter(self.U_mat_density[i].data[..., tl[mat_id_1]:br[mat_id_1], tl[mat_id_0]:br[mat_id_0]])
        
        for i in range(len(self.U_vec)):
            vec_id = self.vec_ids[i % 3]
            self.U_vec[i] = nn.Parameter(self.U_vec[i].data[..., tl[vec_id]:br[vec_id], :])
        
        for i in range(len(self.U_mat)):
            mat_id_0, mat_id_1 = self.mat_ids[i % 3]
            self.U_mat[i] = nn.Parameter(self.U_mat[i].data[..., tl[mat_id_1]:br[mat_id_1], tl[mat_id_0]:br[mat_id_0]])
        
        self.aabb_train = torch.cat([min_pos, max_pos], dim=0)      

        print(f'[INFO] shrink slice: {tl.cpu().numpy().tolist()} - {br.cpu().numpy().tolist()}')
        print(f'[INFO] new aabb: {self.aabb_train.cpu().numpy().tolist()}')

    
    @torch.no_grad()
    def finalize_group(self, U, S):

        if len(U) == 0 or len(S) == 0:
            return nn.ParameterList(), nn.ParameterList()

                                     
        for i in range(len(S)):
            importance = S[i].abs().sum(0)                 
            for j in range(3):
                importance *= U[3 * i + j].view(importance.shape[0], -1).norm(dim=-1)                    
        
            inds = torch.argsort(importance, descending=True)                  

            S[i] = nn.Parameter(S[i].data[:, inds])
            for j in range(3):
                U[3 * i + j] = nn.Parameter(U[3 * i + j].data[:, inds])

                                     

        S = nn.ParameterList([
            nn.Parameter(torch.cat([s.data for s in S], dim=1))
        ])

        U = nn.ParameterList([
            nn.Parameter(torch.cat([v.data for v in U[0::3]], dim=1)),
            nn.Parameter(torch.cat([v.data for v in U[1::3]], dim=1)),
            nn.Parameter(torch.cat([v.data for v in U[2::3]], dim=1)),
        ])

        return U, S


                                                                                                                   
    @torch.no_grad()
    def finalize(self):
        self.U_vec_density, self.S_vec_density = self.finalize_group(self.U_vec_density, self.S_vec_density)
        self.U_mat_density, self.S_mat_density = self.finalize_group(self.U_mat_density, self.S_mat_density)
        self.U_vec, self.S_vec = self.finalize_group(self.U_vec, self.S_vec)
        self.U_mat, self.S_mat = self.finalize_group(self.U_mat, self.S_mat)

                               
        self.rank_vec_density[0] = [self.rank_vec_density[0][-1]]
        self.rank_mat_density[0] = [self.rank_mat_density[0][-1]]
        self.rank_vec[0] = [self.rank_vec[0][-1]]
        self.rank_mat[0] = [self.rank_mat[0][-1]]

        self.group_vec_density[0] = self.rank_vec_density[0]
        self.group_mat_density[0] = self.rank_mat_density[0]
        self.group_vec[0] = self.rank_vec[0]
        self.group_mat[0] = self.rank_mat[0]

        self.K[0] = 1

        self.finalized = True

    
                                                
    @torch.no_grad()
    def compress_group(self, U, S, rank):
        if rank == 0:
            return nn.ParameterList(), nn.ParameterList()
        S[0] = nn.Parameter(S[0].data[:, :rank].clone())                                                    
        for i in range(3):
            U[i] = nn.Parameter(U[i].data[:, :rank].clone())
        return U, S

    @torch.no_grad()
    def compress(self, ranks):
                                                                 
        if not self.finalized:
            self.finalize()
        
        self.U_vec_density, self.S_vec_density = self.compress_group(self.U_vec_density, self.S_vec_density, ranks[0])
        self.U_mat_density, self.S_mat_density = self.compress_group(self.U_mat_density, self.S_mat_density, ranks[1])
        self.U_vec, self.S_vec = self.compress_group(self.U_vec, self.S_vec, ranks[2])
        self.U_mat, self.S_mat = self.compress_group(self.U_mat, self.S_mat, ranks[3])

                       
        self.rank_vec_density[0] = [ranks[0]]
        self.rank_mat_density[0] = [ranks[1]]
        self.rank_vec[0] = [ranks[2]]
        self.rank_mat[0] = [ranks[3]]

        self.group_vec_density[0] = self.rank_vec_density[0]
        self.group_mat_density[0] = self.rank_mat_density[0]
        self.group_vec[0] = self.rank_vec[0]
        self.group_mat[0] = self.rank_mat[0]

    @torch.no_grad()
    def compose(self, other, R=None, s=None, t=None): 
        if not self.finalized:
            self.finalize()
        if not other.finalized:
            other.finalize()

                    
        self.U_vec_density.extend(other.U_vec_density)
        self.S_vec_density.extend(other.S_vec_density)

        self.U_mat_density.extend(other.U_mat_density)
        self.S_mat_density.extend(other.S_mat_density)

        self.U_vec.extend(other.U_vec)
        self.S_vec.extend(other.S_vec)

        self.U_mat.extend(other.U_mat)
        self.S_mat.extend(other.S_mat)

                
        self.rank_vec_density.extend(other.rank_vec_density)
        self.rank_mat_density.extend(other.rank_mat_density)
        self.rank_vec.extend(other.rank_vec)
        self.rank_mat.extend(other.rank_mat)

        self.group_vec_density.extend(other.group_vec_density)
        self.group_mat_density.extend(other.group_mat_density)
        self.group_vec.extend(other.group_vec)
        self.group_mat.extend(other.group_mat)

        self.K.extend(other.K)

                    
        oid = len(self.K) - 1

                                              
        if R is None:
            R = torch.eye(3, dtype=torch.float32)
        elif isinstance(R, np.ndarray):
            R = torch.from_numpy(R.astype(np.float32))
        else:         
            R = R.float()

                                      
        if s is None:
            s = 1
        
                                       
        if t is None:
            t = torch.zeros(3, dtype=torch.float32)
        elif isinstance(t, np.ndarray):
            t = torch.from_numpy(t.astype(np.float32))
        else:         
            t = t.float()

                                             
                                               
        T = torch.eye(4, dtype=torch.float32)
        T[:3, :3] = R * s
        T[:3, 3] = t
        
                                                                                               
        T = torch.inverse(T).to(self.aabb_train.device)
        R = R.T.to(self.aabb_train.device)
        
        self.register_buffer(f'T_{oid}', T)
        self.register_buffer(f'R_{oid}', R)
        self.register_buffer(f'aabb_{oid}', other.aabb_train)
        
                                                                        
                                      
        for _ in range(3):
            self.update_extra_state()
        

                     
    def get_params(self, lr1, lr2):
        params = [
            {'params': self.U_vec_density, 'lr': lr1},
            {'params': self.S_vec_density, 'lr': lr2},
            {'params': self.U_mat_density, 'lr': lr1}, 
            {'params': self.S_mat_density, 'lr': lr2},
            {'params': self.U_vec, 'lr': lr1},
            {'params': self.S_vec, 'lr': lr2},
            {'params': self.U_mat, 'lr': lr1}, 
            {'params': self.S_mat, 'lr': lr2},
        ]
        if self.bg_radius > 0:
            params.append({'params': self.bg_mat, 'lr': lr1})
            params.append({'params': self.bg_S, 'lr': lr2})
        return params
        