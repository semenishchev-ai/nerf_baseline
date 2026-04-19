import numpy as np

import torch
from torch.utils.data import Dataset

import trimesh
import pysdf

def map_color(value, cmap_name='viridis', vmin=None, vmax=None):
                       
                                          
    import matplotlib.cm as cm
    if vmin is None: vmin = value.min()
    if vmax is None: vmax = value.max()
    value = (value - vmin) / (vmax - vmin)                  
    cmap = cm.get_cmap(cmap_name) 
    rgb = cmap(value)[:, :3]                                                        
    return rgb

def plot_pointcloud(pc, sdfs):
                
                  
    color = map_color(sdfs.squeeze(1))
    pc = trimesh.PointCloud(pc, color)
    trimesh.Scene([pc]).show()    

             
class SDFDataset(Dataset):
    def __init__(self, path, size=100, num_samples=2**18, clip_sdf=None):
        super().__init__()
        self.path = path

                   
        self.mesh = trimesh.load(path, force='mesh')

                                                                           
        vs = self.mesh.vertices
        vmin = vs.min(0)
        vmax = vs.max(0)
        v_center = (vmin + vmax) / 2
        v_scale = 2 / np.sqrt(np.sum((vmax - vmin) ** 2)) * 0.95
        vs = (vs - v_center[None, :]) * v_scale
        self.mesh.vertices = vs

        print(f"[INFO] mesh: {self.mesh.vertices.shape} {self.mesh.faces.shape}")

        if not self.mesh.is_watertight:
            print(f"[WARN] mesh is not watertight! SDF maybe incorrect.")
                                          

        self.sdf_fn = pysdf.SDF(self.mesh.vertices, self.mesh.faces)
        
        self.num_samples = num_samples
        assert self.num_samples % 8 == 0, "num_samples must be divisible by 8."
        self.clip_sdf = clip_sdf

        self.size = size

    
    def __len__(self):
        return self.size

    def __getitem__(self, _):

                         
        sdfs = np.zeros((self.num_samples, 1))
                 
        points_surface = self.mesh.sample(self.num_samples * 7 // 8)
                         
        points_surface[self.num_samples // 2:] += 0.01 * np.random.randn(self.num_samples * 3 // 8, 3)
                
        points_uniform = np.random.rand(self.num_samples // 8, 3) * 2 - 1
        points = np.concatenate([points_surface, points_uniform], axis=0).astype(np.float32)

        sdfs[self.num_samples // 2:] = -self.sdf_fn(points[self.num_samples // 2:])[:,None].astype(np.float32)
 
                  
        if self.clip_sdf is not None:
            sdfs = sdfs.clip(-self.clip_sdf, self.clip_sdf)

        results = {
            'sdfs': sdfs,
            'points': points,
        }

                                      

        return results
