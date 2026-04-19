                                   
import torch
import torch.nn.functional as F
from torch.autograd import gradcheck
import numpy as np
from gridencoder.grid import _grid_encode
import random
import os
                               
device=torch.device(0)
input_dim=3    
num_levels=4    
level_dim=2    
per_level_scale=2
base_resolution=4    
log2_hashmap_size=8    
                                                                                        

output_dim = num_levels * level_dim

if level_dim % 2 != 0:
    print('[WARN] detected HashGrid level_dim % 2 != 0, which will cause very slow backward is also enabled fp16! (maybe fix later)')

                     
offsets = []
offset = 0
max_params = 2 ** log2_hashmap_size
for i in range(num_levels):
    resolution = int(np.ceil(base_resolution * per_level_scale ** i))
    params_in_level = min(max_params, (resolution + 1) ** input_dim)                   
                                                                        
    offsets.append(offset)
    offset += params_in_level
offsets.append(offset)

print(offsets)

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)                              
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

             

            
inputs = torch.rand(1, input_dim, dtype= torch.float64, requires_grad=False).to(device)

offsets = torch.from_numpy(np.array(offsets, dtype=np.int32)).to(device)
embeddings = torch.randn(offset, level_dim, dtype=torch.float64, requires_grad=True).to(device) * 0.1

print(inputs)
print(embeddings)


Inputs = (inputs, embeddings, offsets, per_level_scale, base_resolution, inputs.requires_grad)
check_results1 = torch.autograd.gradcheck(_grid_encode.apply, Inputs, eps=1e-2, atol=1e-3, rtol=0.01, fast_mode=False)
print("check_results1", check_results1)
