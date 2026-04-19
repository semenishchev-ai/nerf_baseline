
import numpy as np
import torch
from gridencoder import GridEncoder

B = 1
D = 2

enc = GridEncoder(D=D, L=2, C=1, base_resolution=4, log2_hashmap_size=5).cuda()
                                                             

print(f"=== enc ===")
print(enc.embeddings.shape)
print(enc.embeddings)

                                                 
x = torch.FloatTensor(np.array([
             
            
    [0, 0],
              
             
])).cuda()

                       

print(f"=== x ===")
print(x)
print(x.shape)

y = enc(x, calc_grad_inputs=False)

print(f"=== y ===")
print(y.shape)
print(y)

y.sum().backward()

print(f"=== grad enc ===")
print(enc.embeddings.grad.shape)
print(enc.embeddings.grad)

                    
              