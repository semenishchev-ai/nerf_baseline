import math
from turtle import backward, forward

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd 
import atexit

try:
    import _ffmlp as _backend
except ImportError:
    from .backend import _backend

class _ffmlp_forward(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(ctx, inputs, weights, input_dim, output_dim, hidden_dim, num_layers, activation, output_activation, inference=False, calc_grad_inputs=False):
        
        B = inputs.shape[0]

        inputs = inputs.contiguous()
        weights = weights.contiguous()

                                                                                                                                 
                                                                                                                                       

                         
        outputs = torch.empty(B, output_dim, device=inputs.device, dtype=inputs.dtype)

        if not inference:
            forward_buffer = torch.empty(num_layers, B, hidden_dim, device=inputs.device, dtype=inputs.dtype)
            _backend.ffmlp_forward(inputs, weights, B, input_dim, output_dim, hidden_dim, num_layers, activation, output_activation, forward_buffer, outputs)
            ctx.save_for_backward(inputs, weights, outputs, forward_buffer)
            ctx.dims = (input_dim, output_dim, hidden_dim, num_layers, activation, output_activation, calc_grad_inputs)

                                                                                                                                           
                                                                                                                                                                                     
        else:
            inference_buffer = torch.empty(B, hidden_dim, device=inputs.device, dtype=inputs.dtype)
            _backend.ffmlp_inference(inputs, weights, B, input_dim, output_dim, hidden_dim, num_layers, activation, output_activation, inference_buffer, outputs)

                                                                                                                                           
                                                                                                                                                                                                 


        return outputs
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
                               

        B = grad.shape[0]

        grad = grad.contiguous()

                                                                                                                     
                     

        inputs, weights, outputs, forward_buffer = ctx.saved_tensors

        input_dim, output_dim, hidden_dim, num_layers, activation, output_activation, calc_grad_inputs = ctx.dims

                          
        if calc_grad_inputs:
            grad_inputs = torch.zeros_like(inputs) 
        else:
            grad_inputs = torch.zeros(1, device=grad.device, dtype=grad.dtype)        

        grad_weights = torch.zeros_like(weights)
        backward_buffer = torch.zeros(num_layers, B, hidden_dim, device=grad.device, dtype=grad.dtype)

        _backend.ffmlp_backward(grad, inputs, weights, forward_buffer, B, input_dim, output_dim, hidden_dim, num_layers, activation, output_activation, calc_grad_inputs, backward_buffer, grad_inputs, grad_weights)

                                                                                                                          
                                                                                                                               
                                                                                                                                              
        if calc_grad_inputs:
            return grad_inputs, grad_weights, None, None, None, None, None, None, None, None
        else:
            return None, grad_weights, None, None, None, None, None, None, None, None


ffmlp_forward = _ffmlp_forward.apply


def convert_activation(act):
    if act == 'relu': return 0
    elif act == 'exponential': return 1
    elif act == 'sine': return 2
    elif act == 'sigmoid': return 3
    elif act == 'squareplus': return 4
    elif act == 'softplus': return 5
    else: return 6
    

class FFMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, activation='relu'):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = convert_activation(activation)
        self.output_activation = convert_activation('none')                          

        self.tensorcore_width = 16

        assert hidden_dim in [16, 32, 64, 128, 256], f"FFMLP only support hidden_dim in [16, 32, 64, 128, 256], but got {hidden_dim}"
        assert input_dim > 0 and input_dim % 16 == 0, f"FFMLP input_dim should be 16 * m (m  > 0), but got {input_dim}"
        assert output_dim <= 16, f"FFMLP current only supports output dim <= 16, but got {output_dim}"
        assert num_layers >= 2, f"FFMLP num_layers should be larger than 2 (3 matmuls), but got {num_layers}"
        
                    
        self.padded_output_dim = int(math.ceil(output_dim / 16)) * 16

                                           
        self.num_parameters = hidden_dim * (input_dim + hidden_dim * (num_layers - 1) + self.padded_output_dim)
        self.weights = nn.Parameter(torch.zeros(self.num_parameters))
        self.reset_parameters()

                          
        _backend.allocate_splitk(self.num_layers + 1)

                             
                                                                                                                                                          


    def cleanup(self):
                         
        _backend.free_splitk()
    

    def __repr__(self):
        return f"FFMLP: input_dim={self.input_dim} output_dim={self.output_dim} hidden_dim={self.hidden_dim} num_layers={self.num_layers} activation={self.activation}"


    def reset_parameters(self):
        torch.manual_seed(42)
        std = math.sqrt(3 / self.hidden_dim)
        self.weights.data.uniform_(-std, std)
    

    def forward(self, inputs):
                                
                                 

                                                                                                                    

        B, C = inputs.shape
                                                                                                    

                   
        pad = 128 - (B % 128)
        if pad > 0:
            inputs = torch.cat([inputs, torch.zeros(pad, C, dtype=inputs.dtype, device=inputs.device)], dim=0)

        outputs = ffmlp_forward(inputs, self.weights, self.input_dim, self.padded_output_dim, self.hidden_dim, self.num_layers, self.activation, self.output_activation, not self.training, inputs.requires_grad)

                      
        if B != outputs.shape[0] or self.padded_output_dim != self.output_dim:
            outputs = outputs[:B, :self.output_dim]
    
                                                                                                   

        return outputs