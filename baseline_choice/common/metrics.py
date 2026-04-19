import os
import numpy as np
import torch
import lpips
from torchmetrics.functional import structural_similarity_index_measure

class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)
        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths)
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))
        self.V += psnr
        self.N += 1

    def measure(self):
        return self.V / self.N if self.N > 0 else 0

    def report(self):
        return f'PSNR = {self.measure():.6f}'

class SSIMMeter:
    def __init__(self, device=None):
        self.V = 0
        self.N = 0
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if len(inp.shape) == 3: # [H, W, 3]
                inp = inp.unsqueeze(0)
            inp = inp.permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths)
        ssim = structural_similarity_index_measure(preds, truths)
        self.V += ssim.item()
        self.N += 1

    def measure(self):
        return self.V / self.N if self.N > 0 else 0

    def report(self):
        return f'SSIM = {self.measure():.6f}'

class LPIPSMeter:
    def __init__(self, net='alex', device=None):
        self.V = 0
        self.N = 0
        self.net = net
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fn = lpips.LPIPS(net=net).eval().to(self.device)

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if len(inp.shape) == 3: # [H, W, 3]
                inp = inp.unsqueeze(0)
            inp = inp.permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs
    
    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths)
        v = self.fn(truths, preds, normalize=True).item()
        self.V += v
        self.N += 1
    
    def measure(self):
        return self.V / self.N if self.N > 0 else 0

    def report(self):
        return f'LPIPS ({self.net}) = {self.measure():.6f}'
