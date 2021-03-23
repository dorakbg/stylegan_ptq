# This code is based on this implementation of adaptive rounding https://github.com/yhhhli/BRECQ/blob/main/quant/adaptive_rounding.py 
# but modified to be compatible with PyTorch native quantization interface

from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.quantization.observer import _with_args

class AdaRound(nn.Module):
    
    def __init__(self, observer, quant_min=-128, quant_max=127, scale=1., zero_point=0., **observer_kwargs):
        """Adaround FakeQuantization module

        Args:
            observer (torch.quantization.observer): quantization obserever to initilalize quantizers
            quant_min (int, optional): quantized values range (min). Defaults to -128.
            quant_max (int, optional): quantized values range (max). Defaults to 127.
            scale (float, optional):  Defaults to 1..
            zero_point (float, optional):  Defaults to 0..
        """
        super(AdaRound, self).__init__()
        assert quant_min < quant_max, 'quant_min must be strictly less than quant_max.'
        self.quant_min = quant_min
        self.quant_max = quant_max

        self.activation_post_process = observer(**observer_kwargs)
        assert torch.iinfo(self.activation_post_process.dtype).min <= quant_min, \
               'quant_min out of bound'
        assert quant_max <= torch.iinfo(self.activation_post_process.dtype).max, \
               'quant_max out of bound'
        self.register_buffer('scale', torch.tensor([scale]))
        self.register_buffer('zero_point', torch.tensor([zero_point])) 
        
        self.dtype = self.activation_post_process.dtype
        self.qscheme = self.activation_post_process.qscheme
        assert self.qscheme in (torch.per_channel_symmetric, torch.per_tensor_symmetric)
        self.ch_axis = self.activation_post_process.ch_axis \
            if hasattr(self.activation_post_process, 'ch_axis') else -1
        self.register_buffer('observer_enabeled', torch.tensor([1], dtype=torch.uint8))
        
        self.register_buffer('fake_quant_enabled', torch.tensor([1], dtype=torch.uint8))
        
        self.register_buffer('soft_sigmoid_enabled', torch.tensor([0], dtype=torch.uint8))
        self.register_buffer('hard_sigmoid_enabled', torch.tensor([0], dtype=torch.uint8))


        bitrange = torch.tensor(quant_max - quant_min + 1).double()
        self.bitwidth = int(torch.log2(bitrange).item())
        
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2./3
        
    def enable_floating_point(self):
        self.fake_quant_enabled[0] = 0
        self.observer_enabeled[0] = 0
        self.hard_sigmoid_enabled[0] = 0
        self.soft_sigmoid_enabled[0] = 0
        return self

    def enable_nearest_quant(self):
        self.fake_quant_enabled[0] = 1
        self.observer_enabeled[0] = 1
        self.hard_sigmoid_enabled[0] = 0
        self.soft_sigmoid_enabled[0] = 0
        return self

    @torch.no_grad()
    def enable_sigmoid_quant(self):

        if not hasattr(self, "X"):
            return
        
        self.fake_quant_enabled[0] = 1
        self.observer_enabeled[0] = 0
        self.hard_sigmoid_enabled[0] = 0
        self.soft_sigmoid_enabled[0] = 1
        
        if not hasattr(self, "alpha"):
            scale = self.scale
            for i in range(len(self.X.shape) - 1):
                scale = scale.unsqueeze(-1)
            X_floor = torch.floor(self.X.to(scale.device) / scale)      
            rest = self.X.to(scale.device) / scale - X_floor  # rest of rounding [0, 1)
            alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
            self.alpha = nn.Parameter(alpha)     
    
    def enable_hard_sigmoid(self):
        self.fake_quant_enabled[0] = 1
        self.observer_enabeled[0] = 0
        self.hard_sigmoid_enabled[0] = 1
        self.soft_sigmoid_enabled[0] = 0

    def fake_quant(self, X):
        if self.qscheme == torch.per_channel_symmetric:
            zero_point = torch.LongTensor([i.round() for i in self.zero_point]).to(self.zero_point.device)
            X = torch.fake_quantize_per_channel_affine(
                X, self.scale, zero_point, self.ch_axis,
                self.quant_min, self.quant_max)
        else:
            X = torch.fake_quantize_per_tensor_affine(
                X, float(self.scale.item()), int(self.zero_point.item()),
                self.quant_min, self.quant_max)
        return X
        
    def get_soft_targets(self):
        return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)       
    
    def forward(self, X):
        if self.observer_enabeled[0] == 1:
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.activation_post_process.calculate_qparams()
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            self.scale.resize_(_scale.shape)
            self.scale.copy_(_scale)
            self.zero_point.resize_(_zero_point.shape)
            self.zero_point.copy_(_zero_point)

        if not hasattr(self, "X"):
            self.X = X           
            
        if self.fake_quant_enabled[0] == 0:
            return X      
        
        if self.soft_sigmoid_enabled[0]==1 or self.hard_sigmoid_enabled[0]==1:
            scale = self.scale
            for i in range(len(self.X.shape) - 1):
                scale = scale.unsqueeze(-1)
            X_floor = torch.floor(X / scale)
            if self.soft_sigmoid_enabled[0]:
                X_int = X_floor + self.get_soft_targets()
            else:
                X_int = X_floor + (self.alpha >= 0).float()  
            X_quant = torch.clamp(X_int, self.quant_min, self.quant_max)
            X = X_quant * scale
        else:
            X = self.fake_quant(X)

        return X
    
    with_args = classmethod(_with_args)
    
    
def enable_floating_point(mod):
    if type(mod) == AdaRound:
        mod.enable_floating_point()

def enable_nearest_quant(mod):
    if type(mod) == AdaRound:
        mod.enable_nearest_quant()

def enable_sigmoid_quant(mod):
    if type(mod) == AdaRound:
        mod.enable_sigmoid_quant()  

def enable_hard_sigmoid(mod):
    if type(mod) == AdaRound:
        mod.enable_hard_sigmoid() 

def weighted_mse(y, y_hat, weights):
    return ((y - y_hat).pow(2) * weights).mean()  
    
class LossFunction:
    def __init__(self,
                 block,
                 weight_mode: str = 'adaround',
                 weight: float = 0.01,
                 max_count: int = 2000,
                 b_range: tuple = (20, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.2):
        self.block = block
        self.weight_mode = weight_mode
        self.weight = weight
        self.loss_start = max_count * warmup

        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0

    def __call__(self, y, y_hat, weights):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy
        :param pred: output from quantized model
        :param tgt: output from FP model
        :param grad: gradients to compute fisher information
        :return: total loss function
        """
        self.count += 1
        rec_loss = weighted_mse(y, y_hat, weights)

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.weight_mode == 'STE':
            b = round_loss = 0
        elif self.weight_mode == 'adaround':
            round_loss = 0
            for name, module in self.block.named_modules():
                if isinstance(module, AdaRound):
                    round_vals = module.get_soft_targets()
                    round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss
        if self.count % 100 == 0:
            print('Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'.format(
                  float(total_loss), float(rec_loss), float(round_loss), b, self.count))
        return total_loss


class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        Cosine annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))