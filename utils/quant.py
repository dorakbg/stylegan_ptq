from torch import nn
import torch.nn.functional as F
import torch
import numpy as np


from torch.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver

_NBITS = 8
_ACTMAX = 4.0


class MovingAverageQuantileObserver(MinMaxObserver):

    def __init__(self, averaging_constant=0.01, q_min=0.0, q_max=1.0, dtype=torch.quint8,
                 qscheme=torch.per_tensor_affine, reduce_range=False,
                 quant_obs_min=None, quant_obs_max=None):
        self.averaging_constant = averaging_constant
        super(MovingAverageQuantileObserver, self).__init__(dtype=dtype,
                                                            qscheme=qscheme,
                                                            reduce_range=reduce_range,
                                                            quant_min=quant_obs_min,
                                                            quant_max=quant_obs_max)
        self.q_min = q_min
        self.q_max = q_max
        self.quant_min = quant_obs_min
        self.quant_max = quant_obs_max

    def forward(self, x_orig):
        x = x_orig.detach()  # avoid keeping autograd tape
        min_val = self.min_val
        max_val = self.max_val

        if self.q_min == 0.0:
            min_now = torch.min(x)
        else:
            min_now = torch.quantile(x, self.q_min)

        if self.q_max == 1.0:
            max_now = torch.max(x)
        else:
            max_now = torch.quantile(x, self.q_max)

        if min_val == float('inf') and max_val == float('-inf'):
            min_val = min_now
            max_val = max_now
        else:
            max_val = max_val + self.averaging_constant * (max_now - max_val)
            min_val = min_val + self.averaging_constant * (min_now - min_val)

        self.min_val = min_val
        self.max_val = max_val
        return x_orig

    @torch.jit.export
    def extra_repr(self):
        return "min_val={}, max_val={}, q_min={}, q_max={}, quant_min={}, quant_max={}".format(self.min_val, self.max_val, self.q_min, self.q_max, self.quant_min, self.quant_max)
    
    
class MovingAveragePerChannelQuantileObserver(PerChannelMinMaxObserver):

    def __init__(self, averaging_constant=0.01, ch_axis=0, dtype=torch.quint8,
                 qscheme=torch.per_channel_affine, reduce_range=False, q_min=0.0, q_max=1.0,
                 quant_min=None, quant_max=None):
        super(MovingAveragePerChannelQuantileObserver, self).__init__(
            ch_axis=ch_axis, dtype=dtype, qscheme=qscheme,
            reduce_range=reduce_range, quant_min=quant_min, quant_max=quant_max)
        self.averaging_constant = averaging_constant
        self.q_min = q_min
        self.q_max = q_max

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_vals.dtype)
        min_vals = self.min_vals
        max_vals = self.max_vals
        x_dim = x.size()

        new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
        new_axis_list[self.ch_axis] = 0
        new_axis_list[0] = self.ch_axis
        y = x.permute(new_axis_list)
        y = torch.flatten(y, start_dim=1)
        if min_vals.numel() == 0 or max_vals.numel() == 0:
            min_vals, max_vals = torch.quantile(y, self.q_min, 1), torch.quantile(y, self.q_max, 1)
        else:
            min_vals_cur, max_vals_cur = torch._aminmax(y, 1)
            min_vals = min_vals + self.averaging_constant * (min_vals_cur - min_vals)
            max_vals = max_vals + self.averaging_constant * (max_vals_cur - max_vals)
        self.min_vals.resize_(min_vals.shape)
        self.max_vals.resize_(max_vals.shape)
        self.min_vals.copy_(min_vals)
        self.max_vals.copy_(max_vals)
        return x_orig
    
    
    
class ConstantObserver(MinMaxObserver):

    def __init__(self, q_min=0.0, q_max=1.0, dtype=torch.quint8,
                 qscheme=torch.per_tensor_affine, reduce_range=8):
        super(ConstantObserver, self).__init__(dtype=dtype,
                                                            qscheme=qscheme,
                                                            reduce_range=reduce_range)
        self.q_min = q_min
        self.q_max = q_max

    def forward(self, x_orig):

        self.min_val = 0
        self.max_val = _ACTMAX
        return x_orig
    
    
    @torch.jit.export
    def _calculate_qparams(self, min_val, max_val):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
        r"""Calculates the quantization parameters, given min and max
        value tensors. Works for both per tensor and per channel cases
        Args:
            min_val: Minimum values per channel
            max_val: Maximum values per channel
        Returns:
            scales: Scales tensor of shape (#channels,)
            zero_points: Zero points tensor of shape (#channels,)
        """
        if min_val.numel() == 0 or max_val.numel() == 0:
            warnings.warn(
                "must run observer before calling calculate_qparams.\
                                    Returning default scale and zero point "
            )
            return torch.tensor([1.0]), torch.tensor([0])

        if min_val.dim() == 0 or max_val.dim() == 0:
            assert min_val <= max_val, "min {} should be less than max {}".format(
                min_val, max_val
            )
        else:
            assert torch.sum(min_val <= max_val) == len(min_val), "min {} should be less than max {}".format(
                min_val, max_val
            )

        if self.dtype == torch.qint8:
            qmin, qmax = -2**(self.reduce_range - 1), 2**(self.reduce_range - 1) - 1
         #   if self.reduce_range:
         #       qmin, qmax = -64, 63
         #   else:
         #       qmin, qmax = -128, 127
        else:
            qmin, qmax = 0, 2**(self.reduce_range) - 1
          #  if self.reduce_range:
          #      qmin, qmax = 0, 127
          #  else:
          #      qmin, qmax = 0, 255

        min_val = torch.zeros_like(min_val)#torch.min(min_val, torch.zeros_like(min_val))
        max_val = torch.ones_like(min_val) * _ACTMAX##torch.max(max_val, torch.zeros_like(max_val))

        scale = torch.ones(min_val.size(), dtype=torch.float32)
        zero_point = torch.zeros(min_val.size(), dtype=torch.int64)
        device = 'cuda' if min_val.is_cuda else 'cpu'

        if self.qscheme == torch.per_tensor_symmetric or self.qscheme == torch.per_channel_symmetric:
            max_val = torch.max(-min_val, max_val)
            scale = max_val / (float(qmax - qmin) / 2)
            scale = torch.max(scale, torch.tensor(self.eps, device=device, dtype=scale.dtype))
            if self.dtype == torch.quint8:
                zero_point = zero_point.new_full(zero_point.size(), 128)
        else:
            scale = (max_val - min_val) / float(qmax - qmin)
            scale = torch.max(scale, torch.tensor(self.eps, device=device, dtype=scale.dtype))
            zero_point = qmin - torch.round(min_val / scale)
            zero_point = torch.max(zero_point, torch.tensor(qmin, device=device, dtype=zero_point.dtype))
            zero_point = torch.min(zero_point, torch.tensor(qmax, device=device, dtype=zero_point.dtype))
        # For scalar values, cast them to Tensors of size 1 to keep the shape
        # consistent with default values in FakeQuantize.
        if len(scale.shape) == 0:
            # TODO: switch to scale.item() after adding JIT support
            scale = torch.tensor([float(scale)], dtype=scale.dtype)
        if len(zero_point.shape) == 0:
            # TODO: switch to zero_point.item() after adding JIT support
            zero_point = torch.tensor([int(zero_point)], dtype=zero_point.dtype)

        return scale, zero_point