import torch
import torch.nn.functional as F
from typing import Optional
from configuration import Config


EPS = 1e-6


def get_range(bits):
    return -(2 ** (bits - 1)), (2 ** (bits - 1)) - 1


def quantize(
    t: torch.Tensor,
    dequantize: bool,
    scale: torch.Tensor,
    min_range: int,
    max_range: int,
):
    scaled_t = t * scale
    q_t_rounded = torch.round(scaled_t)
    if dequantize:
        q_t_rounded = (q_t_rounded - scaled_t).detach() + scaled_t
    q_t = torch.clamp(q_t_rounded, min=min_range, max=max_range)
    if dequantize:
        dq_t = q_t / scale
        return dq_t
    else:
        return q_t, scale


def aap_quantization(
    t: torch.Tensor,
    dequantize: bool,
    initial_percentile: Optional[float] = None,
    delta: Optional[float] = None,
):
    gamma = torch.quantile(input=t.abs(), q=initial_percentile)
    scale = 1 / ((gamma * delta) + EPS)
    return quantize(
        t=t,
        dequantize=dequantize,
        scale=scale,
        min_range=-1,
        max_range=1,
    )
    

def abs_max_quantization(
    t: torch.Tensor,
    dequantize: bool,
    bits: int,
):
    min_range, max_range = get_range(bits)
    act_max = t.abs().max()
    scale = max_range / (act_max + EPS)
    return quantize(
        t=t,
        dequantize=dequantize,
        scale=scale,
        min_range=min_range,
        max_range=max_range,
    )


class AAPLinearTraining(torch.nn.Linear):
    def __init__(self, in_features, out_features, config: Config, *args, **kwargs):
        super().__init__(in_features, out_features, *args, **kwargs)
        self.act_bits: int = config.act_bits
        self.initial_percentile = config.initial_percentile
        self.delta = torch.nn.Parameter(torch.ones((1,)), requires_grad=True)

    def forward(self, x: torch.Tensor):
        noisy_x = abs_max_quantization(x, dequantize=True, bits=self.act_bits)
        self.noisy_w = aap_quantization(
            self.weight,
            dequantize=True,
            initial_percentile=self.initial_percentile,
            delta=self.delta,
        )
        return F.linear(input=noisy_x, weight=self.noisy_w, bias=self.bias)


class AAPLinearInference(AAPLinearTraining):
    def forward(self, x):
        q_w = self.weight

        q_x, s_x = abs_max_quantization(x, dequantize=False, bits=self.act_bits)
        
        q_out = F.linear(input=q_x, weight=q_w)
        fp_out = q_out / (s_x * self.s_w)
        fp_out = fp_out + self.bias
        return fp_out

