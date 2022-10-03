import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

from performer.utils.constants import SAMPLE_RATE

from .fft_conv import fft_conv1d, fft_conv1d_new


class ConvolutionalReverb(nn.Module):
    def __init__(self, duration=3, in_ch=1, out_ch=1, block_ratio=5):
        super().__init__()
        self.duration = duration
        self.in_ch = in_ch
        self.out_ch = out_ch
        if block_ratio < 1:
            raise RuntimeError("Block ratio must be greater than 1.")
        self.block_ratio = block_ratio
        # first, (if reversed, last) bit of ir should always be 1
        self.ir = nn.Parameter(self.init_ir())

    def init_ir(self):
        length = self.duration * SAMPLE_RATE
        ir = torch.randn(self.out_ch, self.in_ch, length)
        envelop = torch.exp(-4.0 * torch.linspace(0.0, self.duration, steps=length))
        ir *= envelop
        ir = ir / torch.norm(ir, p=2, dim=-1, keepdim=True)

        # we can train a mono synth and controller, add stereo width using reverb
        # [output dimension, input_dimension, time]
        return ir

    def forward(self, x: torch.Tensor):
        out, _ = fft_conv1d_new(x, torch.tanh(self.ir))

        return out
