import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

from src.utils.constants import SAMPLE_RATE

from .fft_conv import fft_conv1d


class ConvolutionalReverb(nn.Module):
    def __init__(self, duration=3, in_ch=1, out_ch=1):
        super().__init__()
        self.duration = duration
        self.in_ch = in_ch
        self.out_ch = out_ch
        # first, (if reversed, last) bit of ir should always be 1
        self.ir = nn.Parameter(self.init_ir())

    def init_ir(self):
        length = self.duration * SAMPLE_RATE - 1
        ir = torch.randn(self.out_ch, self.in_ch, length)
        envelop = torch.exp(-4.0 * torch.linspace(0.0, self.duration, steps=length))
        ir *= envelop
        ir = ir / torch.norm(ir, p=2, dim=-1, keepdim=True)
        ir = ir.flip(-1)

        # we can train a mono synth and controller, add stereo width using reverb
        # [output dimension, input_dimension, time]
        return ir

    def forward(self, x: torch.Tensor):
        ir = torch.concat([self.ir, torch.ones(*self.ir.shape[:-1], 1, device=x.device)], dim=-1)
        x = F.pad(x, (ir.shape[-1] - 1, 0))
        out = fft_conv1d(x, self.ir)

        return out

    def forward_live(self, x: torch.Tensor):
        ir = torch.concat([self.ir, torch.ones(*self.ir.shape[:-1], 1, device=x.device)], dim=-1)
        x = F.pad(x, (ir.shape[-1] - 1, ir.shape[-1] - 1))
        out = fft_conv1d(x, self.ir)

        return out
