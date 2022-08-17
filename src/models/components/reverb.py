import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

from src.utils.constants import SAMPLE_RATE

from .fft_conv import fft_conv1d


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
        length = self.duration * SAMPLE_RATE - 1
        ir = torch.randn(self.out_ch, self.in_ch, length)
        envelop = torch.exp(-4.0 * torch.linspace(0.0, self.duration, steps=length))
        ir *= envelop
        ir = ir / torch.norm(ir, p=2, dim=-1, keepdim=True)
        ir = ir.flip(-1)

        # we can train a mono synth and controller, add stereo width using reverb
        # [output dimension, input_dimension, time]
        return ir * 1e-4

    def forward(self, x: torch.Tensor):
        ir = torch.concat(
            [torch.tanh(self.ir), torch.ones(*self.ir.shape[:-1], 1, device=x.device)], dim=-1
        )
        out = F.pad(x, (ir.shape[-1] - 1, 0))
        out = fft_conv1d(out, ir, block_ratio=self.block_ratio)

        return out
