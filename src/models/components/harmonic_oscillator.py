import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from einops.layers.torch import Reduce

from src.utils.constants import HOP_LENGTH, SAMPLE_RATE
from src.utils.helpers import bins_to_cents, cents_to_freqs


class HarmonicOscillator(nn.Module):
    def __init__(self, n_harmonics: int = 64, n_channels: int = 1):
        super().__init__()

        self.n_harmonics = n_harmonics
        self.n_channels = n_channels

        harmonics = torch.arange(1, self.n_harmonics + 1, step=1)
        self.register_buffer("harmonics", harmonics, persistent=False)

        self.sum_sinusoids = Reduce("b c o t -> b c t", "sum")

    def forward(
        self, f0: torch.Tensor, master_amplitude: torch.Tensor, overtone_amplitudes: torch.Tensor
    ):
        # f0.shape = [batch, n_channels, time]
        # master_amplitude.shape = [batch, n_channels, time]
        # overtone_amplitudes = [batch, n_channels, n_harmonics, time]

        # Convert f0 from Hz to radians / sample
        # This is faster and does not explode freq values when using 16-bit precision.
        f0 = f0 / SAMPLE_RATE
        f0 = f0 * 2 * np.pi

        # Calculate overtone frequencies
        overtone_fs = torch.einsum("bct,o->bcot", f0, self.harmonics)

        # set amplitudes of overtones above Nyquist to 0.0
        overtone_amplitudes[overtone_fs > np.pi] = 0.0
        # normalize harmonic_distribution so it always sums to one
        overtone_amplitudes /= torch.sum(overtone_amplitudes, dim=2, keepdim=True)
        # scale individual overtone amplitudes by the master amplitude
        overtone_amplitudes = torch.einsum("bcot,bct->bcot", overtone_amplitudes, master_amplitude)

        # stretch controls by hop_size
        # refactor stretch into a function or a method
        # overtone_fs = self.pre_stretch(overtone_fs)
        overtone_fs = F.interpolate(
            overtone_fs,
            size=(overtone_fs.shape[-2], (f0.shape[-1] - 1) * HOP_LENGTH),
            mode="bilinear",
            align_corners=True,
        )
        # overtone_fs = self.post_stretch(overtone_fs)
        # overtone_amplitudes = self.pre_stretch(overtone_amplitudes)
        overtone_amplitudes = F.interpolate(
            overtone_amplitudes,
            size=(overtone_amplitudes.shape[-2], (f0.shape[-1] - 1) * HOP_LENGTH),
            mode="bilinear",
            align_corners=True,
        )
        # overtone_amplitudes = self.post_stretch(overtone_amplitudes)

        # calculate phases and sinusoids
        # TODO: randomizing phases. Is it necessary?
        overtone_fs[:, :, :, 0] = 3.14159265 * (
            torch.rand(*overtone_fs.shape[:-1], device=overtone_fs.device) * 2 - 1
        )
        phases = torch.cumsum(overtone_fs, dim=-1)
        sinusoids = torch.sin(phases)

        # scale sinusoids by their corresponding amplitudes and sum them to get the final signal
        sinusoids = torch.einsum("bcot,bcot->bcot", sinusoids, overtone_amplitudes)
        signal = self.sum_sinusoids(sinusoids)

        return signal
