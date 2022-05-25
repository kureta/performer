import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F  # noqa
from einops import rearrange

from src.utils.constants import HOP_LENGTH, N_FFT


class FilteredNoise(nn.Module):
    def __init__(
        self,
        n_bands: int = 128,
        n_channels: int = 1,
    ):
        super().__init__()

        self.n_bands = n_bands
        self.n_channels = n_channels

    def init_noise(self, batch_size, n_steps, device):
        noise = torch.rand(batch_size, self.n_channels, (n_steps - 1) * HOP_LENGTH, device=device)
        noise = noise * 2.0 - 1.0

        return noise

    def forward(self, filter_bands):
        # filter_bands.shape = [batch, n_channels, n_bands, time]

        # Generate white noise
        batch_size, _, _, n_steps = filter_bands.shape
        # noise.shape = [batch, n_channels, time]
        noise = self.init_noise(batch_size, n_steps, filter_bands.device)

        # Get frames
        padded_noise = F.pad(noise, (N_FFT // 2, N_FFT // 2))
        # noise_frames.shape = [batch, n_channels, n_sample (window_length), n_frames (time)]
        noise_frames = padded_noise.unfold(-1, N_FFT, HOP_LENGTH)

        # Stretch filter to window_length // 2
        filter_ = rearrange(filter_bands, "b c f t -> (b t) c f")
        filter_ = F.interpolate(filter_, size=N_FFT // 2, mode="nearest")
        filter_ = rearrange(filter_, "(b t) c f -> b c t f", b=batch_size, t=n_steps)

        # Prepend 0 DC offset
        dc = torch.zeros(*filter_.shape[:-1], 1).to(filter_.device)
        filter_ = torch.concat([dc, filter_], dim=-1)

        # apply filter to noise
        fft_noise_frames = fft.rfft(noise_frames)
        filtered_fft_noise_frames = filter_ * fft_noise_frames
        filtered_noise_frames = fft.irfft(filtered_fft_noise_frames)
        filtered_noise_frames *= torch.hann_window(N_FFT, periodic=False, device=filter_.device)

        # overlap add
        # I forgot what I have done here, but it seems to work
        b, c = filtered_noise_frames.shape[0], filtered_noise_frames.shape[1]
        stacked_noise = rearrange(filtered_noise_frames, "b c t f -> (b c) f t")
        filtered_noise = F.fold(
            stacked_noise, (1, padded_noise.shape[-1]), (1, N_FFT), stride=(1, HOP_LENGTH)
        )
        filtered_noise = rearrange(filtered_noise, "(b c) 1 1 t -> b c t", b=b, c=c)

        # remove padding and return
        start = N_FFT // 2
        end = -start
        return filtered_noise[:, :, start:end]
