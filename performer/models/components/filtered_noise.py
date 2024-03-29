import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F  # noqa


class FilteredNoise(nn.Module):
    def __init__(
        self,
        n_bands: int = 128,
        n_channels: int = 1,
    ):
        super().__init__()

        self.n_bands = n_bands
        self.n_channels = n_channels
        win = torch.hann_window(3072, periodic=False)
        self.register_buffer("win", win, persistent=False)

    def init_noise(self, batch_size: int, n_steps: int, device: torch.device) -> torch.Tensor:
        noise = torch.rand(batch_size, self.n_channels, (n_steps - 1) * 192, device=device)
        noise = noise * 2.0 - 1.0

        return noise

    def forward(self, filter_bands: torch.Tensor) -> torch.Tensor:
        # filter_bands.shape = [batch, n_channels, n_bands, time]

        # Generate white noise
        batch_size, _, _, n_steps = filter_bands.shape
        # noise.shape = [batch, n_channels, time]
        noise = self.init_noise(batch_size, n_steps, filter_bands.device)

        # Get frames
        padded_noise = F.pad(noise, (3072 // 2, 3072 // 2))
        # noise_frames.shape = [batch, n_channels, n_sample (window_length), n_frames (time)]
        noise_frames = padded_noise.unfold(-1, 3072, 192)

        b, c, f, t = filter_bands.shape
        # b c f t -> b t c f
        filter_ = filter_bands.permute((0, 3, 1, 2))
        # b t c f -> (b t) c f
        filter_ = filter_.reshape((b * t, c, f))

        # Stretch filter to window_length // 2
        filter_ = F.interpolate(filter_, size=3072 // 2, mode="nearest")

        bt, c, f = filter_.shape
        # t = torch.div(bt, b, rounding_mode="trunc")
        # (b t) c f -> b t c f
        filter_ = filter_.reshape((b, t, c, f))
        # b t c f -> b c t f
        filter_ = filter_.permute((0, 2, 1, 3))

        # Prepend 0 DC offset
        dc = torch.zeros_like(filter_[..., 0:1]).to(filter_.device)
        filter_ = torch.concat([dc, filter_], dim=-1)

        # apply filter to noise
        fft_noise_frames = fft.rfft(noise_frames)
        filtered_fft_noise_frames = filter_ * fft_noise_frames
        filtered_noise_frames = fft.irfft(filtered_fft_noise_frames)
        self.win = self.win.to(filter_.device)
        filtered_noise_frames *= self.win

        # overlap add
        # I forgot what I have done here, but it seems to work
        b, c, t, f = filtered_noise_frames.shape
        # b c t f -> b c f t
        stacked_noise = filtered_noise_frames.permute((0, 1, 3, 2))
        # b c f t -> (b c) f t
        stacked_noise = stacked_noise.reshape((b * c, f, t))

        filtered_noise = F.fold(
            stacked_noise, (1, padded_noise.shape[-1]), (1, 3072), stride=(1, 192)
        )
        bc, _, _, t = filtered_noise.shape
        # c = torch.div(bc, b, rounding_mode="trunc")
        # (b c) 1 1 t -> b c t
        filtered_noise = filtered_noise.reshape((b, c, t))

        # remove padding and return
        start = 3072 // 2
        end = -start
        return filtered_noise[:, :, start:end]
