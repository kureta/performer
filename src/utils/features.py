import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF
import torchcrepe
from torchmetrics import MeanAbsoluteError

from .constants import (
    CREPE_HOP_LENGTH,
    CREPE_N_BINS,
    CREPE_SAMPLE_RATE,
    HOP_LENGTH,
    N_FFT,
    SAMPLE_RATE,
)
from .helpers import bins_to_cents, cents_to_bins, cents_to_freqs, freqs_to_cents


class LoudnessMAE(MeanAbsoluteError):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loudness = Loudness()

    def update(self, pred, target):
        amp_pred = self.loudness.get_amp(pred)
        amp_target = self.loudness.get_amp(target)
        super().update(amp_pred, amp_target)


class Loudness(nn.Module):
    def __init__(self):
        super().__init__()
        frequencies = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT).astype("float32")
        a_weighting = librosa.A_weighting(frequencies)[None, :].astype("float32")
        self.register_buffer("a_weighting", torch.from_numpy(a_weighting))

    def get_amp(self, x):
        # to mono
        x = x.mean(1)
        window = torch.hann_window(N_FFT).to(x.device)
        s = torch.stft(
            x,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            window=window,
            return_complex=True,
            # pad_mode="reflect",
            center=False,
            normalized=True,
        ).transpose(1, 2)

        # Compute power.
        amplitude = torch.abs(s)
        power = amplitude**2

        weighting = 10 ** (self.a_weighting / 10)
        power = power * weighting

        power = torch.mean(power, dim=-1)
        loudness = 10.0 * torch.log10(
            torch.maximum(torch.tensor(1e-10, device=power.device), power)
        )
        loudness = torch.maximum(loudness, loudness.max() - 80.0)

        return loudness.unsqueeze(1)


def normalize_f0(x):
    return cents_to_bins(freqs_to_cents(x)) / (CREPE_N_BINS - 1)


def denormalize_f0(x):
    return cents_to_freqs(bins_to_cents(x * (CREPE_N_BINS - 1)))


def get_f0(x, batch_size=128):
    device = x.device
    # to mono and resample
    x = AF.resample(x.mean(1), SAMPLE_RATE, CREPE_SAMPLE_RATE)

    f0 = torchcrepe.predict(
        x,
        sample_rate=CREPE_SAMPLE_RATE,
        hop_length=CREPE_HOP_LENGTH,
        pad=False,
        fmin=31.7,
        batch_size=batch_size,
        decoder=torchcrepe.decode.weighted_argmax,
        device=device,
        return_periodicity=False,
    )

    return f0.unsqueeze(1)


def get_normalized_f0(self, x):
    return normalize_f0(self.get_f0(x))
