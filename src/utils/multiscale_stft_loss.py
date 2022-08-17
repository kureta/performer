import torch
import torch.nn.functional as F
from einops import rearrange


def multiscale_stft(signal, scales, overlap):
    """
    Compute a stft on several scales, with a constant overlap value.
    Parameters
    ----------
    signal: torch.Tensor
        input signal to process ( B X C X T )

    scales: list
        scales to use
    overlap: float
        overlap between windows ( 0 - 1 )
    """
    signal = rearrange(signal, "b c t -> (b c) t")
    stfts = []
    for s in scales:
        S = torch.stft(  # noqa
            signal,
            s,
            int(s * (1 - overlap)),
            s,
            torch.hann_window(s).to(signal),
            center=True,
            normalized=True,
            return_complex=True,
        ).abs()
        stfts.append(S)
    return stfts


def lin_distance(x, y):
    return torch.norm(x - y)


def log_distance(x, y):
    return torch.norm(torch.log(x + 1e-7) - torch.log(y + 1e-7))


def log_l1_loss(x, y):
    return F.l1_loss(torch.log(x + 1e-7), torch.log(y + 1e-7))


def distance(x, y):
    scales = [4096, 2048, 1024, 512, 256, 128]
    x = multiscale_stft(x, scales, 0.75)
    y = multiscale_stft(y, scales, 0.75)

    lin = sum(map(F.l1_loss, x, y)) / len(scales)
    log = sum(map(log_l1_loss, x, y)) / len(scales)

    return lin + log
