import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split as _random_split


def detect_anomaly(var: torch.Tensor, name: str):
    isnan = any(torch.any(torch.isnan(i)) for i in var)
    isinf = any(torch.any(torch.isinf(i)) for i in var)
    isneginf = any(torch.any(torch.isneginf(i)) for i in var)

    var_min = var.min()
    var_max = var.max()

    print(f"{name} min = {var_min} | {name} max = {var_max}")

    if isnan:
        print(f"{name}  contains nan!")
        exit()
    if isinf:
        print(f"{name}  contains inf!")
        exit()
    if isneginf:
        print(f"{name}  contains neginf!")
        exit()


def bins_to_cents(bins):
    cents = 20 * bins + 1997.3794084376191

    # Trade quantization error for noise
    return cents


def cents_to_freqs(cents):
    return 10 * 2 ** (cents / 1200)


def freqs_to_cents(freq):
    return 1200 * torch.log2(freq / 10.0)


def cents_to_bins(cents):
    return (cents - 1997.3794084376191) / 20


def random_split(
    dataset: Dataset,
    val_split: float = 0.1,
    train: bool = True,
    random_seed: int = 32,
) -> Dataset:
    g = torch.Generator().manual_seed(random_seed)
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_data, val_data = _random_split(dataset, [n_train, n_val], g)
    if train:
        return train_data
    return val_data
