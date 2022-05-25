import torch
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.features = torch.load(path)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]['f0'], self.features[idx]['loudness'], self.features[idx]['audio']
