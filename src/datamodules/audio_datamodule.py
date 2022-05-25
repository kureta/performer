from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .components.audio_dataset import AudioDataset


class AudioDataModule(LightningDataModule):
    def __init__(self, batch_size, data_dir, num_workers, pin_memory):
        super().__init__()
        self.save_hyperparameters()

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        self.train_dataset = AudioDataset(self.hparams.data_dir)
        self.val_dataset = AudioDataset(self.hparams.data_dir)
        self.val_dataset.features = self.val_dataset.features[:64]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
