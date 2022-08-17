import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from .components.ddsp_dataset import DDSPDataset


class DDSPDataModule(LightningDataModule):
    def __init__(
        self,
        data_path,
        wav_dir,
        example_duration,
        example_hop_length,
        train_val_test_split,
        batch_size,
        num_workers,
        pin_memory,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self):
        DDSPDataset(self.hparams.data_path, self.hparams.wav_dir)

    def setup(self, stage=None):
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = DDSPDataset(
                self.hparams.data_path,
                self.hparams.wav_dir,
                example_duration=self.hparams.example_duration,
                example_hop_length=self.hparams.example_hop_length,
            )
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
