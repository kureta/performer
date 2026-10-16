import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from .components.ddsp_dataset import DDSPDataset


class DDSPDataModule(LightningDataModule):
    def __init__(
        self,
        data_path,
        wav_dir,
        train_val_test_split,
        example_duration=4,
        example_hop_length=1,
        batch_size=8,
        num_workers=8,
        pin_memory=True,
        fmin=31.7,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self):
        DDSPDataset(
            self.hparams.data_path,
            self.hparams.wav_dir,
            example_duration=self.hparams.example_duration,
            example_hop_length=self.hparams.example_hop_length,
            fmin=self.hparams.fmin,
        )

    def setup(self, stage=None):
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = DDSPDataset(
                self.hparams.data_path,
                self.hparams.wav_dir,
                example_duration=self.hparams.example_duration,
                example_hop_length=self.hparams.example_hop_length,
                fmin=self.hparams.fmin,
            )
            train, val, test = (int(x * len(dataset)) for x in self.hparams.train_val_test_split)
            test += len(dataset) - (train + test + val)
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=[train, val, test],
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
