from hydra.core.config_store import ConfigStore
from hydra_zen import builds
from omegaconf import MISSING
from pytorch_lightning import LightningDataModule
from torchvision import datasets, transforms

from src.datamodules import ddsp_datamodule
from src.utils.helpers import random_split
from src.utils.utils import sbuilds

#################
# CIFAR10 Dataset
#################

# transforms.Compose takes a list of transforms
# - Each transform can be configured and appended to the list

# Define a function to split the dataset into train and validation sets

# The base configuration for torchvision.dataset.CIFAR10
# - `transform` is left as None and defined later


# Uses the classmethod `LightningDataModule.from_datasets`
# - Each dataset is a dataclass with training or testing transforms
CIFARNormalize = builds(
    transforms.Normalize,
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010],
)
TrainTransforms = builds(
    transforms.Compose,
    transforms=[
        builds(transforms.RandomCrop, size=32, padding=4),
        builds(transforms.RandomHorizontalFlip),
        builds(transforms.ColorJitter, brightness=0.25, contrast=0.25, saturation=0.25),
        builds(transforms.RandomRotation, degrees=2),
        builds(transforms.ToTensor),
        CIFARNormalize,
    ],
)
TestTransforms = builds(
    transforms.Compose,
    transforms=[builds(transforms.ToTensor), CIFARNormalize],
)
SplitDataset = sbuilds(random_split, dataset=MISSING)
CIFAR10 = builds(
    datasets.CIFAR10,
    root=MISSING,
    train=True,
    transform=None,
    download=True,
)
CIFAR10DataModule = builds(
    LightningDataModule.from_datasets,
    num_workers=4,
    batch_size=256,
    train_dataset=SplitDataset(
        dataset=CIFAR10(root="${...root}", transform=TrainTransforms),
        train=True,
    ),
    val_dataset=SplitDataset(
        dataset=CIFAR10(root="${...root}", transform=TestTransforms),
        train=False,
    ),
    test_dataset=CIFAR10(root="${..root}", transform=TestTransforms, train=False),
    zen_meta=dict(root="${paths.data_dir}"),
)

DDSPDataModule = sbuilds(ddsp_datamodule.DDSPDataModule)


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(group="data", name="cifar10", node=CIFAR10DataModule)
    cs.store(group="data", name="ddsp", node=DDSPDataModule)
