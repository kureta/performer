import torch
from hydra.core.config_store import ConfigStore
from hydra_zen import builds
from torchmetrics import Accuracy, MetricCollection

from src.configs.optimizer import StepLR
from src.models.model import (
    BaseImageClassification,
    ResNet18Classifier,
    ResNet50Classifier,
)

##########################
# PyTorch Lightning Module
##########################
ImageClassification = builds(
    BaseImageClassification,
    optim=None,
    predict=builds(torch.nn.Softmax, dim=1),
    criterion=builds(torch.nn.CrossEntropyLoss),
    lr_scheduler=StepLR,
    metrics=builds(
        MetricCollection,
        builds(dict, Accuracy=builds(Accuracy)),
        hydra_convert="all",
    ),
)
ResNet18 = builds(ResNet18Classifier, builds_bases=(ImageClassification,))
ResNet50 = builds(ResNet50Classifier, builds_bases=(ImageClassification,))


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(group="model", name="resnet18", node=ResNet18)
    cs.store(group="model", name="resnet50", node=ResNet50)
