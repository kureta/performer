import torch
from hydra.core.config_store import ConfigStore

from src.utils.utils import pbuilds

####################################
# PyTorch Optimizer and LR Scheduler
####################################
SGD = pbuilds(torch.optim.SGD, lr=0.1, momentum=0.9)
Adam = pbuilds(torch.optim.Adam, lr=0.1)
StepLR = pbuilds(torch.optim.lr_scheduler.StepLR, step_size=50, gamma=0.1)


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(group="model/optim", name="sgd", node=SGD)
    cs.store(group="model/optim", name="adam", node=Adam)
    cs.store(group="model/optim", name="none", node=None)
