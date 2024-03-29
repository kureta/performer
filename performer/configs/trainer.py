import torch
from hydra.core.config_store import ConfigStore
from hydra_zen import builds
from pytorch_lightning import Trainer

###################
# Lightning Trainer
###################
TrainerConf = builds(
    Trainer,
    callbacks=None,  # easily build a list of callbacks
    accelerator="gpu",
    devices=builds(torch.cuda.device_count),  # use all GPUs on the system
    num_nodes=1,
    max_epochs=10000,
    populate_full_signature=True,
)


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(group="trainer", name="trainer", node=TrainerConf)
