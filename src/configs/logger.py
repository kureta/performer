import pytorch_lightning.loggers
from hydra.core.config_store import ConfigStore

from src.utils.utils import sbuilds

WandBConf = sbuilds(pytorch_lightning.loggers.WandbLogger)


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(group="trainer/logger", name="wandb", node=WandBConf)
