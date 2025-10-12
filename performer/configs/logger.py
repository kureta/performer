import pytorch_lightning.loggers
from hydra.core.config_store import ConfigStore
from hydra_zen import builds

from performer.utils.utils import sbuilds

WandBConf = sbuilds(
    pytorch_lightning.loggers.WandbLogger,
    save_dir="${paths.output_dir}",
    offline=False,
    id=None,
    anonymous=None,
    project="performer",
    log_model=True,
    prefix="",
    group="",
    tags=[],
    job_type="",
)
CVSConf = sbuilds(
    pytorch_lightning.loggers.CSVLogger,
    save_dir="${paths.output_dir}",
    name="cvs",
    prefix="",
)
TensorboardConf = sbuilds(
    pytorch_lightning.loggers.TensorBoardLogger,
    save_dir="${paths.output_dir}",
    name=None,
    log_graph=True,
    default_hp_metric=True,
    prefix="",
)


DefaultLoggerConf = builds(
    list,
    [
        WandBConf,
        CVSConf,
        TensorboardConf,
    ],
)


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(group="trainer/logger", name="wandb", node=WandBConf)
    cs.store(group="trainer/logger", name="cvs", node=CVSConf)
    cs.store(group="trainer/logger", name="tensorboard", node=TensorboardConf)
    cs.store(group="trainer/logger", name="multi", node=DefaultLoggerConf)
