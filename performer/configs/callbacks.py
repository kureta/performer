from hydra.core.config_store import ConfigStore
from hydra_zen import builds
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichModelSummary

from performer.utils.utils import sbuilds

RichModelSummaryConf = sbuilds(
    RichModelSummary,
    max_depth=-1,
)

ModelCheckpointConf = sbuilds(
    ModelCheckpoint,
    dirpath="${paths.output_dir}/checkpoints",
    filename="epoch_{epoch:03d}",
    monitor="val/loss",
    mode="min",
    save_last=True,
    auto_insert_metric_name=False,
)


EarlyStoppingConf = sbuilds(
    EarlyStopping,
    monitor="val/loss",
    patience=20,
    mode="min",
)


DefaultCallbacksConf = builds(
    list,
    [
        RichModelSummaryConf,
        ModelCheckpointConf,
        EarlyStoppingConf,
    ],
)


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(group="trainer/callbacks", name="default", node=DefaultCallbacksConf)
