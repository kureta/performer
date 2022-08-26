# Copyright (c) 2021 Massachusetts Institute of Technology
from pathlib import Path

from hydra.core.config_store import ConfigStore
from hydra_zen import MISSING, make_config

# Experiment Configs
# - Replaces config.yaml
Config = make_config(
    #
    # Experiment Defaults: See https://hydra.cc/docs/next/advanced/defaults_list
    defaults=[
        "_self_",  # See https://hydra.cc/docs/upgrades/1.0_to_1.1/default_composition_order
        {"data": "cifar10"},
        {"model": "resnet18"},
        {"model/optim": "sgd"},
        {"trainer": "trainer"},
        {"trainer/callbacks": "default"},
        {"trainer/logger": "multi"},
        {"paths": "paths"},
    ],
    #
    # Experiment Modules
    paths=MISSING,
    data=MISSING,
    model=MISSING,
    trainer=MISSING,
    #
    # Experiment Constants
    random_seed=928,
    testing=False,
    ckpt_path=None,
    task_name="train",
)


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
