# Copyright (c) 2021 Massachusetts Institute of Technology
import inspect

import hydra
import pyrootutils
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.nn import Module

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)


# Experiment Task Function
def task_fn(cfg: DictConfig) -> Module:
    # Set seed BEFORE instantiating anything
    pl.seed_everything(cfg.random_seed, workers=True)

    # Data and Lightning Modules
    data = instantiate(cfg.data)
    pl_module = instantiate(cfg.model)

    # Load a checkpoint if defined
    if cfg.ckpt_path is not None:
        ckpt_data = torch.load(cfg.ckpt_path)
        assert "state_dict" in ckpt_data
        pl_module.load_state_dict(ckpt_data["state_dict"])

    # The PL Trainer
    trainer = instantiate(cfg.trainer)

    # Set training or testing mode
    if cfg.testing:
        trainer.test(pl_module, datamodule=data)
    else:
        trainer.fit(pl_module, datamodule=data)

    return pl_module


@hydra.main(version_base="1.2", config_path=None, config_name="config")
def main(cfg: DictConfig):
    return task_fn(cfg)


# loads all modules under `src.configs` package
def register_configs():
    import src.configs

    for name, obj in inspect.getmembers(src.configs):
        if not name.startswith("_") and not name == "inspect":
            obj.register_configs()


if __name__ == "__main__":
    register_configs()

    main()
