from hydra.conf import HydraConf
from hydra.core.config_store import ConfigStore


def register_configs() -> None:
    Hydra = HydraConf()

    Hydra.run.dir = "${paths.log_dir}/${task_name}/runs/${now:%Y-%m-%d}_${now:%H-%M-%S}"
    Hydra.sweep.dir = "${paths.log_dir}/${task_name}/multiruns/${now:%Y-%m-%d}_${now:%H-%M-%S}"
    Hydra.sweep.subdir = "${hydra.job.num}"
    Hydra.defaults.extend(
        [{"override hydra_logging": "colorlog"}, {"override job_logging": "colorlog"}]
    )

    cs = ConfigStore.instance()
    cs.store(
        group="hydra",
        name="config",
        node=Hydra,
        provider="hydra",
    )
