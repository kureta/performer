from hydra.conf import HydraConf, OverridesConf, RunDir, SweepDir

Hydra = HydraConf(
    defaults=[{"override hydra_logging": "colorlog"}, {"override job_logging": "colorlog"}],
    run=RunDir("${paths.log_dir}/${task_name}/runs/${now:%Y-%m-%d}_${now:%H-%M-%S}"),
    sweep=SweepDir(
        dir="${paths.log_dir}/${task_name}/multiruns/boklar/${now:%Y-%m-%d}_${now:%H-%M-%S}",
        subdir="${hydra.job.num}",
    ),
)


def register_configs() -> None:
    pass
