from dataclasses import dataclass

from hydra.core.config_store import ConfigStore


@dataclass
class PathsConfig:
    root_dir: str = "${oc.env:PROJECT_ROOT}"
    data_dir: str = "${paths.root_dir}/data"
    log_dir: str = "${paths.root_dir}/logs"
    output_dir: str = "${hydra:runtime.output_dir}"
    work_dir: str = "${hydra:runtime.cwd}"


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(group="paths", name="paths", node=PathsConfig)
