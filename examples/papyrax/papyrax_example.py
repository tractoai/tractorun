import logging
import os
from pathlib import Path
import sys
from typing import (
    Any,
    List,
    Optional,
)

from papyrax.training.config import TrainerConfig
from papyrax.training.trainer import train_model
from papyrax.utils.config import ConfigBase
from papyrax.utils.debug import enable_debugging
from papyrax.utils.logging import configure_logging
from papyrax.utils.nccl import run_nccl_all_reduce_perf_test
from papyrax.utils.tqdm import logging_redirect_tqdm

from tractorun.mesh import Mesh
from tractorun.resources import Resources
from tractorun.run import run
from tractorun.toolbox import Toolbox


LOGGER = logging.getLogger(__name__)


class MeshConfig(ConfigBase):
    node_count: int
    process_per_node: int
    gpu_per_process: int
    pool_trees: Optional[List[str]]


class ResourcesConfig(ConfigBase):
    cpu_limit: Optional[float]
    memory_limit: Optional[int]


class Config(ConfigBase):
    training_root: str
    config_path: str
    config_overrides: List[str]
    override_values: dict[str, Any]
    docker_image: str

    mesh: MeshConfig
    resources: ResourcesConfig


def train(
    config: TrainerConfig,
    log_level: int,
    logging_config_path: Path | None,
    enable_cloud_logging: bool,
    toolbox: Toolbox,
) -> None:
    # Some large models require a higher recursion limit
    sys.setrecursionlimit(10000)

    configure_logging(
        log_level=log_level,
        logging_config_path=logging_config_path,
        enable_cloud_logging=enable_cloud_logging,
        runtime_attributes={
            "checkpoint_dir": (str(config.checkpointing.checkpoint_dir) if config.checkpointing is not None else None),
            "rank": toolbox.mesh.node_count,
        },
    )

    if config.run_nccl_tests_before_training:
        LOGGER.info("Running NCCL tests before training")
        run_nccl_all_reduce_perf_test()

    with enable_debugging(), logging_redirect_tqdm():
        train_model(config)


def main() -> None:
    main_local_papyrax()


def main_local_papyrax() -> None:
    config: Config = Config.from_cli([os.environ.get("CHIFFA_CONFIG_PATH")], [])
    trainer_config: TrainerConfig = TrainerConfig.from_cli(
        [config.config_path], config.config_overrides, config.override_values
    )

    mesh = Mesh(
        node_count=config.mesh.node_count,
        process_per_node=config.mesh.process_per_node,
        gpu_per_process=config.mesh.gpu_per_process,
        pool_trees=config.mesh.pool_trees,
    )

    resources = Resources(
        cpu_limit=config.resources.cpu_limit,
        memory_limit=config.resources.memory_limit,
    )

    def wrapped_train(toolbox: Toolbox) -> None:
        train(
            config=trainer_config,
            log_level=logging.DEBUG,
            logging_config_path=None,
            enable_cloud_logging=False,
            toolbox=toolbox,
        )

    run(
        wrapped_train,
        yt_path=config.training_root,
        mesh=mesh,
        resources=resources,
        docker_image=config.docker_image,
        local=False,
    )


if __name__ == "__main__":
    main()
