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

# from papyrax.utils.nccl import run_nccl_all_reduce_perf_test
from papyrax.utils.tqdm import logging_redirect_tqdm

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
    log_level: int,
    logging_config_path: Path | None,
    enable_cloud_logging: bool,
    toolbox: Toolbox,
) -> None:
    trainer_config: TrainerConfig = TrainerConfig.from_cli(
        os.environ["CHIFFA_CONFIG_PATH"],
        [],
        {},
    )
    # Some large models require a higher recursion limit
    sys.setrecursionlimit(10000)

    configure_logging(
        log_level=log_level,
        logging_config_path=logging_config_path,
        enable_cloud_logging=enable_cloud_logging,
        runtime_attributes={
            "checkpoint_dir": None,
            "rank": toolbox.mesh.node_count,
        },
    )

    # if config.run_nccl_tests_before_training:
    #     LOGGER.info("Running NCCL tests before training")
    #     run_nccl_all_reduce_perf_test()

    with enable_debugging(), logging_redirect_tqdm():
        train_model(trainer_config)


if __name__ == "__main__":
    train()
