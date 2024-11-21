import uuid

from tests.utils import (
    GENERIC_DOCKER_IMAGE,
    make_cli_args,
    make_run_config,
    run_config_file,
)
from tractorun.backend.generic import GenericBackend
from tractorun.cli.tractorun_runner import make_configuration
from tractorun.mesh import Mesh
from tractorun.run import run
from tractorun.toolbox import Toolbox


def test_configuration_pool() -> None:
    _, _, config = make_configuration(make_cli_args("--mesh.pool", "cli"))
    assert config.mesh.pool == "cli"

    run_config = make_run_config({"mesh": {"pool": "config"}})
    with run_config_file(run_config) as run_config_path:
        _, _, config = make_configuration(["--run-config-path", run_config_path])
    assert config.mesh.pool == "config"

    with run_config_file(run_config) as run_config_path:
        _, _, config = make_configuration(["--run-config-path", run_config_path, "--mesh.pool", "cli"])
    assert config.mesh.pool == "cli"


def test_configuration_pool_trees() -> None:
    _, _, config = make_configuration(make_cli_args("--mesh.pool-trees", "cli1", "--mesh.pool-trees", "cli2"))
    assert config.mesh.pool_trees == ["cli1", "cli2"]

    run_config = make_run_config({"mesh": {"pool_trees": ["config1", "config2"]}})
    with run_config_file(run_config) as run_config_path:
        _, _, config = make_configuration(["--run-config-path", run_config_path])
    assert config.mesh.pool_trees == ["config1", "config2"]

    with run_config_file(run_config) as run_config_path:
        _, _, config = make_configuration(
            make_cli_args(
                "--run-config-path", run_config_path, "--mesh.pool-trees", "cli1", "--mesh.pool-trees", "cli2"
            )
        )
    assert config.mesh.pool_trees == ["cli1", "cli2"]


def test_run_pickle(yt_path: str) -> None:
    def checker(toolbox: Toolbox) -> None:
        pass

    title = f"super title {uuid.uuid4()}"

    run_info = run(
        checker,
        yt_path=yt_path,
        title=title,
        mesh=Mesh(node_count=1, process_per_node=1, gpu_per_process=0, pool_trees=["some_tree"], pool="some_pool"),
        backend=GenericBackend(),
        docker_image=GENERIC_DOCKER_IMAGE,
        dry_run=True,
    )
    assert run_info.operation_spec["pool"] == "some_pool"
    assert run_info.operation_spec["pool_trees"] == ["some_tree"]
