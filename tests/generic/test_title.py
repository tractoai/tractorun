import uuid

from tests.utils import (
    DOCKER_IMAGE,
    make_cli_args,
    make_run_config,
    run_config_file,
)
from tractorun.backend.generic import GenericBackend
from tractorun.cli.tractorun_runner import make_configuration
from tractorun.mesh import Mesh
from tractorun.run import run
from tractorun.toolbox import Toolbox


def test_configuration() -> None:
    _, _, config = make_configuration(make_cli_args("--title", "cli title"))
    assert config.title == "cli title"

    run_config = make_run_config({"title": "config title"})
    with run_config_file(run_config) as run_config_path:
        _, _, config = make_configuration(["--run-config-path", run_config_path])
    assert config.title == "config title"

    with run_config_file(run_config) as run_config_path:
        _, _, config = make_configuration(["--run-config-path", run_config_path, "--title", "cli title"])
    assert config.title == "cli title"


def test_run_pickle(yt_path: str) -> None:
    def checker(toolbox: Toolbox) -> None:
        pass

    title = f"super title {uuid.uuid4()}"

    run_info = run(
        checker,
        yt_path=yt_path,
        title=title,
        mesh=Mesh(node_count=1, process_per_node=1, gpu_per_process=0),
        backend=GenericBackend(),
        docker_image=DOCKER_IMAGE,
        dry_run=True,
    )
    assert run_info.operation_spec["title"] == title
