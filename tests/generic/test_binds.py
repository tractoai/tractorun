import json
import pathlib
import typing as t

from tests.utils import (
    DOCKER_IMAGE,
    TractoCli,
    get_data_path,
    run_config_file,
)
from tests.yt_instances import YtInstance
from tractorun.backend.generic import GenericBackend
from tractorun.bind import (
    BindCypress,
    BindLocal,
)
from tractorun.cli.tractorun_runner import make_configuration
from tractorun.mesh import Mesh
from tractorun.run import run
from tractorun.toolbox import Toolbox


FILE_PATH = "/bind/file"
DIR_PATH = "/bind/dir"


def checker_dir(toolbox: Toolbox) -> None:
    assert pathlib.Path(DIR_PATH).is_dir()
    assert (pathlib.Path(DIR_PATH) / "some_file").is_file()


def get_file_checker(file_path: str = FILE_PATH) -> t.Callable[[Toolbox], None]:
    def checker_file(toolbox: Toolbox) -> None:
        assert pathlib.Path(file_path).is_file()

    return checker_file


def test_local_bind_file_pickle(yt_instance: YtInstance, yt_path: str) -> None:
    yt_client = yt_instance.get_client()

    run(
        get_file_checker(),
        backend=GenericBackend(),
        yt_path=yt_path,
        binds_local=[
            BindLocal(
                source=get_data_path("../data/binds/another_file"),
                destination=FILE_PATH,
            ),
        ],
        mesh=Mesh(node_count=1, process_per_node=1, gpu_per_process=0),
        yt_client=yt_client,
        docker_image=DOCKER_IMAGE,
    )


def test_local_bind_dir_pickle(yt_instance: YtInstance, yt_path: str) -> None:
    yt_client = yt_instance.get_client()

    run(
        checker_dir,
        backend=GenericBackend(),
        yt_path=yt_path,
        binds_local=[
            BindLocal(
                source=get_data_path("../data/binds/bind_dir"),
                destination=DIR_PATH,
            ),
        ],
        mesh=Mesh(node_count=1, process_per_node=1, gpu_per_process=0),
        yt_client=yt_client,
        docker_image=DOCKER_IMAGE,
    )


def test_cypress_bind_file(yt_instance: YtInstance, yt_path: str) -> None:
    yt_client = yt_instance.get_client()

    cypress_file_path = "//tmp/foo"
    destination_path = "bar"
    yt_client.write_file(cypress_file_path, b"hello")

    run(
        get_file_checker(destination_path),
        backend=GenericBackend(),
        yt_path=yt_path,
        binds_cypress=[
            BindCypress(
                source=cypress_file_path,
                destination=destination_path,
            ),
        ],
        mesh=Mesh(node_count=1, process_per_node=1, gpu_per_process=0),
        yt_client=yt_client,
        docker_image=DOCKER_IMAGE,
    )


def test_cypress_bind_from_run_config(yt_instance: YtInstance, yt_path: str) -> None:
    yt_client = yt_instance.get_client()

    cypress_file_to_check = "//tmp/foo"
    yt_client.write_file(cypress_file_to_check, b"hello")

    run_config = {
        "mesh": {
            "node_count": 1,
            "process_per_node": 1,
            "gpu_per_process": 0,
        },
        "bind_cypress": [
            "//tmp/foo:bar",
        ],
    }

    with run_config_file(run_config) as run_config_path:
        tracto_cli = TractoCli(
            command=["python3", "/tractorun_tests/check_cypress_bind.py"],
            args=[
                "--run-config-path",
                run_config_path,
                "--yt-path",
                yt_path,
                "--bind-local",
                f"{get_data_path('../data/check_cypress_bind.py')}:/tractorun_tests/check_cypress_bind.py",
                "--user-config",
                json.dumps({"CYPRESS_FILE_TO_CHECK": "./bar"}),
            ],
        )
        op_run = tracto_cli.run()

    assert op_run.is_exitcode_valid()
    assert op_run.is_operation_state_valid(yt_client=yt_client, job_count=1)


def test_cypress_binds_config_from_cli():
    _, _, config = make_configuration(["--yt-path", "foo", "--bind-cypress", "//tmp/foo:bar", "command"])
    assert config.bind_cypress == [BindCypress(source="//tmp/foo", destination="bar")]
