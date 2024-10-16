import json
import os
import pathlib
import typing as t

import pytest

from tests.utils import (
    DOCKER_IMAGE,
    TractoCli,
    get_data_path,
    make_cli_args,
    make_run_config,
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


@pytest.fixture
def cypress_file(yt_instance: YtInstance, yt_path: str) -> str:
    yt_client = yt_instance.get_client()
    file_path = f"{yt_path}/cypress_file"
    yt_client.write_file(file_path, b"hello")
    yt_client.set_attribute(file_path, "executable", True)
    return file_path


def test_configuration_cypress() -> None:
    _, _, config = make_configuration(
        make_cli_args("--bind-cypress", "//tmp/cli:cli", "command"),
    )
    assert config.bind_cypress == [BindCypress(source="//tmp/cli", destination="cli")]

    run_config = make_run_config(
        {
            "bind_cypress": ["//tmp/config:config"],
        }
    )
    with run_config_file(run_config) as run_config_path:
        _, _, config = make_configuration(["--run-config-path", run_config_path])
    assert config.bind_cypress == [BindCypress(source="//tmp/config", destination="config")]

    with run_config_file(run_config) as run_config_path:
        _, _, config = make_configuration(
            make_cli_args("--run-config-path", run_config_path, "--bind-cypress", "//tmp/cli:cli"),
        )
    assert config.bind_cypress == [BindCypress(source="//tmp/cli", destination="cli")]


def test_configuration_local() -> None:
    _, _, config = make_configuration(make_cli_args("--bind-local", "//tmp/cli:/tmp/cli"))
    assert config.bind_local == [BindLocal(source="//tmp/cli", destination="/tmp/cli")]

    run_config = make_run_config(
        {
            "bind_local": ["//tmp/config:/tmp/config"],
        }
    )
    with run_config_file(run_config) as run_config_path:
        _, _, config = make_configuration(["--run-config-path", run_config_path])
    assert config.bind_local == [BindLocal(source="//tmp/config", destination="/tmp/config")]

    with run_config_file(run_config) as run_config_path:
        _, _, config = make_configuration(["--run-config-path", run_config_path, "--bind-local", "//tmp/cli:/tmp/cli"])
    assert config.bind_local == [BindLocal(source="//tmp/cli", destination="/tmp/cli")]


def checker_dir(toolbox: Toolbox) -> None:
    assert pathlib.Path(DIR_PATH).is_dir()
    assert (pathlib.Path(DIR_PATH) / "some_file").is_file()


def get_file_checker(file_path: str = FILE_PATH, check_executable: bool = False) -> t.Callable[[Toolbox], None]:
    def checker_file(toolbox: Toolbox) -> None:
        assert pathlib.Path(file_path).is_file()
        if check_executable:
            assert os.access(file_path, os.X_OK)

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


def test_cypress_bind_file(yt_instance: YtInstance, yt_path: str, cypress_file: str) -> None:
    yt_client = yt_instance.get_client()

    destination_path = "bar"

    run(
        get_file_checker(destination_path, check_executable=True),
        backend=GenericBackend(),
        yt_path=yt_path,
        binds_cypress=[
            BindCypress(
                source=cypress_file,
                destination=destination_path,
            ),
        ],
        mesh=Mesh(node_count=1, process_per_node=1, gpu_per_process=0),
        yt_client=yt_client,
        docker_image=DOCKER_IMAGE,
    )


def test_cypress_bind_from_run_config(yt_instance: YtInstance, yt_path: str, cypress_file: str) -> None:
    yt_client = yt_instance.get_client()

    run_config = {
        "mesh": {
            "node_count": 1,
            "process_per_node": 1,
            "gpu_per_process": 0,
        },
        "bind_cypress": [
            f"{cypress_file}:bar",
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
