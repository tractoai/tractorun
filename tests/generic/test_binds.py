import json
import os
import pathlib
import typing as t

import pytest
import yt.wrapper as yt

from tests.utils import (
    GENERIC_DOCKER_IMAGE,
    TractoCli,
    get_data_path,
    make_cli_args,
    make_run_config,
    run_config_file,
)
from tests.yt_instances import YtInstance
from tractorun.backend.generic import GenericBackend
from tractorun.bind import (
    BindAttributes,
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

    _, _, config = make_configuration(
        make_cli_args(
            "--bind-cypress",
            json.dumps(
                {
                    "source": "//tmp/cli",
                    "destination": "cli",
                }
            ),
            "command",
        ),
    )
    assert config.bind_cypress == [BindCypress(source="//tmp/cli", destination="cli", attributes=BindAttributes())]

    _, _, config = make_configuration(
        make_cli_args(
            "--bind-cypress",
            json.dumps(
                {
                    "source": "//tmp/cli",
                    "destination": "cli",
                    "attributes": {"format": "cli"},
                }
            ),
            "command",
        ),
    )
    assert config.bind_cypress == [
        BindCypress(source="//tmp/cli", destination="cli", attributes=BindAttributes(format="cli"))
    ]

    run_config = make_run_config(
        {
            "bind_cypress": [
                {
                    "source": "//tmp/config",
                    "destination": "config",
                    "attributes": {"format": "config"},
                }
            ],
        },
    )
    with run_config_file(run_config) as run_config_path:
        _, _, config = make_configuration(["--run-config-path", run_config_path])
    assert config.bind_cypress == [
        BindCypress(source="//tmp/config", destination="config", attributes=BindAttributes(format="config"))
    ]

    with run_config_file(run_config) as run_config_path:
        _, _, config = make_configuration(
            make_cli_args(
                "--bind-cypress",
                json.dumps(
                    {
                        "source": "//tmp/cli",
                        "destination": "cli",
                        "attributes": {"format": "cli"},
                    }
                ),
                "command",
            ),
        )
    assert config.bind_cypress == [
        BindCypress(source="//tmp/cli", destination="cli", attributes=BindAttributes(format="cli"))
    ]


def test_configuration_local() -> None:
    _, _, config = make_configuration(make_cli_args("--bind-local", "//tmp/cli:/tmp/cli"))
    assert config.bind_local == [BindLocal(source="//tmp/cli", destination="/tmp/cli")]

    _, _, config = make_configuration(
        make_cli_args("--bind-cypress", json.dumps({"source": "//tmp/cli", "destination": "/tmp/cli"}), "command"),
    )
    assert config.bind_cypress == [BindCypress(source="//tmp/cli", destination="/tmp/cli")]

    run_config = make_run_config(
        {
            "bind_local": [
                {
                    "source": "//tmp/config",
                    "destination": "/tmp/config",
                }
            ],
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


def get_file_checker(
    file_paths: list[str] | None = None, check_executable: bool = False
) -> t.Callable[[Toolbox], None]:
    if file_paths is None:
        file_paths = [FILE_PATH]

    def checker_file(toolbox: Toolbox) -> None:
        for file_path in file_paths:
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
                source=str(get_data_path("../data/binds/another_file")),
                destination=FILE_PATH,
            ),
        ],
        mesh=Mesh(node_count=1, process_per_node=1, gpu_per_process=0),
        yt_client=yt_client,
        docker_image=GENERIC_DOCKER_IMAGE,
    )


def test_local_bind_dir_pickle(yt_instance: YtInstance, yt_path: str) -> None:
    yt_client = yt_instance.get_client()

    run(
        checker_dir,
        backend=GenericBackend(),
        yt_path=yt_path,
        binds_local=[
            BindLocal(
                source=str(get_data_path("../data/binds/bind_dir")),
                destination=DIR_PATH,
            ),
        ],
        mesh=Mesh(node_count=1, process_per_node=1, gpu_per_process=0),
        yt_client=yt_client,
        docker_image=GENERIC_DOCKER_IMAGE,
    )


def test_cypress_bind_file(yt_instance: YtInstance, yt_path: str, cypress_file: str) -> None:
    yt_client = yt_instance.get_client()

    destination_path = "bar"

    run(
        get_file_checker([destination_path], check_executable=True),
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
        docker_image=GENERIC_DOCKER_IMAGE,
    )


def test_cypress_bind_file_attrs(yt_instance: YtInstance, yt_path: str, cypress_file: str) -> None:
    yt_client = yt_instance.get_client()

    def checker() -> None:
        pass

    run_info = run(
        checker,
        backend=GenericBackend(),
        yt_path=yt_path,
        mesh=Mesh(node_count=1, process_per_node=1, gpu_per_process=0),
        yt_client=yt_client,
        docker_image=GENERIC_DOCKER_IMAGE,
        dry_run=True,
        binds_cypress=[
            BindCypress(
                source=cypress_file,
                destination="local_file",
                attributes=BindAttributes(
                    executable=False,
                    format="dummy",
                    bypass_artifact_cache=True,
                ),
            )
        ],
    )
    attached_file = run_info.operation_spec["tasks"]["task"]["file_paths"][-1]
    assert str(attached_file) == cypress_file
    assert attached_file.attributes == {
        "executable": False,
        "file_name": "local_file",
        "bypass_artifact_cache": True,
        "format": "dummy",
    }


def test_cypress_bind_map_node(yt_instance: YtInstance, yt_path: str) -> None:
    yt_client = yt_instance.get_client()

    yt_map_path = f"{yt_path}/some_dir"
    yt_client.create("map_node", yt_map_path)
    yt_client.create("map_node", f"{yt_map_path}/nested")

    destination = "some_folder"

    file_suffixes = ["foo", "bar", "nested/some", "nested/another"]
    for file_suffix in file_suffixes:
        file_path = f"{yt_map_path}/{file_suffix}"
        yt_client.write_file(file_path, b"hello")
        yt_client.set_attribute(file_path, "executable", True)

    run(
        get_file_checker(
            file_paths=[f"{destination}/{file_suffix}" for file_suffix in file_suffixes],
            check_executable=True,
        ),
        backend=GenericBackend(),
        yt_path=yt_path,
        binds_cypress=[
            BindCypress(
                source=yt_map_path,
                destination=destination,
            ),
        ],
        mesh=Mesh(node_count=1, process_per_node=1, gpu_per_process=0),
        yt_client=yt_client,
        docker_image=GENERIC_DOCKER_IMAGE,
    )


def test_cypress_bind_map_node_spec(yt_instance: YtInstance, yt_path: str) -> None:
    yt_client = yt_instance.get_client()

    yt_map_path = f"{yt_path}/some_dir"
    yt_client.create("map_node", yt_map_path)
    yt_client.create("map_node", f"{yt_map_path}/nested")

    # test warning for non-file nodes
    yt_client.create("document", f"{yt_map_path}/document")

    destination = "some_local_dir"

    file_suffixes = ["foo", "bar", "nested/some", "nested/another"]
    for file_suffix in file_suffixes:
        file_path = f"{yt_map_path}/{file_suffix}"
        yt_client.write_file(file_path, b"hello")
        yt_client.set_attribute(file_path, "executable", True)

    with pytest.warns(UserWarning, match="Skip .*/document because it is not a file, but document"):
        run_info = run(
            get_file_checker(file_suffixes, check_executable=True),
            backend=GenericBackend(),
            yt_path=yt_path,
            binds_cypress=[
                BindCypress(
                    source=f"{yt_map_path}",
                    destination=destination,
                    attributes=BindAttributes(
                        executable=False,
                        format="dummy",
                        bypass_artifact_cache=True,
                    ),
                ),
            ],
            mesh=Mesh(node_count=1, process_per_node=1, gpu_per_process=0),
            yt_client=yt_client,
            docker_image=GENERIC_DOCKER_IMAGE,
            dry_run=True,
        )
    attached_files = run_info.operation_spec["tasks"]["task"]["file_paths"]
    attached_files = [
        attach for attach in attached_files if isinstance(attach, yt.ypath.FilePath)  # skip yt wrapper files
    ]
    attached_files = list(sorted(attached_files, key=str))
    file_suffixes = list(sorted(file_suffixes))
    for file_suffix, attached_file in zip(file_suffixes, attached_files):
        assert str(attached_file) == f"{yt_map_path}/{file_suffix}"
        assert attached_file.attributes == {
            "executable": False,
            "file_name": f"{destination}/{file_suffix}",
            "bypass_artifact_cache": True,
            "format": "dummy",
        }


def test_cypress_bind_map_node_symlinks(yt_instance: YtInstance, yt_path: str) -> None:
    yt_client = yt_instance.get_client()

    yt_map_path = f"{yt_path}/some_dir"
    yt_map_path_1 = f"{yt_map_path}/1"
    yt_map_path_2 = f"{yt_map_path}/2"

    yt_client.create("map_node", yt_map_path)
    yt_client.create("map_node", yt_map_path_1)
    yt_client.create("map_node", yt_map_path_2)
    yt_client.link(link_path=f"{yt_map_path_1}/link_to_2", target_path=yt_map_path_2)

    destination = "some_local_dir"

    file_suffixes = ["foo", "bar"]
    for base_folder in [yt_map_path_1, yt_map_path_2]:
        for file_suffix in file_suffixes:
            file_path = f"{base_folder}/{file_suffix}"
            yt_client.write_file(file_path, b"hello")

    def checker(_: Toolbox) -> None:
        pass

    run_info = run(
        checker,
        backend=GenericBackend(),
        yt_path=yt_path,
        binds_cypress=[
            BindCypress(
                source=yt_map_path_1,
                destination=destination,
            ),
        ],
        mesh=Mesh(node_count=1, process_per_node=1, gpu_per_process=0),
        yt_client=yt_client,
        docker_image=GENERIC_DOCKER_IMAGE,
        dry_run=True,
    )
    attached_files = run_info.operation_spec["tasks"]["task"]["file_paths"]
    attached_files = [
        str(attach) for attach in attached_files if isinstance(attach, yt.ypath.FilePath)  # skip yt wrapper files
    ]
    attached_files = list(sorted(attached_files))
    assert attached_files == [
        f"{yt_path}/some_dir/1/bar",
        f"{yt_path}/some_dir/1/foo",
        f"{yt_path}/some_dir/1/link_to_2/bar",
        f"{yt_path}/some_dir/1/link_to_2/foo",
    ]


def test_cypress_bind_from_run_config(yt_instance: YtInstance, yt_path: str, cypress_file: str) -> None:
    yt_client = yt_instance.get_client()

    run_config = {
        "mesh": {
            "node_count": 1,
            "process_per_node": 1,
            "gpu_per_process": 0,
        },
        "bind_cypress": [
            {
                "source": f"{cypress_file}",
                "destination": "bar",
            },
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
