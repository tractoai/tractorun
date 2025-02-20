import json
import os
import uuid

from _pytest.monkeypatch import MonkeyPatch
import pytest

from tests.utils import (
    GENERIC_DOCKER_IMAGE,
    TractoCli,
    get_data_path,
)
from tests.yt_instances import YtInstance
from tractorun.backend.generic import GenericBackend
from tractorun.bind import BindLocal
from tractorun.cli.tractorun_runner import CliRunInfo
from tractorun.env import EnvVariable
from tractorun.exception import TractorunConfigurationError
from tractorun.mesh import Mesh
from tractorun.private.helpers import AttrSerializer
from tractorun.run import (
    run,
    run_script,
)
from tractorun.toolbox import Toolbox


def test_important_spec_options(yt_path: str) -> None:
    def checker() -> None:
        pass

    run_info = run(
        checker,
        yt_path=yt_path,
        docker_image=GENERIC_DOCKER_IMAGE,
        mesh=Mesh(node_count=1, process_per_node=1, gpu_per_process=0),
        backend=GenericBackend(),
        dry_run=True,
    )
    assert run_info.operation_spec["annotations"]["is_tractorun"] is True
    assert run_info.operation_spec["fail_on_job_restart"] is True
    assert run_info.operation_spec["is_gang"] is True


def test_prepare_dataset(yt_instance: YtInstance, mnist_ds_path: str) -> None:
    yt_client = yt_instance.get_client()
    row_count = yt_client.get_attribute(mnist_ds_path, "row_count")
    assert row_count == 100
    schema = yt_client.get_attribute(mnist_ds_path, "schema")
    assert len(schema) == 2


def test_run(yt_instance: YtInstance, yt_path: str, mnist_ds_path: str) -> None:
    def checker(toolbox: Toolbox) -> None:
        pass

    yt_client = yt_instance.get_client()

    operation_title = f"test operation {uuid.uuid4()}"
    task_title = f"test operation's task {uuid.uuid4()}"

    mesh = Mesh(node_count=1, process_per_node=1, gpu_per_process=0)

    run(
        checker,
        backend=GenericBackend(),
        yt_path=yt_path,
        mesh=mesh,
        yt_client=yt_client,
        docker_image=GENERIC_DOCKER_IMAGE,
        title=operation_title,
        yt_task_spec={"title": task_title},
    )

    operations = yt_client.list_operations(filter=operation_title)["operations"]
    assert len(operations) == 1

    operation_id = operations[0]["id"]

    operation_spec = yt_client.get_operation(operation_id)["spec"]
    assert operation_spec["title"] == operation_title
    assert operation_spec["tasks"]["task"]["title"] == task_title


def test_run_script(yt_instance: YtInstance, yt_path: str, mnist_ds_path: str) -> None:
    yt_client = yt_instance.get_client()

    tracto_cli = TractoCli(
        command=["python3", "/tractorun_tests/dummy_script.py"],
        args=[
            "--mesh.node-count",
            "1",
            "--mesh.process-per-node",
            "1",
            "--mesh.gpu-per-process",
            "0",
            "--yt-path",
            yt_path,
            "--bind-local",
            f"{get_data_path('../data/dummy_script.py')}:/tractorun_tests/dummy_script.py",
        ],
    )
    op_run = tracto_cli.run()
    assert op_run.is_exitcode_valid()
    assert op_run.is_operation_state_valid(yt_client=yt_client, job_count=1)


def test_run_script_with_config(yt_instance: YtInstance, yt_path: str, mnist_ds_path: str) -> None:
    yt_client = yt_instance.get_client()

    tracto_cli = TractoCli(
        command=["python3", "/tractorun_tests/dummy_script.py"],
        args=[
            "--run-config-path",
            str(get_data_path("../data/run_config.yaml")),
            "--yt-path",
            yt_path,
            "--user-config",
            json.dumps({"MNIST_DS_PATH": mnist_ds_path}),
            "--bind-local",
            f"{get_data_path('../data/dummy_script.py')}:/tractorun_tests/dummy_script.py",
        ],
    )
    op_run = tracto_cli.run()
    assert op_run.is_exitcode_valid()
    assert op_run.is_operation_state_valid(yt_client=yt_client, job_count=2)


@pytest.mark.parametrize(
    "env,expected",
    [
        ({"YT_BASE_LAYER": "custom_image_1"}, "custom_image_1"),
        ({"YT_JOB_DOCKER_IMAGE": "custom_image_2"}, "custom_image_2"),
    ],
)
def test_docker_image_script(yt_path: str, env: dict[str, str], expected: str, monkeypatch: MonkeyPatch) -> None:
    tracto_cli = TractoCli(
        command=["python3", "/tractorun_tests/dummy_script.py"],
        args=[
            "--mesh.node-count",
            "1",
            "--mesh.process-per-node",
            "1",
            "--mesh.gpu-per-process",
            "0",
            "--yt-path",
            yt_path,
            "--bind-local",
            f"{get_data_path('../data/dummy_script.py')}:/tractorun_tests/dummy_script.py",
        ],
        docker_image=None,
    )
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    run_info = tracto_cli.dry_run()
    assert run_info["configuration"]["effective_config"]["docker_image"] == expected
    assert run_info["run_info"]["operation_spec"]["tasks"]["task"]["docker_image"] == expected


@pytest.mark.parametrize(
    "env,expected",
    [
        ({"YT_BASE_LAYER": "custom_image_1"}, "custom_image_1"),
        ({"YT_JOB_DOCKER_IMAGE": "custom_image_2"}, "custom_image_2"),
    ],
)
def test_docker_image_pickle(yt_path: str, env: dict[str, str], expected: str, monkeypatch: MonkeyPatch) -> None:
    def checker() -> None:
        pass

    for key, value in env.items():
        monkeypatch.setenv(key, value)
    run_info = run(
        checker,
        yt_path=yt_path,
        mesh=Mesh(node_count=1, process_per_node=1, gpu_per_process=0),
        backend=GenericBackend(),
        dry_run=True,
    )
    assert run_info.operation_spec["tasks"]["task"]["docker_image"] == expected


def test_without_docker_image_script(yt_path: str) -> None:
    tracto_cli = TractoCli(
        command=["python3", "/tractorun_tests/dummy_script.py"],
        args=[
            "--mesh.node-count",
            "1",
            "--mesh.process-per-node",
            "1",
            "--mesh.gpu-per-process",
            "0",
            "--yt-path",
            yt_path,
            "--bind-local",
            f"{get_data_path('../data/dummy_script.py')}:/tractorun_tests/dummy_script.py",
        ],
        docker_image=None,
    )
    with pytest.raises(AssertionError):
        tracto_cli.dry_run()


def test_without_docker_image_pickle(yt_path: str) -> None:
    def checker() -> None:
        pass

    with pytest.raises(TractorunConfigurationError):
        run(
            checker,
            yt_path=yt_path,
            mesh=Mesh(node_count=1, process_per_node=1, gpu_per_process=0),
            backend=GenericBackend(),
            dry_run=True,
        )


def test_run_cli_command_from_python(yt_path: str) -> None:
    run_script(
        ["python3", "/tractorun_tests/dummy_script.py"],
        yt_path=yt_path,
        binds_local=[
            BindLocal(
                source=str(get_data_path("../data/dummy_script.py")), destination="/tractorun_tests/dummy_script.py"
            ),
        ],
        binds_local_lib=[
            str(get_data_path("../../tractorun")),
        ],
        docker_image=GENERIC_DOCKER_IMAGE,
        mesh=Mesh(node_count=1, process_per_node=1, gpu_per_process=0),
    )


def test_script_debug_info(yt_instance: YtInstance, yt_path: str, mnist_ds_path: str) -> None:
    tracto_cli = TractoCli(
        command=["python3", "/tractorun_tests/dummy_script.py"],
        args=[
            "--mesh.node-count",
            "1",
            "--mesh.process-per-node",
            "1",
            "--mesh.gpu-per-process",
            "0",
            "--yt-path",
            yt_path,
            "--bind-local",
            f"{get_data_path('../data/dummy_script.py')}:/tractorun_tests/dummy_script.py",
            "--no-wait",
        ],
    )
    op_run = tracto_cli.run()
    assert op_run.is_exitcode_valid()
    serializer = AttrSerializer(CliRunInfo)
    run_info = serializer.deserialize(op_run.stdout.decode("utf-8"))
    assert run_info.run_info is not None
    assert run_info.run_info.operation_id is not None


def test_change_working_dirs(yt_instance: YtInstance, yt_path: str, mnist_ds_path: str) -> None:
    def checker(toolbox: Toolbox) -> None:
        assert os.environ["HOME"] == "/tmp"
        assert os.environ["PWD"] == "/tmp"
        assert os.environ["TMPDIR"] == "/tmp"

    yt_client = yt_instance.get_client()

    operation_title = f"test operation {uuid.uuid4()}"
    task_title = f"test operation's task {uuid.uuid4()}"

    mesh = Mesh(node_count=1, process_per_node=1, gpu_per_process=0)

    run(
        checker,
        backend=GenericBackend(),
        yt_path=yt_path,
        mesh=mesh,
        yt_client=yt_client,
        env=[
            EnvVariable(
                name="HOME",
                value="/tmp",
            ),
            EnvVariable(
                name="PWD",
                value="/tmp",
            ),
            EnvVariable(
                name="TMPDIR",
                value="/tmp",
            ),
        ],
        docker_image=GENERIC_DOCKER_IMAGE,
        title=operation_title,
        yt_task_spec={"title": task_title},
    )


def test_run_defaults_only_script(yt_instance: YtInstance, yt_path: str, mnist_ds_path: str) -> None:
    yt_client = yt_instance.get_client()

    tracto_cli = TractoCli(
        command=["python3", "/tractorun_tests/dummy_script.py"],
        # ok, bind and docker image is necessary for test
        docker_image=GENERIC_DOCKER_IMAGE,
        args=[
            "--bind-local",
            f"{get_data_path('../data/dummy_script.py')}:/tractorun_tests/dummy_script.py",
        ],
    )
    op_run = tracto_cli.run()
    assert op_run.is_exitcode_valid()
    assert op_run.is_operation_state_valid(yt_client=yt_client, job_count=1)


def test_run_defaults_only_pickling(yt_instance: YtInstance, yt_path: str, mnist_ds_path: str) -> None:
    def train_func(toolbox: Toolbox) -> None:
        pass

    # ok, docker image is necessary for test
    run(
        train_func,
        backend=GenericBackend(),
        docker_image=GENERIC_DOCKER_IMAGE,
    )
