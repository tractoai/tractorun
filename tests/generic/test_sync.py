import pytest
from yt.common import YT_NULL_TRANSACTION_ID

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
from tractorun.cli.tractorun_runner import make_configuration
from tractorun.mesh import Mesh
from tractorun.run import run


def test_configuration() -> None:
    _, _, config = make_configuration(make_cli_args())
    assert config.no_wait is False

    run_config = make_run_config({})
    with run_config_file(run_config) as run_config_path:
        _, _, config = make_configuration(["--run-config-path", run_config_path])
    assert config.no_wait is False

    run_config = make_run_config({"no_wait": False})
    with run_config_file(run_config) as run_config_path:
        _, _, config = make_configuration(["--run-config-path", run_config_path, "--no-wait"])
    assert config.no_wait is True

    run_config = make_run_config({"no_wait": True})
    with run_config_file(run_config) as run_config_path:
        _, _, config = make_configuration(["--run-config-path", run_config_path])
    assert config.no_wait is True


@pytest.mark.parametrize(
    "no_wait,expected",
    [
        (True, True),
        (False, False),
    ],
)
def test_transaction_pickle(no_wait: bool, expected: bool, yt_instance: YtInstance, yt_path: str) -> None:
    yt_client = yt_instance.get_client()

    def dummy(toolbox: TractoCli) -> None:
        pass

    run_info = run(
        dummy,
        backend=GenericBackend(),
        yt_path=yt_path,
        mesh=Mesh(node_count=1, process_per_node=1, gpu_per_process=0),
        yt_client=yt_client,
        docker_image=GENERIC_DOCKER_IMAGE,
        no_wait=no_wait,
    )
    operation_info = yt_client.get_operation(run_info.operation_id)
    assert (operation_info["user_transaction_id"] == YT_NULL_TRANSACTION_ID) is expected


@pytest.mark.parametrize(
    "no_wait,expected",
    [
        (True, True),
        (False, False),
    ],
)
def test_transaction_script(no_wait: bool, expected: bool, yt_instance: YtInstance, yt_path: str) -> None:
    yt_client = yt_instance.get_client()
    args = [
        "--yt-path",
        yt_path,
        "--bind-local",
        f"{get_data_path('../data/dummy_script.py')}:/tractorun_tests/dummy_script.py",
    ]
    if no_wait:
        args.append("--no-wait")

    tracto_cli = TractoCli(
        command=["python3", "/tractorun_tests/dummy_script.py"],
        args=args,
    )
    op_run = tracto_cli.run()
    assert op_run.is_exitcode_valid()
    operation_info = op_run.get_operation_info(yt_client)
    assert (operation_info["user_transaction_id"] == YT_NULL_TRANSACTION_ID) is expected
