import json
import tempfile

import pytest
import yaml
import yt.wrapper as yt

from tests.utils import (
    DOCKER_IMAGE,
    TractoCli,
    get_data_path,
)
from tests.yt_instances import YtInstance
from tractorun.backend.generic import GenericBackend
from tractorun.mesh import Mesh
from tractorun.run import run
from tractorun.sidecar import (
    RestartPolicy,
    RestartVerdict,
    Sidecar,
    SidecarRun,
    SidecarRunner,
)
from tractorun.toolbox import Toolbox


def test_pickle(yt_instance: YtInstance, yt_path: str) -> None:
    yt_client = yt_instance.get_client()
    attr_key = "test_key"
    attr_value = "test_value"

    def checker(toolbox: Toolbox) -> None:
        import time

        client = toolbox.yt_client
        value = None
        attempts = 0
        while value is None:
            try:
                value = client.get(f"{yt_path}/@{attr_key}")
            except yt.errors.YtResolveError:
                pass
            time.sleep(5)
            if attempts > 5:
                raise Exception("Something wrong with sidecar")
            attempts += 1
        assert value == attr_value

    mesh = Mesh(node_count=1, process_per_node=1, gpu_per_process=0)
    run(
        checker,
        backend=GenericBackend(),
        yt_path=yt_path,
        sidecars=[
            Sidecar(
                command=["yt", "set", f"{yt_path}/@{attr_key}", attr_value],
                restart_policy=RestartPolicy.ON_FAILURE,
            )
        ],
        mesh=mesh,
        yt_client=yt_client,
        docker_image=DOCKER_IMAGE,
    )


def test_cli_args(yt_instance: YtInstance, yt_path: str) -> None:
    yt_client = yt_instance.get_client()

    attr_key = "test_key"
    attr_value = "test_value"

    tracto_cli = TractoCli(
        command=["python3", "/tractorun_tests/sidecar_script.py"],
        args=[
            "--yt-path",
            yt_path,
            "--user-config",
            json.dumps(
                {
                    "attr_key": attr_key,
                    "attr_value": attr_value,
                    "yt_path": yt_path,
                },
            ),
            "--sidecar",
            json.dumps(
                {
                    "command": ["yt", "set", f"{yt_path}/@{attr_key}", attr_value],
                    "restart_policy": RestartPolicy.ON_FAILURE,
                },
            ),
            "--bind",
            f"{get_data_path('../data/sidecar_script.py')}:/tractorun_tests",
        ],
    )
    op_run = tracto_cli.run()
    assert op_run.is_exitcode_valid()
    assert op_run.is_operation_state_valid(yt_client=yt_client, job_count=1)


def test_cli_config(yt_instance: YtInstance, yt_path: str) -> None:
    yt_client = yt_instance.get_client()

    attr_key = "test_key"
    attr_value = "test_value"

    run_config = {
        "mesh": {
            "node_count": 1,
            "process_per_mode": 1,
            "gpu_per_process": 0,
        },
        "sidecars": [
            {
                "command": ["yt", "set", f"{yt_path}/@{attr_key}", attr_value],
                "restart_policy": RestartPolicy.ON_FAILURE.value,
            },
        ],
    }
    with tempfile.NamedTemporaryFile(mode="w") as f:
        yaml.safe_dump(run_config, f)

        tracto_cli = TractoCli(
            command=["python3", "/tractorun_tests/sidecar_script.py"],
            args=[
                "--run-config-path",
                f.name,
                "--yt-path",
                yt_path,
                "--user-config",
                json.dumps(
                    {
                        "attr_key": attr_key,
                        "attr_value": attr_value,
                        "yt_path": yt_path,
                    },
                ),
                "--bind",
                f"{get_data_path('../data/sidecar_script.py')}:/tractorun_tests",
            ],
        )
        op_run = tracto_cli.run()
    assert op_run.is_exitcode_valid()
    assert op_run.is_operation_state_valid(yt_client=yt_client, job_count=1)


COMMAND_SUCCESS = ["python3", "-c", "import sys; sys.exit(0)"]
COMMAND_FAILED = ["python3", "-c", "import sys; sys.exit(1)"]


@pytest.mark.parametrize(
    "command,policy,verdict",
    [
        (
            COMMAND_SUCCESS,
            RestartPolicy.ALWAYS,
            RestartVerdict.restart,
        ),
        (
            COMMAND_SUCCESS,
            RestartPolicy.ON_FAILURE,
            RestartVerdict.skip,
        ),
        (
            COMMAND_SUCCESS,
            RestartPolicy.FAIL,
            RestartVerdict.fail,
        ),
        (
            COMMAND_SUCCESS,
            RestartPolicy.NEVER,
            RestartVerdict.skip,
        ),
        (
            COMMAND_FAILED,
            RestartPolicy.ALWAYS,
            RestartVerdict.restart,
        ),
        (
            COMMAND_FAILED,
            RestartPolicy.ON_FAILURE,
            RestartVerdict.restart,
        ),
        (
            COMMAND_FAILED,
            RestartPolicy.FAIL,
            RestartVerdict.fail,
        ),
        (
            COMMAND_FAILED,
            RestartPolicy.NEVER,
            RestartVerdict.skip,
        ),
    ],
)
def test_sidecar(command: list[str], policy: RestartPolicy, verdict: RestartVerdict) -> None:
    sidecar = Sidecar(command=command, restart_policy=policy)
    runner = SidecarRunner(command=sidecar.command, env={})
    process = runner.run()
    sidecar_run = SidecarRun(
        runner=runner,
        restart_policy=policy,
        process=process,
    )
    sidecar_run.wait()
    assert sidecar_run.need_restart() == verdict
