import json

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
    Sidecar,
)
from tractorun.toolbox import Toolbox


def test_success_with_pickle(yt_instance: YtInstance, yt_path: str) -> None:
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


def test_success_with_cli_args(yt_instance: YtInstance, yt_path: str) -> None:
    yt_client = yt_instance.get_client()

    attr_key = "test_key"
    attr_value = "test_value"

    tracto_cli = TractoCli(
        command=["python3", "/tractorun_tests/sidecar_script.py"],
        args=[
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
        yt_path=yt_path,
    )
    op_run = tracto_cli.run()
    assert op_run.is_exitcode_valid()
    assert op_run.is_operation_state_valid(yt_client=yt_client, job_count=1)
