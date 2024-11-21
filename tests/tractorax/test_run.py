import json
import sys
from typing import Callable

import pytest

from tests.utils import (
    TRACTORAX_DOCKER_IMAGE,
    TractoCli,
    get_data_path,
)
from tests.yt_instances import YtInstance
from tractorun.mesh import Mesh
from tractorun.run import run
from tractorun.toolbox import Toolbox


def _get_simple_train(mnist_ds_path: str) -> Callable:
    def _simple_train(toolbox: Toolbox) -> None:
        pass

    return _simple_train


def test_run_pickle(can_test_jax: bool, yt_instance: YtInstance, yt_path: str, mnist_ds_path: str) -> None:
    if not can_test_jax:
        pytest.skip("jax can't be run on this platform")

    from jax import (
        grad,
        jit,
    )

    from tractorun.backend.tractorax import Tractorax

    def checker(_: Toolbox) -> None:
        @jit
        def f(x: int) -> int:
            return x**2 + 3 * x + 1

        grad_f = grad(f)
        print(grad_f(1.0), file=sys.stderr)

    yt_client = yt_instance.get_client()

    train_func = _get_simple_train(mnist_ds_path)

    mesh = Mesh(node_count=1, process_per_node=1, gpu_per_process=0)
    # The operation did not fail => success!
    run(
        train_func,
        backend=Tractorax(),
        yt_path=yt_path,
        mesh=mesh,
        yt_client=yt_client,
        docker_image=TRACTORAX_DOCKER_IMAGE,
    )


def test_run_script(yt_instance: YtInstance, yt_path: str, mnist_ds_path: str) -> None:
    yt_client = yt_instance.get_client()

    tracto_cli = TractoCli(
        command=["python3", "/tractorun_tests/jax_run_script.py"],
        docker_image=TRACTORAX_DOCKER_IMAGE,
        args=[
            "--mesh.node-count",
            "1",
            "--mesh.process-per-node",
            "1",
            "--mesh.gpu-per-process",
            "0",
            "--yt-path",
            yt_path,
            "--user-config",
            json.dumps({"MNIST_DS_PATH": mnist_ds_path}),
            "--bind-local",
            f"{get_data_path('../data/jax_run_script.py')}:/tractorun_tests/jax_run_script.py",
        ],
    )
    op_run = tracto_cli.run()
    assert op_run.is_exitcode_valid()
    assert op_run.is_operation_state_valid(yt_client=yt_client, job_count=1)
