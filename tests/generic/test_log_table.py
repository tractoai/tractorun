import sys

from tests.utils import DOCKER_IMAGE
from tests.yt_instances import YtInstance
from tractorun.backend.generic import GenericBackend
from tractorun.mesh import Mesh
from tractorun.private.process_manager import OutputType
from tractorun.run import run
from tractorun.sidecar import (
    RestartPolicy,
    Sidecar,
)
from tractorun.toolbox import Toolbox


def test_pickle(yt_instance: YtInstance, yt_path: str) -> None:
    yt_client = yt_instance.get_client()

    def checker(toolbox: Toolbox) -> None:
        self_index = toolbox.coordinator.get_self_index()
        print(f"first stdout line {self_index}", file=sys.stdout)
        print(f"first stderr line {self_index}", file=sys.stderr)
        print(f"second stdout line {self_index}", file=sys.stdout)
        print(f"second stderr line {self_index}", file=sys.stderr)

    mesh = Mesh(node_count=2, process_per_node=2, gpu_per_process=0)
    run(
        checker,
        backend=GenericBackend(),
        yt_path=yt_path,
        sidecars=[
            Sidecar(
                command=["echo", "sidecar_writes_something_stderr", "1>&2"],
                restart_policy=RestartPolicy.ON_FAILURE,
            ),
            Sidecar(
                command=["echo", "sidecar_writes_something_stdout"],
                restart_policy=RestartPolicy.ON_FAILURE,
            ),
        ],
        mesh=mesh,
        yt_client=yt_client,
        docker_image=DOCKER_IMAGE,
    )

    for index in range(4):
        messages: list[tuple[str, str]] = []
        for raw in yt_client.read_table(f"{yt_path}/logs/0/workers/{index}"):
            assert raw["datetime"] is not None
            messages.append(
                (raw["message"], raw["fd"]),
            )
        assert (f"first stdout line {index}", OutputType.stdout) in messages
        assert (f"first stderr line {index}", OutputType.stderr) in messages
        assert (f"second stdout line {index}", OutputType.stdout) in messages
        assert (f"second stderr line {index}", OutputType.stderr) in messages
