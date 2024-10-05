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
        inc = toolbox.coordinator.get_incarnation_id()
        print(f"first stdout line {self_index} {inc}", file=sys.stdout)
        print(f"first stderr line {self_index} {inc}", file=sys.stderr)
        print(f"second stdout line {self_index} {inc}", file=sys.stdout)
        print(f"second stderr line {self_index} {inc}", file=sys.stderr)

    mesh = Mesh(node_count=2, process_per_node=2, gpu_per_process=0)
    for incarnation in range(2):
        sidecar_stdout_message = f"sidecar_writes_something_{incarnation}_{OutputType.stdout}"
        sidecar_stderr_message = f"sidecar_writes_something_{incarnation}_{OutputType.stderr}"
        run(
            checker,
            backend=GenericBackend(),
            yt_path=yt_path,
            sidecars=[
                Sidecar(
                    command=["python3", "-c", f'import sys; print("{sidecar_stdout_message}", file=sys.stdout)'],
                    restart_policy=RestartPolicy.ON_FAILURE,
                ),
                Sidecar(
                    command=["python3", "-c", f'import sys; print("{sidecar_stderr_message}", file=sys.stderr)'],
                    restart_policy=RestartPolicy.ON_FAILURE,
                ),
            ],
            mesh=mesh,
            yt_client=yt_client,
            docker_image=DOCKER_IMAGE,
        )

        for index in range(4):
            messages: list[tuple[str, str]] = []
            for raw in yt_client.read_table(f"{yt_path}/logs/{incarnation}/workers/{index}"):
                assert raw["datetime"] is not None
                messages.append(
                    (raw["message"], raw["fd"]),
                )
            assert (f"first stdout line {index} {incarnation}", OutputType.stdout) in messages
            assert (f"first stderr line {index} {incarnation}", OutputType.stderr) in messages
            assert (f"second stdout line {index} {incarnation}", OutputType.stdout) in messages
            assert (f"second stderr line {index} {incarnation}", OutputType.stderr) in messages

        for index in range(4):
            raws = [raw for raw in yt_client.read_table(f"{yt_path}/logs/{incarnation}/sidecars/{index}")]
            assert len(raws) == 1
            raw = raws[0]
            assert raw["datetime"] is not None
            output_type = OutputType.stdout if index % 2 == 0 else OutputType.stderr
            assert raw["fd"] == output_type
            assert raw["message"] == f"sidecar_writes_something_{incarnation}_{output_type}"
