from tests.utils import (
    DOCKER_IMAGE,
    TractoCli,
)
from tests.yt_instances import YtInstance
from tractorun.backend.generic import GenericBackend
from tractorun.mesh import Mesh
from tractorun.run import run


def test_description(yt_instance: YtInstance, yt_path: str) -> None:
    yt_client = yt_instance.get_client()

    def checker(toolbox: TractoCli) -> None:
        pass

    operation = run(
        checker,
        backend=GenericBackend(),
        yt_path=yt_path,
        mesh=Mesh(node_count=1, process_per_node=1, gpu_per_process=0),
        yt_client=yt_client,
        docker_image=DOCKER_IMAGE,
    )
    assert operation.operation_attributes is not None
    description = operation.operation_attributes["runtime_parameters"]["annotations"]["description"][0]
    assert str(description["tractorun"]["training_dir"]) == yt_path
    assert "primary_address" in description["tractorun"]
    assert int(description["tractorun"]["incarnation"]) == 0
