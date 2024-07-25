import yt.wrapper as yt

from tests.utils import (
    DOCKER_IMAGE,
    get_random_string,
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


def test_success_with_pickle(yt_instance: YtInstance, mnist_ds_path: str) -> None:
    yt_client = yt_instance.get_client()

    yt_training_dir = f"//tmp/{get_random_string(13)}"
    yt_client.create("map_node", yt_training_dir)

    def checker(toolbox: Toolbox) -> None:
        import time

        client = toolbox.yt_client
        value = None
        attempts = 0
        while value is None:
            try:
                value = client.get(f"{yt_training_dir}/@test_key")
            except yt.errors.YtResolveError:
                time.sleep(5)
                if attempts > 5:
                    raise Exception("Something wrong with sidecar")
                attempts += 1

    mesh = Mesh(node_count=1, process_per_node=1, gpu_per_process=0)
    run(
        checker,
        backend=GenericBackend(),
        yt_path=yt_training_dir,
        sidecars=[
            Sidecar(
                command=["yt", "set", f"{yt_training_dir}/@test_key", "test_value"],
                restart_policy=RestartPolicy.FAIL,
            )
        ],
        mesh=mesh,
        yt_client=yt_client,
        docker_image=DOCKER_IMAGE,
    )
