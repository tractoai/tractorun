import os

import yt.wrapper as yt

from tests.utils import (
    DOCKER_IMAGE,
    TractoCli,
)
from tests.yt_instances import YtInstance
from tractorun.backend.generic import GenericBackend
from tractorun.env import EnvVariable
from tractorun.mesh import Mesh
from tractorun.run import run


def test_environment(yt_instance: YtInstance, yt_path: str) -> None:
    yt_client = yt_instance.get_client()

    secret_path = yt_path + "/secret"

    yt.create("document", secret_path, client=yt_client)
    yt.set(secret_path, "secret", client=yt_client)

    def env_checker(toolbox: TractoCli) -> None:
        assert os.environ["SECRET"] == "secret"
        assert os.environ["NOT_SECRET"] == "not_secret"

    run(
        env_checker,
        backend=GenericBackend(),
        yt_path=yt_path,
        mesh=Mesh(node_count=1, process_per_node=1, gpu_per_process=0),
        yt_client=yt_client,
        docker_image=DOCKER_IMAGE,
        env=[
            EnvVariable(name="SECRET", cypress_path=secret_path),
            EnvVariable(name="NOT_SECRET", value="not_secret"),
        ],
    )
