import pathlib
import typing as t

from tests.utils import (
    DOCKER_IMAGE,
    get_data_path,
)
from tests.yt_instances import YtInstance
from tractorun.backend.generic import GenericBackend
from tractorun.bind import (
    BindCypress,
    BindLocal,
)
from tractorun.mesh import Mesh
from tractorun.run import run
from tractorun.toolbox import Toolbox


FILE_PATH = "/bind/file"
DIR_PATH = "/bind/dir"


def checker_dir(toolbox: Toolbox) -> None:
    assert pathlib.Path(DIR_PATH).is_dir()
    assert (pathlib.Path(DIR_PATH) / "some_file").is_file()


def get_file_checker(file_path: str = FILE_PATH) -> t.Callable[[Toolbox], None]:
    def checker_file(toolbox: Toolbox) -> None:
        assert pathlib.Path(file_path).is_file()

    return checker_file


def test_local_bind_file_pickle(yt_instance: YtInstance, yt_path: str) -> None:
    yt_client = yt_instance.get_client()

    run(
        get_file_checker(),
        backend=GenericBackend(),
        yt_path=yt_path,
        binds_local=[
            BindLocal(
                source=get_data_path("../data/binds/another_file"),
                destination=FILE_PATH,
            ),
        ],
        mesh=Mesh(node_count=1, process_per_node=1, gpu_per_process=0),
        yt_client=yt_client,
        docker_image=DOCKER_IMAGE,
    )


def test_local_bind_dir_pickle(yt_instance: YtInstance, yt_path: str) -> None:
    yt_client = yt_instance.get_client()

    run(
        checker_dir,
        backend=GenericBackend(),
        yt_path=yt_path,
        binds_local=[
            BindLocal(
                source=get_data_path("../data/binds/bind_dir"),
                destination=DIR_PATH,
            ),
        ],
        mesh=Mesh(node_count=1, process_per_node=1, gpu_per_process=0),
        yt_client=yt_client,
        docker_image=DOCKER_IMAGE,
    )


def test_cypress_bind_file(yt_instance: YtInstance, yt_path: str) -> None:
    yt_client = yt_instance.get_client()

    cypress_file_path = "//tmp/foo"
    destination_path = "bar"
    yt_client.write_file(cypress_file_path, b"hello")

    run(
        get_file_checker(destination_path),
        backend=GenericBackend(),
        yt_path=yt_path,
        binds_cypress=[
            BindCypress(
                source=cypress_file_path,
                destination=destination_path,
            ),
        ],
        mesh=Mesh(node_count=1, process_per_node=1, gpu_per_process=0),
        yt_client=yt_client,
        docker_image=DOCKER_IMAGE,
    )
