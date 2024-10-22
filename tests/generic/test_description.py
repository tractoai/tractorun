import contextlib
from typing import (
    ContextManager,
    Generator,
)

import attrs
import pytest
import yt.wrapper as yt

from tests.utils import DOCKER_IMAGE
from tests.yt_instances import YtInstance
from tractorun.backend.generic import GenericBackend
from tractorun.description import (
    DescriptionManager,
    Link,
)
from tractorun.mesh import Mesh
from tractorun.private.constants import (
    TRACTORUN_DESCRIPTION_MANAGER_NAME,
    USER_DESCRIPTION_MANAGER_NAME,
)
from tractorun.private.yt_cluster import (
    TractorunClusterConfig,
    make_cypress_link,
)
from tractorun.run import run
from tractorun.toolbox import Toolbox


@contextlib.contextmanager
def _dummy_context_manager() -> Generator[None, None, None]:
    yield


@pytest.mark.parametrize(
    "config_exists",
    [False, True],
)
def test_description_empty_config(config_exists: bool, yt_path: str, yt_instance: YtInstance) -> None:
    # checking that the basic logic works without config or with an empty config
    yt_client = yt_instance.get_client()

    def checker(toolbox: Toolbox) -> None:
        pass

    config_path = f"{yt_path}/empty_config"

    if config_exists:
        yt_client.create("document", path=config_path)

    mesh = Mesh(node_count=1, process_per_node=1, gpu_per_process=0)
    ctx_manager = (
        pytest.warns(
            UserWarning,
            match=f"Cluster config {config_path} does not exist. Some functions are not available. Please specify config's path by tractorun params.",
        )
        if not config_exists
        else _dummy_context_manager()
    )
    assert isinstance(ctx_manager, ContextManager)
    with ctx_manager:
        run_info = run(
            checker,
            backend=GenericBackend(),
            yt_path=yt_path,
            mesh=mesh,
            yt_client=yt_client,
            docker_image=DOCKER_IMAGE,
            cluster_config_path=config_path,
        )
    operation = yt.Operation(id=run_info.operation_id, client=yt_client)
    description = operation.get_attributes()["runtime_parameters"]["annotations"]["description"]
    tractorun_description = description[TRACTORUN_DESCRIPTION_MANAGER_NAME]
    assert tractorun_description is not None


def test_set_tractorun_description(
    yt_instance: YtInstance, cluster_config: TractorunClusterConfig, cluster_config_path: str, yt_path: str
) -> None:
    assert cluster_config.cypress_link_template is not None
    yt_client = yt_instance.get_client()

    def checker(toolbox: Toolbox) -> None:
        pass

    mesh = Mesh(node_count=1, process_per_node=1, gpu_per_process=0)
    run_info = run(
        checker,
        backend=GenericBackend(),
        yt_path=yt_path,
        mesh=mesh,
        yt_client=yt_client,
        docker_image=DOCKER_IMAGE,
        cluster_config_path=cluster_config_path,
    )
    operation = yt.Operation(id=run_info.operation_id, client=yt_client)
    description = operation.get_attributes()["runtime_parameters"]["annotations"]["description"]
    tractorun_description = description[TRACTORUN_DESCRIPTION_MANAGER_NAME]
    assert str(tractorun_description["training_dir"]) == make_cypress_link(
        path=yt_path,
        cypress_link_template=cluster_config.cypress_link_template,
    )
    assert str(tractorun_description["logs"]) == make_cypress_link(
        path=f"{yt_path}/logs/0",
        cypress_link_template=cluster_config.cypress_link_template,
    )
    assert "primary" in tractorun_description
    assert "job_stderr" in tractorun_description["primary"]
    assert "address" in tractorun_description["primary"]
    assert "job_stderr" in tractorun_description["primary"]
    assert int(tractorun_description["incarnation"]) == 0
    assert tractorun_description["mesh"] == attrs.asdict(mesh)  # type: ignore


def test_set_user_description(yt_instance: YtInstance, cluster_config_path: str, yt_path: str) -> None:
    yt_client = yt_instance.get_client()

    def checker(toolbox: Toolbox) -> None:
        toolbox.description_manager.set(
            {
                "wandb": Link(value="https://fake.wandb.url/some/page"),
                "complex_structure": [
                    Link(value="https://another.com"),
                    [True, 1, 0.5, b"value"],
                    {"cats": "dogs"},
                ],
                "custom_info": "foo",
                "cypress_link": toolbox.description_manager.make_cypress_link("//some/path 1"),
            }
        )

    mesh = Mesh(node_count=1, process_per_node=1, gpu_per_process=0)
    run_info = run(
        checker,
        backend=GenericBackend(),
        yt_path=yt_path,
        mesh=mesh,
        yt_client=yt_client,
        docker_image=DOCKER_IMAGE,
        cluster_config_path=cluster_config_path,
    )
    operation = yt.Operation(id=run_info.operation_id, client=yt_client)
    description = operation.get_attributes()["runtime_parameters"]["annotations"]["description"]
    user_description = description[USER_DESCRIPTION_MANAGER_NAME]
    assert user_description == {
        "wandb": Link(value="https://fake.wandb.url/some/page").to_yson(),
        "complex_structure": [
            Link(value="https://another.com").to_yson(),
            [True, 1, 0.5, "value"],
            {"cats": "dogs"},
        ],
        "custom_info": "foo",
        "cypress_link": Link(value="https://yt.tracto.ai/yt/navigation?path=//some/path%201").to_yson(),
    }


def test_make_description() -> None:
    description = DescriptionManager._make_description(
        key=["1", "2"],
        description={
            "foo": 123,
            "bar": [1, 2, {"1": True}],
            "far": None,
            "link": Link(value=None),
            "bla": Link(value="//foo"),
            "links": [Link(value="link1"), Link(value="link2")],
        },
    )
    assert description == {
        "1": {
            "2": {
                "foo": 123,
                "bar": [1, 2, {"1": True}],
                "far": None,
                "link": Link(value=None).to_yson(),
                "bla": Link(value="//foo").to_yson(),
                "links": [Link(value="link1").to_yson(), Link(value="link2").to_yson()],
            },
        },
    }


def test_convert_yson() -> None:
    converted = DescriptionManager._convert_yson(
        {
            "foo": 123,
            "bar": [1, 2, {"1": True}],
            "far": None,
            "link": Link(value=None),
            "bla": Link(value="//foo"),
            "links": [Link(value="link1"), Link(value="link2")],
            "types": [True, 1, 0.5, b"value"],
        }
    )
    assert converted == {
        "foo": 123,
        "bar": [1, 2, {"1": True}],
        "far": None,
        "link": Link(value=None).to_yson(),
        "bla": Link(value="//foo").to_yson(),
        "links": [Link(value="link1").to_yson(), Link(value="link2").to_yson()],
        "types": [True, 1, 0.5, b"value"],
    }


def test_make_cypress_link(yt_instance: YtInstance) -> None:
    link = DescriptionManager(
        operation_id="123-123",
        cypress_link_template="https://fake.cluster?path={path}",
        yt_client=yt_instance.get_client(),
    ).make_cypress_link("//tmp/some/path")
    assert link == Link(value="https://fake.cluster?path=//tmp/some/path")
