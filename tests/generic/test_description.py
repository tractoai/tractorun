import attrs

from tests.utils import (
    DOCKER_IMAGE,
    TractoCli,
)
from tests.yt_instances import YtInstance
from tractorun.backend.generic import GenericBackend
from tractorun.mesh import Mesh
from tractorun.private.constants import TRACTORUN_DESCRIPTION_MANAGER_NAME
from tractorun.private.description import (
    DescriptionManager,
    Link,
)
from tractorun.private.yt_cluster import (
    TractorunClusterConfig,
    make_cypress_link,
)
from tractorun.run import run


def test_set_tractorun_description(
    yt_instance: YtInstance, cluster_config: TractorunClusterConfig, cluster_config_path: str, yt_path: str
) -> None:
    assert cluster_config.cypress_link_template is not None
    yt_client = yt_instance.get_client()

    def checker(toolbox: TractoCli) -> None:
        pass

    mesh = Mesh(node_count=1, process_per_node=1, gpu_per_process=0)
    operation = run(
        checker,
        backend=GenericBackend(),
        yt_path=yt_path,
        mesh=mesh,
        yt_client=yt_client,
        docker_image=DOCKER_IMAGE,
        cluster_config_path=cluster_config_path,
    )
    assert operation.operation_attributes is not None
    description = operation.operation_attributes["runtime_parameters"]["annotations"]["description"]
    tractorun_description = description[TRACTORUN_DESCRIPTION_MANAGER_NAME]
    assert str(tractorun_description["training_dir"]) == make_cypress_link(
        path=yt_path,
        cypress_link_template=cluster_config.cypress_link_template,
    )
    assert "primary" in tractorun_description
    assert "job_stderr" in tractorun_description["primary"]
    assert "address" in tractorun_description["primary"]
    assert "job_stderr" in tractorun_description["primary"]
    assert int(tractorun_description["incarnation"]) == 0
    assert tractorun_description["mesh"] == attrs.asdict(mesh)  # type: ignore


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
        }
    )
    assert converted == {
        "foo": 123,
        "bar": [1, 2, {"1": True}],
        "far": None,
        "link": Link(value=None).to_yson(),
        "bla": Link(value="//foo").to_yson(),
        "links": [Link(value="link1").to_yson(), Link(value="link2").to_yson()],
    }
