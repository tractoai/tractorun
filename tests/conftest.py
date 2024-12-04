import os
from typing import Generator
import warnings

import pytest
import yt_yson_bindings

from tests.utils import (
    get_data_path,
    get_random_string,
)
from tests.yt_instances import (
    YtInstance,
    YtInstanceExternal,
    YtInstanceTestContainers,
)
from tractorun.private.yt_cluster import TractorunClusterConfig


@pytest.fixture(scope="session")
def yt_instance() -> Generator[YtInstance, None, None]:
    yt_mode = os.environ.get("YT_MODE", "testcontainers")
    if yt_mode == "testcontainers":
        with YtInstanceTestContainers() as yt_instance:
            yield yt_instance
    elif yt_mode == "external":
        proxy_url = os.environ["YT_PROXY"]
        yt_token = os.environ.get("YT_TOKEN")
        assert yt_token is not None
        yield YtInstanceExternal(proxy_url=proxy_url, token=yt_token)
    else:
        raise ValueError(f"Unknown yt_mode: {yt_mode}")


@pytest.fixture(scope="session")
def yt_base_dir(yt_instance: YtInstance) -> str:
    yt_client = yt_instance.get_client()

    path = f"//tmp/tractorun_tests/run_{get_random_string(4)}"
    yt_client.create("map_node", path, recursive=True)
    return path


@pytest.fixture(scope="function")
def yt_path(yt_instance: YtInstance, yt_base_dir: str) -> str:
    yt_client = yt_instance.get_client()
    path = f"{yt_base_dir}/{get_random_string(4)}"
    yt_client.create("map_node", path)
    return path


@pytest.fixture(scope="function")
def cluster_config() -> TractorunClusterConfig:
    return TractorunClusterConfig(
        cypress_link_template="https://yt.tracto.ai/yt/navigation?path={path}",
        job_stderr_link_template="https://yt.tracto.ai/api/yt/yt/api/v3/get_job_stderr?operation_id={operation_id}&job_id={job_id}&dump_error_into_response=true",
    )


@pytest.fixture
def cluster_config_path(cluster_config: TractorunClusterConfig, yt_instance: YtInstance, yt_path: str) -> str:
    config_path = f"{yt_path}/tractorun_config.yaml"
    yt_client = yt_instance.get_client()
    yt_client.set(config_path, cluster_config.to_dict())
    return config_path


@pytest.fixture(scope="session")
def mnist_ds_path(yt_base_dir: str, yt_instance: YtInstance) -> Generator[str, None, None]:
    table_path = f"{yt_base_dir}/{get_random_string(4)}"

    yt_client = yt_instance.get_client()
    yt_client.create(
        "table",
        table_path,
        attributes={  # TODO: type_v3?
            "schema": [
                {"name": "data", "type": "string"},
                {"name": "labels", "type": "string"},
            ]
        },
    )

    with open(get_data_path("mnist_small.yson"), "rb") as mnist_file:
        parsed_data = yt_yson_bindings.load(mnist_file, yson_type="list_fragment")
        yt_client.write_table(
            table=table_path,
            input_stream=parsed_data,
        )

    yield table_path

    yt_client.remove(table_path)


@pytest.fixture(scope="session")
def can_test_jax() -> bool:
    # we can't import jax inside orbax container
    # it's a session fixture because we can't import jax twice (I don't know why)
    try:
        import jax  # noqa: F401
    except RuntimeError as e:
        if "This version of jaxlib was built using AVX instructions" in str(e):
            warnings.warn(str(e))
            return False
    return True
