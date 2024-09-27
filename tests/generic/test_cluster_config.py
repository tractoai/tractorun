import pytest

from tests.utils import run_config_file
from tests.yt_instances import YtInstance
from tractorun.cli.tractorun_runner import make_configuration, CLUSTER_CONFIG_PATH_DEFAULT
from tractorun.private.cluster_config import TractorunClusterConfig


CLUSTER_CONFIG = TractorunClusterConfig(
    cypress_link_template="https://yt.tracto.ai/yt/navigation?path={path}",
)


@pytest.fixture
def cluster_config_path(yt_instance: YtInstance, yt_path: str) -> str:
    config_path = f"{yt_path}/tractorun_config.yaml"
    yt_client = yt_instance.get_client()
    config = CLUSTER_CONFIG.to_dict()
    yt_client.set(config_path, config)
    return config_path


def test_load_config(cluster_config_path: str, yt_instance: YtInstance) -> None:
    config = TractorunClusterConfig.load_from_yt(yt_client=yt_instance.get_client(), path=cluster_config_path)
    assert config == CLUSTER_CONFIG


def test_configuration() -> None:
    _, _, config = make_configuration(
        ["--yt-path", "foo", "command"],
    )
    assert config.cluster_config_path == CLUSTER_CONFIG_PATH_DEFAULT

    _, _, config = make_configuration(
        ["--yt-path", "foo", "--cluster-config-path", "//cli_path", "command"]
    )
    assert config.cluster_config_path == "//cli_path"

    run_config = {
        "command": ["foo"],
        "yt_path": "foo",
        "cluster_config_path": "//config_path",
    }
    with run_config_file(run_config) as run_config_path:
        _, _, config = make_configuration(["--run-config-path", run_config_path])
    assert config.cluster_config_path == "//config_path"

    with run_config_file(run_config) as run_config_path:
        _, _, config = make_configuration(
            ["--run-config-path", run_config_path, "--cluster-config-path", "//cli_path"]
        )
    assert config.cluster_config_path == "//cli_path"
