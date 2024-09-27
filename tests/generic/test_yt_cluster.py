from tests.utils import run_config_file
from tests.yt_instances import YtInstance
from tractorun.cli.tractorun_runner import (
    CLUSTER_CONFIG_PATH_DEFAULT,
    make_configuration,
)
from tractorun.private.yt_cluster import (
    TractorunClusterConfig,
    make_cypress_link,
)


def test_load_config(cluster_config_path: str, cluster_config: TractorunClusterConfig, yt_instance: YtInstance) -> None:
    config = TractorunClusterConfig.load_from_yt(yt_client=yt_instance.get_client(), path=cluster_config_path)
    assert config == cluster_config


def test_configuration() -> None:
    _, _, config = make_configuration(
        ["--yt-path", "foo", "command"],
    )
    assert config.cluster_config_path == CLUSTER_CONFIG_PATH_DEFAULT

    _, _, config = make_configuration(["--yt-path", "foo", "--cluster-config-path", "//cli_path", "command"])
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
        _, _, config = make_configuration(["--run-config-path", run_config_path, "--cluster-config-path", "//cli_path"])
    assert config.cluster_config_path == "//cli_path"


def test_make_cypress_link(
    cluster_config_path: str, cluster_config: TractorunClusterConfig, yt_instance: YtInstance
) -> None:
    config = TractorunClusterConfig.load_from_yt(yt_client=yt_instance.get_client(), path=cluster_config_path)
    assert (
        make_cypress_link(path="//some/path", cypress_link_template=config.cypress_link_template)
        == "https://yt.tracto.ai/yt/navigation?path=//some/path"
    )
