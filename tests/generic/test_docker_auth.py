from contextlib import nullcontext
from typing import (
    TYPE_CHECKING,
    ContextManager,
)
import uuid

import pytest
from yt import wrapper as yt

from tests.utils import (
    GENERIC_DOCKER_IMAGE,
    TractoCli,
    get_data_path,
    make_cli_args,
    make_run_config,
    run_config_file,
)
from tests.yt_instances import YtInstance
from tractorun.backend.generic import GenericBackend
from tractorun.cli.tractorun_runner import make_configuration
from tractorun.docker_auth import DockerAuthSecret
from tractorun.exception import DockerAuthDataError
from tractorun.mesh import Mesh
from tractorun.run import run


TEST_SECRET_USER_PASS = {
    "username": "user1",
    "password": "password1",
}
TEST_SECRET_AUTH = {
    "auth": "aksdl",
}


def test_configuration() -> None:
    _, _, config = make_configuration(
        make_cli_args(
            "--docker-auth-secret.cypress-path",
            "//tmp/some_secret_cli",
        )
    )
    assert config.docker_auth_secret == DockerAuthSecret(cypress_path="//tmp/some_secret_cli")

    run_config = make_run_config(
        {
            "docker_auth_secret": {
                "cypress_path": "//tmp/some_secret_config",
            },
        }
    )
    with run_config_file(run_config) as run_config_path:
        _, _, config = make_configuration(["--run-config-path", run_config_path])
    assert config.docker_auth_secret == DockerAuthSecret(cypress_path="//tmp/some_secret_config")

    with run_config_file(run_config) as run_config_path:
        _, _, config = make_configuration(
            ["--run-config-path", run_config_path, "--docker-auth-secret.cypress-path", "//tmp/some_secret_cli"]
        )
    assert config.docker_auth_secret == DockerAuthSecret(cypress_path="//tmp/some_secret_cli")


def create_yt_secret(yt_client: yt.YtClient, secret: dict, yt_path: str) -> str:
    secret_path = yt_path + "/secret"
    yt.create("document", secret_path, client=yt_client)
    yt_client.set(secret_path, secret)
    return secret_path


TEST_DATA = [
    # old format
    (
        "old",
        {
            "username": "user1",
            "password": "password1",
        },
        {
            "username": "user1",
            "password": "password1",
        },
    ),
    (
        "old",
        {
            "auth": "auth_1",
        },
        {
            "auth": "auth_1",
        },
    ),
    # new format
    (
        "new",
        {
            "secrets": {
                "username": {
                    "value": "user2",
                },
                "password": {
                    "value": "password2",
                },
            },
        },
        {
            "username": "user2",
            "password": "password2",
        },
    ),
    (
        "new",
        {
            "secrets": {
                "auth": {
                    "value": "auth_2",
                },
            },
        },
        {
            "auth": "auth_2",
        },
    ),
]


@pytest.mark.parametrize(
    "sec_format,secret,expected",
    TEST_DATA,
)
def test_spec_pickle(sec_format: str, secret: dict, expected: dict, yt_instance: YtInstance, yt_path: str) -> None:
    yt_client = yt_instance.get_client()

    yt_secret_path = create_yt_secret(yt_client, secret, yt_path)

    def checker() -> None:
        pass

    operation_title = f"test operation {uuid.uuid4()}"

    warn_checker = (
        pytest.warns(UserWarning, match="Please use new docker secret format.*")
        if sec_format == "old"
        else nullcontext()
    )
    if TYPE_CHECKING:
        assert isinstance(warn_checker, ContextManager)
    with warn_checker:
        run_info = run(
            checker,
            backend=GenericBackend(),
            yt_path=yt_path,
            mesh=Mesh(node_count=1, process_per_node=1, gpu_per_process=0),
            yt_client=yt_client,
            docker_image=GENERIC_DOCKER_IMAGE,
            title=operation_title,
            local=False,
            dry_run=True,
            docker_auth=DockerAuthSecret(cypress_path=yt_secret_path),
        )
    assert run_info.operation_spec["secure_vault"]["docker_auth"] == expected


@pytest.mark.parametrize(
    "sec_format,secret,expected",
    TEST_DATA,
)
def test_run_cli(
    sec_format: str, yt_instance: YtInstance, secret: dict, expected: dict, yt_path: str, mnist_ds_path: str
) -> None:
    yt_client = yt_instance.get_client()

    yt_secret_path = create_yt_secret(yt_client, secret, yt_path)

    tracto_cli = TractoCli(
        command=["python3", "/tractorun_tests/torch_run_script.py"],
        args=[
            "--mesh.node-count",
            "1",
            "--mesh.process-per-node",
            "1",
            "--mesh.gpu-per-process",
            "0",
            "--yt-path",
            yt_path,
            "--bind-local",
            f"{get_data_path('../data/torch_run_script.py')}:/tractorun_tests/torch_run_script.py",
            "--docker-auth-secret.cypress-path",
            yt_secret_path,
        ],
    )
    run_info = tracto_cli.dry_run()
    assert run_info["run_info"]["operation_spec"]["secure_vault"]["docker_auth"] == expected


@pytest.mark.parametrize(
    "secret",
    [
        {
            "username": "user1",
        },
        {
            "password": "user1",
        },
        {
            "auth_data": "sasdsa",
        },
        {
            "secrets": {},
        },
        {
            "secrets": {
                "auth": {"123": "123"},
            },
        },
    ],
)
def test_invalid_format(secret: dict, yt_instance: YtInstance, yt_path: str) -> None:
    yt_client = yt_instance.get_client()

    yt_secret_path = create_yt_secret(yt_client, secret, yt_path)

    def checker() -> None:
        pass

    operation_title = f"test operation {uuid.uuid4()}"

    with pytest.raises(DockerAuthDataError):
        _ = run(
            checker,
            backend=GenericBackend(),
            yt_path=yt_path,
            mesh=Mesh(node_count=1, process_per_node=1, gpu_per_process=0),
            yt_client=yt_client,
            docker_image=GENERIC_DOCKER_IMAGE,
            title=operation_title,
            local=False,
            dry_run=True,
            docker_auth=DockerAuthSecret(cypress_path=yt_secret_path),
        )
