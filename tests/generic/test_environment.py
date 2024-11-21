import json
import os

import pytest
import yt.wrapper as yt

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
from tractorun.env import EnvVariable
from tractorun.mesh import Mesh
from tractorun.run import run


SECRET_ENV_VALUE = "secret"
NOT_SECRET_ENV_VALUE = "not_secret"


def test_configuration() -> None:
    _, _, config = make_configuration(make_cli_args())
    assert config.env == []

    env_json = json.dumps(
        {
            "name": "cli",
            "cypress_path": "raw_cypress_path",
            "value": "raw_value",
        }
    )

    _, _, config = make_configuration(
        make_cli_args(
            "--env",
            env_json,
        ),
    )
    assert config.env == [EnvVariable(name="cli", value="raw_value", cypress_path="raw_cypress_path")]

    run_config = make_run_config(
        {
            "env": [
                {
                    "name": "config",
                    "cypress_path": "raw_cypress_path",
                    "value": "raw_value",
                },
            ]
        },
    )
    with run_config_file(run_config) as run_config_path:
        _, _, config = make_configuration(["--run-config-path", run_config_path])
    assert config.env == [EnvVariable(name="config", value="raw_value", cypress_path="raw_cypress_path")]

    with run_config_file(run_config) as run_config_path:
        _, _, config = make_configuration(
            ["--run-config-path", run_config_path, "--env", env_json],
        )
    assert config.env == [EnvVariable(name="cli", value="raw_value", cypress_path="raw_cypress_path")]


@pytest.fixture
def yt_secret_path(yt_instance: YtInstance, yt_path: str) -> str:
    yt_client = yt_instance.get_client()

    secret_path = yt_path + "/secret"
    yt.create("document", secret_path, client=yt_client)
    yt.set(secret_path, SECRET_ENV_VALUE, client=yt_client)
    return secret_path


def test_environment(yt_secret_path: str, yt_instance: YtInstance, yt_path: str) -> None:
    yt_client = yt_instance.get_client()

    def env_checker(toolbox: TractoCli) -> None:
        assert os.environ["SECRET"] == SECRET_ENV_VALUE
        assert os.environ["NOT_SECRET"] == NOT_SECRET_ENV_VALUE

    run(
        env_checker,
        backend=GenericBackend(),
        yt_path=yt_path,
        mesh=Mesh(node_count=1, process_per_node=1, gpu_per_process=0),
        yt_client=yt_client,
        docker_image=GENERIC_DOCKER_IMAGE,
        env=[
            EnvVariable(name="SECRET", cypress_path=yt_secret_path),
            EnvVariable(name="NOT_SECRET", value="not_secret"),
        ],
    )


def test_run_script(yt_instance: YtInstance, yt_secret_path: str, yt_path: str) -> None:
    yt_client = yt_instance.get_client()

    tracto_cli = TractoCli(
        command=["python3", "/tractorun_tests/env_script.py"],
        args=[
            "--yt-path",
            yt_path,
            "--env",
            json.dumps(
                {
                    "name": "SECRET",
                    "cypress_path": yt_secret_path,
                },
            ),
            "--env",
            json.dumps(
                {
                    "name": "NOT_SECRET",
                    "value": NOT_SECRET_ENV_VALUE,
                },
            ),
            "--user-config",
            json.dumps({"secret_env_value": SECRET_ENV_VALUE, "not_secret_env_value": NOT_SECRET_ENV_VALUE}),
            "--bind-local",
            f"{get_data_path('../data/env_script.py')}:/tractorun_tests/env_script.py",
        ],
    )
    op_run = tracto_cli.run()
    assert op_run.is_exitcode_valid()
    assert op_run.is_operation_state_valid(yt_client=yt_client, job_count=1)


def test_run_script_with_config(yt_instance: YtInstance, yt_path: str, yt_secret_path: str) -> None:
    yt_client = yt_instance.get_client()

    run_config = {
        "mesh": {
            "node_count": 1,
            "process_per_node": 1,
            "gpu_per_process": 0,
        },
        "env": [
            {
                "name": "SECRET",
                "cypress_path": yt_secret_path,
            },
            {
                "name": "NOT_SECRET",
                "value": NOT_SECRET_ENV_VALUE,
            },
        ],
        "user_config": {
            "secret_env_value": SECRET_ENV_VALUE,
            "not_secret_env_value": NOT_SECRET_ENV_VALUE,
        },
    }
    with run_config_file(run_config) as run_config_path:
        tracto_cli = TractoCli(
            command=["python3", "/tractorun_tests/env_script.py"],
            args=[
                "--run-config-path",
                run_config_path,
                "--yt-path",
                yt_path,
                "--bind-local",
                f"{get_data_path('../data/env_script.py')}:/tractorun_tests/env_script.py",
            ],
        )
        op_run = tracto_cli.run()
    assert op_run.is_exitcode_valid()
    assert op_run.is_operation_state_valid(yt_client=yt_client, job_count=1)
