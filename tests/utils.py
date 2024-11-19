import contextlib
import json
import os
import random
import string
import subprocess
import tempfile
import time
from typing import (
    Any,
    Generator,
)
import uuid

import attrs
import yaml
from yt import wrapper as yt


DOCKER_IMAGE: str = "cr.ai.nebius.cloud/crnf2coti090683j5ssi/tractorun/torchesaurus_tests:2024-10-18-22-05-42"
OPERATION_INFO_RETRIES = 5
OPERATION_INFO_TIMEOUT = 2


def get_data_path(filename: str) -> str:
    return os.path.join(os.path.dirname(__file__), "data", filename)


def get_random_string(length: int) -> str:
    return "".join(random.choice(string.ascii_letters) for _ in range(length))


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class TractoCliRun:
    _process: subprocess.Popen
    _operation_title: str

    def is_exitcode_valid(self, exit_code: int = 0) -> bool:
        return self._process.returncode == exit_code

    def get_operation_info(self, yt_client: yt.YtClient) -> dict:
        operations = []
        for _ in range(OPERATION_INFO_RETRIES):
            operations = yt_client.list_operations(filter=self._operation_title)["operations"]
            if operations:
                break
            time.sleep(OPERATION_INFO_TIMEOUT)

        assert len(operations) == 1

        operation_id = operations[0]["id"]
        operation_info = yt_client.get_operation(operation_id)
        return operation_info

    def is_operation_state_valid(self, yt_client: yt.YtClient, job_count: int) -> bool:
        operation_info = self.get_operation_info(yt_client)
        operation_spec = operation_info["spec"]
        return operation_spec["tasks"]["task"]["job_count"] == job_count

    @property
    def stdout(self) -> bytes:
        assert self._process.stdout is not None
        data = self._process.stdout.read()
        return data

    @property
    def stderr(self) -> bytes:
        assert self._process.stderr is not None
        data = self._process.stderr.read()
        return data


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class TractoCli:
    _command: list[str]
    _docker_image: str | None = attrs.field(default=DOCKER_IMAGE)
    _args: list[str]
    _task_spec: dict[str, Any] = attrs.field(default={})
    _operation_spec: dict[str, Any] = attrs.field(default={})

    def run(self) -> TractoCliRun:
        command, operation_title, task_title = self._prepare_command()

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process.wait()
        return TractoCliRun(
            process=process,
            operation_title=operation_title,
        )

    def dry_run(self) -> dict:
        command, operation_title, task_title = self._prepare_command()
        command.append("--dry-run")

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process.wait()
        run = TractoCliRun(
            process=process,
            operation_title=operation_title,
        )
        assert run.is_exitcode_valid()
        return json.loads(run.stdout)

    def _prepare_command(self) -> tuple[list[str], str, str]:
        operation_title = f"test operation {uuid.uuid4()}"
        task_title = f"test operation's task {uuid.uuid4()}"

        def _get_docker_image_arg() -> tuple:
            if self._docker_image is not None:
                return ("--docker-image", self._docker_image)
            return tuple()

        command = [
            get_data_path("../../tractorun/cli/tractorun_runner.py"),
            *_get_docker_image_arg(),
            "--title",
            operation_title,
            "--yt-operation-spec",
            json.dumps(
                {
                    **self._operation_spec,
                },
            ),
            "--yt-task-spec",
            json.dumps(
                {
                    "title": task_title,
                    **self._task_spec,
                },
            ),
            "--bind-local-lib",
            get_data_path("../../tractorun"),
            *self._args,
            *self._command,
        ]
        return command, operation_title, task_title


@contextlib.contextmanager
def run_config_file(config: dict[str, Any]) -> Generator[str, None, None]:
    with tempfile.NamedTemporaryFile(mode="w") as f:
        yaml.safe_dump(config, f)
        yield f.name


def make_cli_args(*cli_args: str) -> list:
    return ["--yt-path", "foo", "--docker-image", DOCKER_IMAGE, *cli_args, "command"]


def make_run_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "command": ["foo"],
        "yt_path": "foo",
        "docker_image": DOCKER_IMAGE,
        **config,
    }
