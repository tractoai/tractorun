import contextlib
import inspect
import json
from pathlib import Path
import random
import string
import subprocess
import tempfile
import time
from types import ModuleType
from typing import (
    Any,
    Generator,
    Iterable,
)
import uuid

import attrs
import yaml
from yt import wrapper as yt


_CURRENT_DOCKER_TAG = "2025-09-30-15-26-34"
TRACTORCH_DOCKER_IMAGE: str = f"ghcr.io/tractoai/tractorun-tractorch-tests:{_CURRENT_DOCKER_TAG}"
TRACTORAX_DOCKER_IMAGE: str = f"ghcr.io/tractoai/tractorun-tractorax-tests:{_CURRENT_DOCKER_TAG}"
GENERIC_DOCKER_IMAGE: str = f"ghcr.io/tractoai/tractorun-generic-tests:{_CURRENT_DOCKER_TAG}"
EXAMPLES_DOCKER_IMAGE: str = f"ghcr.io/tractoai/tractorun-examples-runtime:{_CURRENT_DOCKER_TAG}"
TENSORPROXY_DOCKER_IMAGE: str = f"ghcr.io/tractoai/tractorun-tensorproxy-tests:{_CURRENT_DOCKER_TAG}"

OPERATION_INFO_RETRIES = 5
OPERATION_INFO_TIMEOUT = 2


def get_data_path(filename: str | Path) -> Path:
    return (Path(__file__).parent / "data" / filename).resolve()


def get_random_string(length: int) -> str:
    return str(int(time.time())) + "_" + "".join(random.choice(string.ascii_letters) for _ in range(length))


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class TractoCliRun:
    _process: subprocess.Popen
    _operation_title: str
    _stdout: bytes
    _stderr: bytes

    def is_exitcode_valid(self, exit_code: int = 0) -> bool:
        return self._process.returncode == exit_code

    def validate_exit_code(self, exit_code: int = 0) -> None:
        assert self.is_exitcode_valid(
            exit_code
        ), f"Expected exit code {exit_code}, but got {self._process.returncode}\nstdout:\n{self.stdout.decode()}\nstderr:\n{self.stderr.decode()}"

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
        return self._stdout

    @property
    def stderr(self) -> bytes:
        return self._stderr


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class TractoCli:
    _command: list[str | Path] = attrs.field(default=[])
    _docker_image: str | None = attrs.field(default=GENERIC_DOCKER_IMAGE)
    _args: list[str]
    _task_spec: dict[str, Any] = attrs.field(default={})
    _operation_spec: dict[str, Any] = attrs.field(default={})

    def run(self) -> TractoCliRun:
        command, operation_title, task_title = self._prepare_command()

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        print(f"stdout: {stdout.decode()}")
        print(f"stderr: {stderr.decode()}")
        return TractoCliRun(
            process=process,
            operation_title=operation_title,
            stdout=stdout,
            stderr=stderr,
        )

    def dry_run(self) -> dict:
        command, operation_title, task_title = self._prepare_command()
        command.append("--dry-run")

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        print(f"stdout: {stdout.decode()}")
        print(f"stderr: {stderr.decode()}")
        run = TractoCliRun(
            process=process,
            operation_title=operation_title,
            stdout=stdout,
            stderr=stderr,
        )
        assert run.is_exitcode_valid()
        return json.loads(run.stdout)

    def _prepare_command(self) -> tuple[list[str | Path], str, str]:
        operation_title = f"test operation {uuid.uuid4()}"
        task_title = f"test operation's task {uuid.uuid4()}"

        def _get_docker_image_arg() -> tuple:
            if self._docker_image is not None:
                return ("--docker-image", self._docker_image)
            return tuple()

        command: list[str | Path] = [
            str(get_data_path("../../tractorun/cli/tractorun_runner.py")),
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
            str(get_data_path("../../tractorun")),
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
    return ["--yt-path", "foo", "--docker-image", GENERIC_DOCKER_IMAGE, *cli_args, "command"]


def make_run_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "command": ["foo"],
        "yt_path": "foo",
        "docker_image": GENERIC_DOCKER_IMAGE,
        **config,
    }


def _is_private_module(module: ModuleType | None) -> bool:
    if module is None:
        return False
    private_prefixes = [
        "tractorun.private",
        "tractorun.tests",
        "tractorun.cli",
    ]
    for prefix in private_prefixes:
        if module.__name__.startswith(prefix):
            return True
    return False


def check_no_private_objects_in_public_namespace(modules: Iterable[ModuleType]) -> bool:
    # this test can't check variables with primitive types like strings, int, float and so on
    invalid_public_modules: list[tuple[str, object, ModuleType]] = []
    for module in modules:
        if _is_private_module(module):
            continue
        for name, obj in inspect.getmembers(module):
            if name.startswith("_"):
                continue
            if not isinstance(obj, object):
                continue
            if inspect.ismodule(obj):
                continue
            obj_module = inspect.getmodule(obj)
            if obj_module is None:
                continue
            if not _is_private_module(obj_module):
                continue
            invalid_public_modules.append((name, obj, module))
    return invalid_public_modules == []
