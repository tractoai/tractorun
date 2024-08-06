import json
import os
import random
import string
import subprocess
from typing import Any
import uuid

import attrs
from yt import wrapper as yt


DOCKER_IMAGE: str = "cr.ai.nebius.cloud/crnf2coti090683j5ssi/tractorun/torchesaurus_tests:2024-06-17-15-21-24"


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

    def is_operation_state_valid(self, yt_client: yt.YtClient, job_count: int) -> bool:
        operations = yt_client.list_operations(filter=self._operation_title)["operations"]
        assert len(operations) == 1

        operation_id = operations[0]["id"]
        operation_spec = yt_client.get_operation(operation_id)["spec"]
        return operation_spec["tasks"]["task"]["job_count"] == job_count


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class TractoCli:
    _command: list[str]
    _docker_image: str = attrs.field(default=DOCKER_IMAGE)
    _args: list[str]
    _task_spec: dict[str, Any] = attrs.field(default={})
    _operation_spec: dict[str, Any] = attrs.field(default={})

    def run(self) -> TractoCliRun:
        operation_title = f"test operation {uuid.uuid4()}"
        task_title = f"test operation's task {uuid.uuid4()}"

        command = [
            get_data_path("../../tractorun/cli/tractorun_runner.py"),
            "--docker-image",
            self._docker_image,
            "--yt-operation-spec",
            json.dumps(
                {
                    "title": operation_title,
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
            "--bind-lib",
            get_data_path("../../tractorun"),
            *self._args,
            *self._command,
        ]

        process = subprocess.Popen(command)
        process.wait()
        return TractoCliRun(
            process=process,
            operation_title=operation_title,
        )
