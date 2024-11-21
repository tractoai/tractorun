import json
import sys

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
from tractorun.mesh import Mesh
from tractorun.operation_log import OperationLogMode
from tractorun.private.process_manager import OutputType
from tractorun.run import run
from tractorun.sidecar import (
    RestartPolicy,
    Sidecar,
)
from tractorun.stderr_reader import StderrMode
from tractorun.toolbox import Toolbox


def test_configuration() -> None:
    _, _, config = make_configuration(make_cli_args())
    assert config.operation_log_mode == OperationLogMode.default

    _, _, config = make_configuration(make_cli_args("--operation-log-mode", OperationLogMode.realtime_yt_table.value))
    assert config.operation_log_mode == OperationLogMode.realtime_yt_table

    run_config = make_run_config({"operation_log_mode": OperationLogMode.realtime_yt_table.value})
    with run_config_file(run_config) as run_config_path:
        _, _, config = make_configuration(["--run-config-path", run_config_path])
    assert config.operation_log_mode == OperationLogMode.realtime_yt_table

    with run_config_file(run_config) as run_config_path:
        _, _, config = make_configuration(
            ["--run-config-path", run_config_path, "--operation-log-mode", OperationLogMode.default.value]
        )
    assert config.operation_log_mode == OperationLogMode.default


def validate_logs(yt_client: yt.YtClient, incarnation: int, yt_path: str) -> None:
    for index in range(4):
        messages: list[tuple[str, str]] = []
        for raw in yt_client.read_table(f"{yt_path}/logs/{incarnation}/workers/{index}"):
            assert raw["datetime"] is not None
            messages.append(
                (raw["message"], raw["fd"]),
            )
        assert (f"first stdout line {index} {incarnation}", OutputType.stdout) in messages
        assert (f"first stderr line {index} {incarnation}", OutputType.stderr) in messages
        assert (f"second stdout line {index} {incarnation}", OutputType.stdout) in messages
        assert (f"second stderr line {index} {incarnation}", OutputType.stderr) in messages

    for index in range(4):
        raws = [raw for raw in yt_client.read_table(f"{yt_path}/logs/{incarnation}/sidecars/{index}")]
        assert len(raws) == 1
        raw = raws[0]
        assert raw["datetime"] is not None
        output_type = OutputType.stdout if index % 2 == 0 else OutputType.stderr
        assert raw["fd"] == output_type
        assert raw["message"] == f"sidecar_writes_something_{incarnation}_{output_type}"


def test_pickle(yt_instance: YtInstance, yt_path: str) -> None:
    yt_client = yt_instance.get_client()

    def checker(toolbox: Toolbox) -> None:
        self_index = toolbox.coordinator.get_self_index()
        inc = toolbox.coordinator.get_incarnation_id()
        print(f"first stdout line {self_index} {inc}", file=sys.stdout)
        print(f"first stderr line {self_index} {inc}", file=sys.stderr)
        print(f"second stdout line {self_index} {inc}", file=sys.stdout)
        print(f"second stderr line {self_index} {inc}", file=sys.stderr)

    mesh = Mesh(node_count=2, process_per_node=2, gpu_per_process=0)
    for incarnation in range(2):
        sidecar_stdout_message = f"sidecar_writes_something_{incarnation}_{OutputType.stdout}"
        sidecar_stderr_message = f"sidecar_writes_something_{incarnation}_{OutputType.stderr}"
        run(
            checker,
            backend=GenericBackend(),
            yt_path=yt_path,
            operation_log_mode=OperationLogMode.realtime_yt_table,
            sidecars=[
                Sidecar(
                    command=["python3", "-c", f'import sys; print("{sidecar_stdout_message}", file=sys.stdout)'],
                    restart_policy=RestartPolicy.ON_FAILURE,
                ),
                Sidecar(
                    command=["python3", "-c", f'import sys; print("{sidecar_stderr_message}", file=sys.stderr)'],
                    restart_policy=RestartPolicy.ON_FAILURE,
                ),
            ],
            mesh=mesh,
            yt_client=yt_client,
            docker_image=GENERIC_DOCKER_IMAGE,
        )

        validate_logs(yt_client=yt_client, incarnation=incarnation, yt_path=yt_path)


def test_cli(yt_instance: YtInstance, yt_path: str) -> None:
    yt_client = yt_instance.get_client()

    for incarnation in range(2):
        sidecar_stdout_message = f"sidecar_writes_something_{incarnation}_{OutputType.stdout}"
        sidecar_stderr_message = f"sidecar_writes_something_{incarnation}_{OutputType.stderr}"
        tracto_cli = TractoCli(
            command=["python3", "/tractorun_tests/log_table_script.py"],
            args=[
                "--yt-path",
                yt_path,
                "--mesh.node-count",
                "2",
                "--mesh.process-per-node",
                "2",
                "--operation-log-mode",
                OperationLogMode.realtime_yt_table.value,
                "--sidecar",
                json.dumps(
                    {
                        "command": ["python3", "-c", f'import sys; print("{sidecar_stdout_message}", file=sys.stdout)'],
                        "restart_policy": RestartPolicy.ON_FAILURE,
                    },
                ),
                "--sidecar",
                json.dumps(
                    {
                        "command": ["python3", "-c", f'import sys; print("{sidecar_stderr_message}", file=sys.stderr)'],
                        "restart_policy": RestartPolicy.ON_FAILURE,
                    },
                ),
                "--bind-local",
                f"{get_data_path('../data/log_table_script.py')}:/tractorun_tests/log_table_script.py",
                "--proxy-stderr-mode",
                StderrMode.primary,
            ],
        )
        op_run = tracto_cli.run()
        assert op_run.is_exitcode_valid()
        assert op_run.is_operation_state_valid(yt_client=yt_client, job_count=2)

        validate_logs(yt_client=yt_client, incarnation=incarnation, yt_path=yt_path)
