import json
import sys
import threading

from _pytest.capture import CaptureFixture
import attrs
import pytest
from yt.wrapper import YtOperationFailedError

from tests.utils import (
    DOCKER_IMAGE,
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
from tractorun.private.stderr_reader import (
    STDERR_READER_THREAD_NAME,
    YtStderrReader,
)
from tractorun.run import run
from tractorun.stderr_reader import StderrMode
from tractorun.toolbox import Toolbox


TEST_STRINGS = ["hello", "my dear", "friend"]


def test_configuration() -> None:
    _, _, config = make_configuration(make_cli_args())
    assert config.proxy_stderr_mode == StderrMode.disabled

    _, _, config = make_configuration(make_cli_args("--proxy-stderr-mode", StderrMode.primary.value))
    assert config.proxy_stderr_mode == StderrMode.primary

    run_config = make_run_config({"proxy_stderr_mode": StderrMode.primary.value})
    with run_config_file(run_config) as run_config_path:
        _, _, config = make_configuration(["--run-config-path", run_config_path])
    assert config.proxy_stderr_mode == StderrMode.primary

    with run_config_file(run_config) as run_config_path:
        _, _, config = make_configuration(
            ["--run-config-path", run_config_path, "--proxy-stderr-mode", StderrMode.disabled],
        )
    assert config.proxy_stderr_mode == StderrMode.disabled


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class MockStderrProvider:
    _lines: list[bytes]
    _index: int = attrs.field(default=0)

    def read(self) -> bytes | None:
        while self._index < len(self._lines):
            line = self._lines[self._index]
            self._index += 1
            return line
        return None


@pytest.mark.parametrize(
    "lines,expected",
    [
        (
            [b"a", b"ab"],
            [b"a", b"b"],
        ),
        (
            [b"abcd", b"abcde"],
            [b"abcd", b"e"],
        ),
        (
            [b"abcd", b"abcdef"],
            [b"abcd", b"ef"],
        ),
        (
            [b"", b"a"],
            [b"", b"a"],
        ),
        (
            [b"a", b"b"],
            [b"a", b"b"],
        ),
        (
            [b"ab", b"ab"],
            [b"ab", b""],
        ),
        (
            [b"a", b"abcde", b"defg", b"abc"],
            [b"a", b"bcde", b"fg", b"abc"],
        ),
    ],
)
def test_stderr_reader(lines: list[bytes], expected: list[bytes]) -> None:
    mock = MockStderrProvider(lines=lines)
    reader = YtStderrReader(stderr_getter=mock.read, stop_on_none=True)
    result = list(reader.get_output())
    assert result == expected


@pytest.mark.parametrize(
    "mode",
    [StderrMode.primary, StderrMode.disabled],
)
def test_operation_pickling(
    mode: StderrMode, yt_path: str, yt_instance: YtInstance, capsys: CaptureFixture[str]
) -> None:
    def checker(toolbox: Toolbox) -> None:
        marker = "primary" if toolbox.coordinator.is_primary() else "secondary"
        for test_string in TEST_STRINGS:
            print(f"{marker} {test_string}", file=sys.stderr)

    yt_client = yt_instance.get_client()
    mesh = Mesh(node_count=2, process_per_node=1, gpu_per_process=0)
    run(
        checker,
        backend=GenericBackend(),
        yt_path=yt_path,
        mesh=mesh,
        yt_client=yt_client,
        docker_image=DOCKER_IMAGE,
        proxy_stderr_mode=mode,
    )
    captured = capsys.readouterr()
    match mode:
        case StderrMode.disabled:
            for s in TEST_STRINGS:
                assert f"{s}\n" not in captured.out
        case StderrMode.primary:
            for s in TEST_STRINGS:
                assert f"primary {s}\n" in captured.out
            for s in TEST_STRINGS:
                assert f"secondary {s}\n" not in captured.out
        case _:
            raise Exception(f"Unknown mode {mode}")


def test_with_realtime_table(yt_path: str, yt_instance: YtInstance, capsys: CaptureFixture[str]) -> None:
    def checker(toolbox: Toolbox) -> None:
        for test_string in TEST_STRINGS:
            print(test_string)

    yt_client = yt_instance.get_client()
    mesh = Mesh(node_count=1, process_per_node=1, gpu_per_process=0)
    run(
        checker,
        backend=GenericBackend(),
        yt_path=yt_path,
        mesh=mesh,
        yt_client=yt_client,
        docker_image=DOCKER_IMAGE,
        proxy_stderr_mode=StderrMode.primary,
        operation_log_mode=OperationLogMode.realtime_yt_table,
    )
    captured = capsys.readouterr()
    for s in TEST_STRINGS:
        assert f"{s}\n" in captured.out


def test_operation_cli(yt_instance: YtInstance, yt_path: str) -> None:
    yt_client = yt_instance.get_client()

    tracto_cli = TractoCli(
        command=["python3", "/tractorun_tests/stderr_script.py"],
        args=[
            "--yt-path",
            yt_path,
            "--user-config",
            json.dumps(
                {
                    "test_strings": TEST_STRINGS,
                },
            ),
            "--bind-local",
            f"{get_data_path('../data/stderr_script.py')}:/tractorun_tests/stderr_script.py",
            "--proxy-stderr-mode",
            StderrMode.primary,
        ],
    )
    op_run = tracto_cli.run()
    assert op_run.is_exitcode_valid()
    assert op_run.is_operation_state_valid(yt_client=yt_client, job_count=1)

    stdout = op_run.stdout.decode("utf-8")
    for s in TEST_STRINGS:
        assert f"{s}\n" in stdout


@pytest.mark.parametrize(
    "mode",
    [StderrMode.primary, StderrMode.disabled],
)
def test_stop_on_fail(mode: StderrMode, yt_path: str, yt_instance: YtInstance) -> None:
    def checker(toolbox: Toolbox) -> None:
        print("message", file=sys.stderr)
        raise Exception("fail operation")

    yt_client = yt_instance.get_client()
    mesh = Mesh(node_count=1, process_per_node=1, gpu_per_process=0)
    with pytest.raises(YtOperationFailedError):
        run(
            checker,
            backend=GenericBackend(),
            yt_path=yt_path,
            mesh=mesh,
            yt_client=yt_client,
            docker_image=DOCKER_IMAGE,
            proxy_stderr_mode=mode,
            yt_operation_spec={"max_failed_job_count": 1},
        )
    names = [thread.name for thread in threading.enumerate()]
    assert STDERR_READER_THREAD_NAME not in names


def test_multiple_processes(yt_path: str, yt_instance: YtInstance, capsys: CaptureFixture[str]) -> None:
    message = "test message"

    def checker(toolbox: Toolbox) -> None:
        print(message, file=sys.stderr)

    yt_client = yt_instance.get_client()
    mesh = Mesh(node_count=1, process_per_node=2, gpu_per_process=0)
    run(
        checker,
        backend=GenericBackend(),
        yt_path=yt_path,
        mesh=mesh,
        yt_client=yt_client,
        docker_image=DOCKER_IMAGE,
        proxy_stderr_mode=StderrMode.primary,
    )
    captured = capsys.readouterr()
    assert message in captured.out


def test_read_two_operations(yt_path: str, yt_instance: YtInstance, capsys: CaptureFixture[str]) -> None:
    key = "iter_id"

    def checker(toolbox: Toolbox) -> None:
        user_config = toolbox.get_user_config()
        id_ = user_config[key]
        print(f"message {id_}", file=sys.stderr)

    yt_client = yt_instance.get_client()
    mesh = Mesh(node_count=1, process_per_node=1, gpu_per_process=0)
    run(
        checker,
        backend=GenericBackend(),
        yt_path=yt_path,
        mesh=mesh,
        yt_client=yt_client,
        docker_image=DOCKER_IMAGE,
        proxy_stderr_mode=StderrMode.primary,
        user_config={key: "1"},
    )
    captured = capsys.readouterr()
    assert "message 1" in captured.out

    run(
        checker,
        backend=GenericBackend(),
        yt_path=yt_path,
        mesh=mesh,
        yt_client=yt_client,
        docker_image=DOCKER_IMAGE,
        proxy_stderr_mode=StderrMode.primary,
        user_config={key: "2"},
    )
    captured = capsys.readouterr()
    assert "message 2" in captured.out, captured.err
