import json
import sys

from _pytest.capture import CaptureFixture
import attrs
import pytest

from tests.utils import DOCKER_IMAGE, TractoCli, get_data_path, run_config_file
from tests.yt_instances import YtInstance
from tractorun.backend.generic import GenericBackend
from tractorun.mesh import Mesh
from tractorun.run import run
from tractorun.stderr_reader import YtStderrReader, StderrMode
from tractorun.toolbox import Toolbox


TEST_STRINGS = ["hello", "my dear", "friend"]


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
def test_operation_pickling(mode: StderrMode, yt_path: str, yt_instance: YtInstance, capsys: CaptureFixture[str]) -> None:
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


def test_operation_cli_args(yt_instance: YtInstance, yt_path: str) -> None:
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


def test_operation_cli_config(yt_instance: YtInstance, yt_path: str) -> None:
    yt_client = yt_instance.get_client()

    run_config = {
        "proxy_stderr_mode": StderrMode.primary.value,
    }

    with run_config_file(run_config) as run_config_path:
        tracto_cli = TractoCli(
            command=["python3", "/tractorun_tests/stderr_script.py"],
            args=[
                "--run-config-path",
                run_config_path,
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
            ],
        )
        op_run = tracto_cli.run()
    assert op_run.is_exitcode_valid()
    assert op_run.is_operation_state_valid(yt_client=yt_client, job_count=1)

    stdout = op_run.stdout.decode("utf-8")
    for s in TEST_STRINGS:
        assert f"{s}\n" in stdout
