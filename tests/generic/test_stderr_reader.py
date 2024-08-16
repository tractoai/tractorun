import sys
from typing import Generator

import attrs
import pytest
from _pytest.capture import CaptureFixture
from yt import wrapper as yt

from tests.utils import DOCKER_IMAGE
from tests.yt_instances import YtInstance
from tractorun.backend.generic import GenericBackend
from tractorun.mesh import Mesh
from tractorun.run import run
from tractorun.stderr_reader import YtStderrReader
from tractorun.toolbox import Toolbox


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


def test_stderr_from_operation_pickling(yt_path: str, yt_instance: YtInstance, capsys: CaptureFixture[str]):
    test_strings = ["hello", "my dear", "friend"]

    def checker(toolbox: Toolbox) -> None:
        for line in test_strings:
            print(line, file=sys.stderr)

    yt_client = yt_instance.get_client()
    mesh = Mesh(node_count=1, process_per_node=1, gpu_per_process=0)
    run(
        checker,
        backend=GenericBackend(),
        yt_path=yt_path,
        mesh=mesh,
        yt_client=yt_client,
        docker_image=DOCKER_IMAGE,
    )
    captured = capsys.readouterr()
    for line in test_strings:
        assert f"{line}\\n" in captured.out
