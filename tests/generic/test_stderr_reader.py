import attrs
import pytest

from tractorun.stderr_reader import YtStderrReader


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class TestStderrProvider:
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
        # (
        #     [b"ab", b"ab"],
        #     [b"ab", b"ab"],
        # ),
        (
            [b"a", b"abcde", b"defg", b"abc"],
            [b"a", b"bcde", b"fg", b"abc"],
        ),
    ],
)
def test_stderr_reader(lines: list[bytes], expected: list[bytes]) -> None:
    mock = TestStderrProvider(lines=lines)
    reader = YtStderrReader(stderr_getter=mock.read, stop_on_none=True)
    result = list(reader.tail_output())
    assert result == expected
