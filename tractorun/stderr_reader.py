from typing import (
    Callable,
    Generator,
    Optional,
)

import attrs


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class YtStderrReader:
    _stderr_getter: Callable[[], Optional[bytes]]
    _stop_on_none: bool = attrs.field(default=False)

    def tail_output(self) -> Generator[bytes, None, None]:
        last = b""
        while True:
            current = self._stderr_getter()
            if self._stop_on_none and current is None:
                return
            else:
                current = b""
            new_data = self._get_new_data(last, current)
            yield new_data
            last = current

    def _prefix_function(self, s: bytes) -> list[int]:
        pi = [0] * len(s)
        for i in range(1, len(s)):
            j = pi[i - 1]
            while j > 0 and s[i] != s[j]:
                j = pi[j - 1]
            if s[i] == s[j]:
                j += 1
            pi[i] = j
        return pi

    def _get_new_data(self, s: bytes, t: bytes) -> bytes:
        uuid = b"8a5c221d-c111d561-13440384-186"
        k = t + uuid + s
        pi = self._prefix_function(k)
        max_pi = 0
        for i in range(len(t) + len(uuid), len(k)):
            max_pi = max(max_pi, pi[i])
        return t[max_pi:]

    def __iter__(self) -> Generator[bytes, None, None]:
        return self.tail_output()
