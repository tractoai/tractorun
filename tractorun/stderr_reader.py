import base64
import enum
import pickle
import threading
import time
from typing import (
    Callable,
    Generator,
    Optional,
)

import attrs
from yt import wrapper as yt
from yt.wrapper.errors import YtError

from tractorun.mesh import Mesh
from tractorun.training_dir import TrainingDir


YT_RETRY_INTERVAL = 5


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class YtStderrReader:
    _stderr_getter: Callable[[], Optional[bytes]]
    _stop_on_none: bool = attrs.field(default=False)

    def get_output(self) -> Generator[bytes, None, None]:
        last = b""
        while True:
            current = self._stderr_getter()
            if current is None:
                if self._stop_on_none:
                    return
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
        return self.get_output()


def get_job_stderr(yt_client: yt.YtClient, operation_id: str, job_id: str, retry_interval: float = YT_RETRY_INTERVAL) -> Callable[[], bytes]:
    def _wrapped() -> bytes:
        while True:
            try:
                data = yt_client.get_job_stderr(operation_id=operation_id, job_id=job_id).read()
                return data
            except YtError:
                time.sleep(YT_RETRY_INTERVAL)
    return _wrapped


class StderrSource(str, enum.Enum):
    master = "master"
    all = "all"


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class StderrReaderWorker:
    _prev_incarnation_id: int
    _training_dir: TrainingDir
    _yt_client_config_pickled: str
    _source: StderrSource
    _mesh: Mesh
    _stop: bool = False

    _polling_interval: float = attrs.field(default=1.0)
    _yt_retry_interval: float = attrs.field(default=YT_RETRY_INTERVAL)

    def _start(self) -> None:
        yt_config = pickle.loads(base64.b64decode(self._yt_client_config_pickled))
        yt_client = yt.YtClient(config=yt_config)

        incarnation = self._prev_incarnation_id
        topology = []
        operation_id = None
        while incarnation == self._prev_incarnation_id or operation_id is None or len(topology) != self._mesh.node_count:
            try:
                incarnation = yt_client.get(self._training_dir.base_path + "/@incarnation_id")
                incarnation_path = self._training_dir.get_incarnation_path(incarnation)
                operation_id = yt_client.get(incarnation_path + "/@incarnation_operation_id")
                topology = yt_client.get(incarnation_path + "/@topology")
            except Exception:
                pass

        if self._source == StderrSource.master:
            job_ids = [topology[0]["job_id"]]
        else:
            job_ids = [el["job_id"] for el in topology]
        output_streams: list[str, Generator[bytes, None, None]] = []
        for job_id in job_ids:
            stderr_getter = get_job_stderr(yt_client=yt_client, operation_id=operation_id, job_id=job_id, retry_interval=self._yt_retry_interval)
            output_streams.append(
                (
                    job_id,YtStderrReader(stderr_getter=stderr_getter).get_output(),
                ),
            )

        while not self._stop:
            for job_id, output_stream in output_streams:
                try:
                    data = next(output_stream)
                    if data:
                        print(f"{job_id}\t{data}", end="")
                except Exception:
                    pass
            time.sleep(self._polling_interval)

    def stop(self):
        self._stop = True

    def start(self) -> None:
        stderr_thread = threading.Thread(target=self._start)
        stderr_thread.start()
