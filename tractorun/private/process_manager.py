import base64
import contextlib
import datetime
import enum
from multiprocessing import (
    Process,
    Queue,
)
import pickle
import queue
import selectors
import sys
import time
from typing import (
    TYPE_CHECKING,
    Generator,
    Mapping,
    NewType,
    Optional,
)

import attrs
from yt import type_info
from yt import wrapper as yt
from yt.common import update_inplace

from tractorun.mesh import Mesh
from tractorun.private.sidecar import (
    RestartVerdict,
    SidecarRun,
)
from tractorun.private.training_dir import TrainingDir
from tractorun.private.worker import WorkerRun
from tractorun.private.yt_cluster import TractorunClusterConfig
from tractorun.sidecar import Sidecar


SidecarIndex = NewType("SidecarIndex", int)


class OutputType(str, enum.Enum):
    stdout: str = "stdout"
    stderr: str = "stderr"


BULK_TABLE_WRITE_SIZE = 100
WAIT_LOG_RECORDS_TIMEOUT = 5
IO_QUEUE_MAXSIZE = 10000
YT_LOG_WRITER_JOIN_TIMEOUT = 10
QUEUE_TIMEOUT = 0.01


STOP_LOG_WRITER = object()


class _NoMessages:
    pass


class _LastMessage:
    pass


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class LogRecord:
    _message: str
    _datetime: datetime.datetime
    _fd: OutputType

    def _datetime_to_unixtime(self, datetime_obj: datetime.datetime) -> int:
        # workaround for https://github.com/ytsaurus/ytsaurus/issues/309
        return int(time.mktime(datetime_obj.timetuple()))

    @staticmethod
    def get_yt_schema() -> yt.schema.TableSchema:
        # I don't trust the implementation of yt_dataclass
        # so let's define transformations explicitly

        schema = yt.schema.TableSchema()
        schema.add_column("datetime", type_info.Datetime)
        schema.add_column("message", type_info.String)
        schema.add_column("fd", type_info.String)
        return schema

    def to_dict(self) -> dict:
        return {
            "message": self._message,
            "fd": self._fd.value,
            "datetime": self._datetime_to_unixtime(self._datetime),
        }


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class YtLogWriter:
    _yt_client_config: str
    _queue: Queue
    _log_path: str

    def _get_next_message(self) -> LogRecord | _NoMessages:
        try:
            return self._queue.get(timeout=QUEUE_TIMEOUT)
        except queue.Empty:
            return _NoMessages()

    def write_log(self) -> None:
        yt_config = pickle.loads(base64.b64decode(self._yt_client_config))
        yt_client = yt.YtClient(config=yt_config)
        yt_client.create(
            "table",
            self._log_path,
            attributes={"schema": LogRecord.get_yt_schema().to_yson_type()},
        )
        got_last_message = False
        while not got_last_message:
            messages: list[dict[str, str | int]] = []
            while True:
                message = self._get_next_message()
                match message:
                    case _NoMessages():
                        time.sleep(WAIT_LOG_RECORDS_TIMEOUT)
                    case _LastMessage():
                        got_last_message = True
                        break
                    case LogRecord():
                        messages.append(message.to_dict())
                        if len(messages) >= BULK_TABLE_WRITE_SIZE:
                            break
            if not messages and not got_last_message:
                time.sleep(WAIT_LOG_RECORDS_TIMEOUT)
                continue
            yt_client.write_table(
                yt.TablePath(self._log_path, append=True),
                messages,
            )


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class SelectorMeta:
    queue: Queue
    output_type: OutputType


class ProcessManagerPollStatus(enum.IntEnum):
    success: int = 0
    fail: int = 1
    running: int = -1


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class SidecarRunMeta:
    sidecar_run: SidecarRun
    queue: Queue


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class ProcessManager:
    _sidecar_runs: dict[SidecarIndex, SidecarRunMeta]
    _worker_runs: list[WorkerRun]
    _io_queues: list[Queue]
    _log_writers: list[Process]
    _io_selector: selectors.DefaultSelector

    @classmethod
    @contextlib.contextmanager
    def start(
        cls,
        command: list[str],
        sidecars: list[Sidecar],
        training_dir: TrainingDir,
        yt_client_config: str,
        cluster_config: TractorunClusterConfig,
        mesh: Mesh,
        node_index: int,
        os_environ: Mapping,
        tp_env: dict,
        spec_env: dict,
    ) -> Generator["ProcessManager", None, None]:
        pm = ProcessManager._start(
            command=command,
            sidecars=sidecars,
            training_dir=training_dir,
            yt_client_config=yt_client_config,
            cluster_config=cluster_config,
            mesh=mesh,
            node_index=node_index,
            os_environ=os_environ,
            tp_env=tp_env,
            spec_env=spec_env,
        )
        yield pm
        pm.stop()

    @classmethod
    def _start(
        cls,
        command: list[str],
        sidecars: list[Sidecar],
        training_dir: TrainingDir,
        yt_client_config: str,
        cluster_config: TractorunClusterConfig,
        mesh: Mesh,
        node_index: int,
        os_environ: Mapping,
        tp_env: dict,
        spec_env: dict,
    ) -> "ProcessManager":
        worker_runs = []
        sidecar_runs: dict[SidecarIndex, SidecarRun] = {}
        selector = selectors.DefaultSelector()

        yt_config = pickle.loads(base64.b64decode(yt_client_config))
        update_inplace(
            yt_config,
            {
                "pickling": {
                    "module_filter": None,
                },
            },
        )

        for proc_index in range(mesh.process_per_node):
            port = int(os_environ[f"YT_PORT_{proc_index}"])
            self_index = node_index * mesh.process_per_node + proc_index
            worker_run = WorkerRun.run(
                command=command,
                mesh=mesh,
                self_index=self_index,
                node_index=node_index,
                proc_index=proc_index,
                port=port,
                training_dir=training_dir,
                cluster_config=cluster_config,
                yt_config=yt_config,
                env={
                    **os_environ,
                    **tp_env,
                    **spec_env,
                },
            )
            worker_runs.append(worker_run)

        for local_index, sidecar in enumerate(sidecars):
            sidecar_index = SidecarIndex(node_index * len(sidecars) + local_index)
            sidecar_run = SidecarRun.run(
                sidecar=sidecar,
                env={
                    **os_environ,
                    "YT_PROXY": yt_config["proxy"]["url"],
                    "YT_TOKEN": yt_config["token"],
                    **spec_env,  # TODO(gritukan): Make separate env for sidecars
                },
            )
            sidecar_runs[sidecar_index] = sidecar_run

        yt_log_writers: list[Process] = []
        io_queues: list[Queue] = []
        for worker_run in worker_runs:
            io_queue: Queue = Queue(maxsize=IO_QUEUE_MAXSIZE)
            selector.register(
                worker_run.stdout(),
                selectors.EVENT_READ,
                data=SelectorMeta(queue=io_queue, output_type=OutputType.stdout),
            )
            selector.register(
                worker_run.stderr(),
                selectors.EVENT_READ,
                data=SelectorMeta(queue=io_queue, output_type=OutputType.stderr),
            )
            io_queues.append(io_queue)
            yt_log_writer = Process(
                target=YtLogWriter(
                    yt_client_config=yt_client_config,
                    queue=io_queue,
                    log_path=f"{training_dir.worker_logs_path}/{worker_run.worker_config.self_index}",
                ).write_log
            )
            yt_log_writer.start()
            yt_log_writers.append(yt_log_writer)

        sidecar_runs_meta: dict[SidecarIndex, SidecarRunMeta] = {}
        for sidecar_index, sidecar_run in sidecar_runs.items():
            io_queue = Queue(maxsize=IO_QUEUE_MAXSIZE)
            selector.register(
                sidecar_run.stdout(),
                selectors.EVENT_READ,
                data=SelectorMeta(queue=io_queue, output_type=OutputType.stdout),
            )
            selector.register(
                sidecar_run.stderr(),
                selectors.EVENT_READ,
                data=SelectorMeta(queue=io_queue, output_type=OutputType.stderr),
            )
            io_queues.append(io_queue)
            yt_log_writer = Process(
                target=YtLogWriter(
                    yt_client_config=yt_client_config,
                    queue=io_queue,
                    log_path=f"{training_dir.sidecar_logs_path}/{sidecar_index}",
                ).write_log
            )
            yt_log_writer.start()
            yt_log_writers.append(yt_log_writer)
            sidecar_runs_meta[sidecar_index] = SidecarRunMeta(
                sidecar_run=sidecar_run,
                queue=io_queue,
            )
        return ProcessManager(
            sidecar_runs=sidecar_runs_meta,
            worker_runs=worker_runs,
            io_queues=io_queues,
            log_writers=yt_log_writers,
            io_selector=selector,
        )

    def _restart_sidecar(self, sidecar_index: SidecarIndex) -> None:
        sidecar_run_meta = self._sidecar_runs[sidecar_index]
        sidecar_run = sidecar_run_meta.sidecar_run
        io_queue = sidecar_run_meta.queue
        self._io_selector.unregister(sidecar_run.stdout())
        self._io_selector.unregister(sidecar_run.stderr())

        sidecar_run = sidecar_run.restart()
        new_sidecar_run_meta = SidecarRunMeta(
            sidecar_run=sidecar_run,
            queue=io_queue,
        )
        self._sidecar_runs[sidecar_index] = new_sidecar_run_meta
        self._io_selector.register(
            sidecar_run.stdout(),
            selectors.EVENT_READ,
            data=SelectorMeta(queue=io_queue, output_type=OutputType.stdout),
        )
        self._io_selector.register(
            sidecar_run.stderr(),
            selectors.EVENT_READ,
            data=SelectorMeta(queue=io_queue, output_type=OutputType.stderr),
        )

    def poll(self) -> ProcessManagerPollStatus:
        status = self._poll_processes()
        self._process_logs()
        return status

    def _poll_processes(self) -> ProcessManagerPollStatus:
        exit_codes = [worker_run.poll() for worker_run in self._worker_runs]
        match check_status(exit_codes):
            case PoolStatus.failed:
                return ProcessManagerPollStatus.fail
            case PoolStatus.success:
                return ProcessManagerPollStatus.success

        for sidecar_index, sidecar_run_meta in self._sidecar_runs.items():
            sidecar_run = sidecar_run_meta.sidecar_run
            match sidecar_run.need_restart():
                case RestartVerdict.restart:
                    print(f"Restart sidecar {sidecar_run.command}", file=sys.stderr)
                    self._restart_sidecar(sidecar_index)
                case RestartVerdict.fail:
                    print(f"Sidecar {sidecar_run.command} has been failed", file=sys.stderr)
                    return ProcessManagerPollStatus.fail
                case RestartVerdict.skip:
                    pass
                case RestartVerdict.unknown:
                    print(f"Warning: unknown restart policy for {sidecar_run.command}", file=sys.stderr)
                case _:
                    print(f"Warning: unknown restart verdict for {sidecar_run.command}", file=sys.stderr)
        return ProcessManagerPollStatus.running

    def _process_logs(self) -> None:
        for key, _ in self._io_selector.select(timeout=-1):
            if TYPE_CHECKING:
                assert isinstance(key, selectors.SelectorKey)
            meta: SelectorMeta = key.data
            io_queue = meta.queue
            lines = key.fileobj.readlines()  # type: ignore
            if not lines:
                continue
            for line in lines:
                line = line.rstrip()
                print(line, file=sys.stderr)
                record = LogRecord(
                    message=line,
                    fd=meta.output_type,
                    datetime=datetime.datetime.now(),
                )
                try:
                    io_queue.put(record, timeout=QUEUE_TIMEOUT)
                except queue.Full:
                    print("io queue is full, can't write data to the table", file=sys.stderr)

    def stop(self) -> None:
        for io_queue in self._io_queues:
            try:
                io_queue.put_nowait(_LastMessage())
            except queue.Full:
                pass
        for worker_run in self._worker_runs:
            worker_run.terminate()
        for sidecar_run_meta in self._sidecar_runs.values():
            sidecar_run_meta.sidecar_run.terminate()
        self._io_selector.close()
        for yt_log_writer in self._log_writers:
            yt_log_writer.join(timeout=YT_LOG_WRITER_JOIN_TIMEOUT)
            if yt_log_writer.is_alive():
                yt_log_writer.kill()
                yt_log_writer.join()
            yt_log_writer.close()
        for io_queue in self._io_queues:
            try:
                io_queue.put_nowait(_LastMessage())
            except queue.Full:
                pass
            io_queue.close()


def has_failed(exit_codes: list[Optional[int]]) -> bool:
    return any(code is not None and code != 0 for code in exit_codes)


def is_success(exit_codes: list[Optional[int]]) -> bool:
    return all(code == 0 for code in exit_codes)


class PoolStatus(enum.IntEnum):
    running = enum.auto()
    success = enum.auto()
    failed = enum.auto()


def check_status(exit_codes: list[Optional[int]]) -> PoolStatus:
    if has_failed(exit_codes):
        return PoolStatus.failed
    if is_success(exit_codes):
        return PoolStatus.success
    return PoolStatus.running
