import abc
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


WorkerIndex = NewType("WorkerIndex", int)
SidecarIndex = NewType("SidecarIndex", int)


class OutputType(str, enum.Enum):
    stdout: str = "stdout"
    stderr: str = "stderr"


BULK_TABLE_WRITE_SIZE = 100
WAIT_LOG_RECORDS_TIMEOUT = 5
IO_QUEUE_MAXSIZE = 10000
YT_LOG_WRITER_JOIN_TIMEOUT = 10
QUEUE_TIMEOUT = 0.01


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

    def __call__(self) -> None:
        return self._write_log()

    def _write_log(self) -> None:
        import os

        pid = os.getpid()
        yt_config = pickle.loads(base64.b64decode(self._yt_client_config))
        yt_client = yt.YtClient(config=yt_config)
        yt_client.create(
            "table",
            self._log_path,
            attributes={"schema": LogRecord.get_yt_schema().to_yson_type()},
        )
        got_last_message = False
        print(pid, "hello from writer", file=sys.stderr)
        while not got_last_message:
            messages: list[dict[str, str | int]] = []
            while True:
                message = self._get_next_message()
                print(pid, "got message", message, file=sys.stderr)
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
                print(pid, "all messages", messages, file=sys.stderr)
            print(pid, "out of cycle")
            if not messages and not got_last_message:
                print(pid, "wait", file=sys.stderr)
                time.sleep(WAIT_LOG_RECORDS_TIMEOUT)
                continue
            print(pid, "going to write messages", messages, file=sys.stderr)
            yt_client.write_table(
                yt.TablePath(self._log_path, append=True),
                messages,
            )

    def _get_next_message(self) -> LogRecord | _NoMessages:
        try:
            return self._queue.get(timeout=QUEUE_TIMEOUT)
        except queue.Empty:
            return _NoMessages()


class ProcessManagerPollStatus(enum.IntEnum):
    success: int = 0
    fail: int = 1
    running: int = -1


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class LogHandler(abc.ABC):
    @abc.abstractmethod
    def process(self, record: LogRecord) -> None:
        pass

    @abc.abstractmethod
    def stop(self) -> None:
        pass


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class LogHandlerFactory(abc.ABC):
    @abc.abstractmethod
    def create_for_worker(self, worker_index: WorkerIndex, worker_run: WorkerRun) -> LogHandler:
        pass

    @abc.abstractmethod
    def create_for_sidecar(self, sidecar_index: SidecarIndex, sidecar_run: SidecarRun) -> LogHandler:
        pass


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class YTLogHandler(LogHandler):
    _queue: Queue
    _log_writer: Process

    @classmethod
    def start(cls, yt_client_config: str, log_path: str) -> "YTLogHandler":
        io_queue: Queue = Queue(maxsize=IO_QUEUE_MAXSIZE)
        yt_log_writer = Process(
            target=YtLogWriter(
                yt_client_config=yt_client_config,
                queue=io_queue,
                log_path=log_path,
            )
        )
        yt_log_writer.start()
        return YTLogHandler(queue=io_queue, log_writer=yt_log_writer)

    def process(self, record: LogRecord) -> None:
        try:
            self._queue.put(record, timeout=QUEUE_TIMEOUT)
        except queue.Full:
            print("io queue is full, can't write data to the table", file=sys.stderr)

    def stop(self) -> None:
        self._queue.put(_LastMessage())
        self._log_writer.join(timeout=YT_LOG_WRITER_JOIN_TIMEOUT)
        if self._log_writer.is_alive():
            self._log_writer.kill()
            self._log_writer.join()
        self._log_writer.close()
        self._queue.close()


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class YTLogHandlerFactory(LogHandlerFactory):
    _yt_client_config: str
    _training_dir: TrainingDir

    def create_for_worker(self, worker_index: WorkerIndex, worker_run: WorkerRun) -> YTLogHandler:
        log_path = f"{self._training_dir.worker_logs_path}/{worker_index}"
        return YTLogHandler.start(log_path=log_path, yt_client_config=self._yt_client_config)

    def create_for_sidecar(self, sidecar_index: SidecarIndex, sidecar_run: SidecarRun) -> YTLogHandler:
        log_path = f"{self._training_dir.sidecar_logs_path}/{sidecar_index}"
        return YTLogHandler.start(log_path=log_path, yt_client_config=self._yt_client_config)


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class SelectorMeta:
    log_handlers: list[LogHandler]
    output_type: OutputType


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class SidecarRunMeta:
    sidecar_run: SidecarRun
    log_handlers: list[LogHandler]


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class ProcessManager:
    _sidecar_runs: dict[SidecarIndex, SidecarRunMeta]
    _worker_runs: dict[WorkerIndex, WorkerRun]
    _io_selector: selectors.DefaultSelector
    _log_handlers: list[LogHandler]

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
        log_handler_factories: list[LogHandlerFactory],
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
            log_handler_factories=log_handler_factories,
        )
        yield pm
        pm._stop()

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
        log_handler_factories: list[LogHandlerFactory],
    ) -> "ProcessManager":
        worker_runs: dict[WorkerIndex, WorkerRun] = {}
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
            worker_index = WorkerIndex(self_index)
            worker_runs[worker_index] = worker_run

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

        log_handlers = []

        for worker_index, worker_run in worker_runs.items():
            worker_log_handlers = []
            for factory in log_handler_factories:
                handler = factory.create_for_worker(worker_index, worker_run)
                log_handlers.append(handler)
                worker_log_handlers.append(handler)
            fds = {
                OutputType.stdout: worker_run.stdout(),
                OutputType.stderr: worker_run.stderr(),
            }
            for output_type, fd in fds.items():
                selector.register(
                    fd,
                    selectors.EVENT_READ,
                    data=SelectorMeta(log_handlers=worker_log_handlers, output_type=output_type),
                )

        sidecar_runs_meta: dict[SidecarIndex, SidecarRunMeta] = {}
        for sidecar_index, sidecar_run in sidecar_runs.items():
            sidecar_log_handlers = []
            for factory in log_handler_factories:
                handler = factory.create_for_sidecar(sidecar_index, sidecar_run)
                log_handlers.append(handler)
                sidecar_log_handlers.append(handler)
            fds = {
                OutputType.stdout: sidecar_run.stdout(),
                OutputType.stderr: sidecar_run.stderr(),
            }
            for output_type, fd in fds.items():
                selector.register(
                    fd,
                    selectors.EVENT_READ,
                    data=SelectorMeta(log_handlers=sidecar_log_handlers, output_type=output_type),
                )
            sidecar_runs_meta[sidecar_index] = SidecarRunMeta(
                sidecar_run=sidecar_run,
                log_handlers=sidecar_log_handlers,
            )
        return ProcessManager(
            sidecar_runs=sidecar_runs_meta,
            worker_runs=worker_runs,
            io_selector=selector,
            log_handlers=log_handlers,
        )

    def _restart_sidecar(self, sidecar_index: SidecarIndex) -> None:
        sidecar_run_meta = self._sidecar_runs[sidecar_index]
        sidecar_run = sidecar_run_meta.sidecar_run
        log_handlers = sidecar_run_meta.log_handlers
        for fd in [sidecar_run.stdout(), sidecar_run.stderr()]:
            self._io_selector.unregister(fd)

        sidecar_run = sidecar_run.restart()
        new_sidecar_run_meta = SidecarRunMeta(
            sidecar_run=sidecar_run,
            log_handlers=log_handlers,
        )
        self._sidecar_runs[sidecar_index] = new_sidecar_run_meta

        fds = {
            OutputType.stdout: sidecar_run.stdout(),
            OutputType.stderr: sidecar_run.stderr(),
        }
        for output_type, fd in fds.items():
            self._io_selector.register(
                fd,
                selectors.EVENT_READ,
                data=SelectorMeta(log_handlers=log_handlers, output_type=output_type),
            )

    def poll(self) -> ProcessManagerPollStatus:
        status = self._poll_processes()
        self._process_logs()
        return status

    def _poll_processes(self) -> ProcessManagerPollStatus:
        exit_codes = [worker_run.poll() for worker_run in self._worker_runs.values()]
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
                for handler in meta.log_handlers:
                    handler.process(record)

    def _stop(self) -> None:
        for worker_run in self._worker_runs.values():
            worker_run.terminate()
        for sidecar_run_meta in self._sidecar_runs.values():
            sidecar_run_meta.sidecar_run.terminate()
        self._io_selector.close()
        for handler in self._log_handlers:
            handler.stop()


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
