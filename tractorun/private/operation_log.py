import abc
import base64
import datetime
import logging
from multiprocessing import (
    Process,
    Queue,
)
import pickle
import queue
import time

import attrs
from yt import type_info
from yt import wrapper as yt

from tractorun.private.sidecar import (
    SidecarIndex,
    SidecarRun,
)
from tractorun.private.training_dir import TrainingDir
from tractorun.private.worker import (
    WorkerIndex,
    WorkerRun,
)


BULK_TABLE_WRITE_SIZE = 100
WAIT_LOG_RECORDS_TIMEOUT = 5
IO_QUEUE_MAXSIZE = 10000
YT_LOG_WRITER_JOIN_TIMEOUT = 10
QUEUE_TIMEOUT = 0.01


_LOGGER = logging.getLogger(__name__)


class _NoMessages:
    pass


class _LastMessage:
    pass


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class LogRecord:
    _message: str
    _datetime: datetime.datetime
    _fd: str

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
            "fd": self._fd,
            "datetime": self._datetime_to_unixtime(self._datetime),
        }


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
            _LOGGER.warning("io queue is full, can't write data to the table")

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
class YtLogWriter:
    _yt_client_config: str
    _queue: Queue
    _log_path: str

    def __call__(self) -> None:
        return self._write_log()

    def _write_log(self) -> None:
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

    def _get_next_message(self) -> LogRecord | _NoMessages:
        try:
            return self._queue.get(timeout=QUEUE_TIMEOUT)
        except queue.Empty:
            return _NoMessages()
