import base64
import enum
import fcntl
import json
import multiprocessing
import os
import pickle
import queue
import selectors
from selectors import SelectorKey
import subprocess
import sys
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Optional,
)
import warnings

import attrs
from yt import type_info
from yt.common import update_inplace
import yt.wrapper as yt

from tractorun import __version__
from tractorun.env import EnvVariable
from tractorun.mesh import Mesh
from tractorun.private.constants import TRACTO_CONFIG_ENV_VAR
from tractorun.private.helpers import AttrSerializer
from tractorun.private.sidecar import (
    RestartVerdict,
    SidecarRun,
)
from tractorun.private.tensorproxy import TensorproxyBootstrap
from tractorun.private.training_dir import TrainingDir
from tractorun.private.yt_cluster import TractorunClusterConfig
from tractorun.sidecar import Sidecar


PROCESSES_POLL_TIMEOUT = 10
WAIT_LOG_RECORDS_TIMEOUT = 5
BULK_TABLE_WRITE_SIZE = 1000
IO_QUEUE_MAXSIZE = 10000


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class LibVersions:
    tractorun: str
    ytsaurus_client: str

    @staticmethod
    def create() -> "LibVersions":
        return LibVersions(
            tractorun=__version__,
            ytsaurus_client=yt.__version__,
        )


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class ProcConfig:
    mesh: Mesh
    port: int
    node_index: int
    proc_index: int
    self_index: int
    training_dir: TrainingDir
    yt_client_config: str
    cluster_config: TractorunClusterConfig


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class BootstrapConfig:
    mesh: Mesh
    sidecars: list[Sidecar]
    env: list[EnvVariable]
    training_dir: TrainingDir
    yt_client_config: str
    tensorproxy: Optional[TensorproxyBootstrap]
    lib_versions: LibVersions
    cluster_config: TractorunClusterConfig


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class LogRecord:
    _self_index: int
    _message: str
    _source: str
    _fd: Literal["stdout"] | Literal["stderr"]

    @staticmethod
    def get_yt_schema() -> yt.schema.TableSchema:
        # I don't trust the implementation of yt_dataclass
        # so let's define transformations explicitly

        schema = yt.schema.TableSchema()
        schema.add_column("self_index", type_info.Int16)
        schema.add_column("message", type_info.String)
        schema.add_column("source", type_info.String)
        schema.add_column("fd", type_info.String)
        return schema

    def to_dict(self) -> dict:
        return {
            "self_index": self._self_index,
            "message": self._message,
            "source": self._source,
            "fd": self._fd,
        }


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class YtLogWriter:
    _yt_client_config: str
    _queue: multiprocessing.Queue
    _log_path: str

    def write_log(self) -> None:
        yt_config = pickle.loads(base64.b64decode(self._yt_client_config))
        yt_client = yt.YtClient(config=yt_config)
        yt_client.create("table", self._log_path, attributes={"schema": LogRecord.get_yt_schema().to_yson_type()})
        while True:
            messages: list[dict[str, str | int]] = []
            while not self._queue.empty():
                message: LogRecord = self._queue.get(block=True, timeout=None)
                messages.append(message.to_dict())
                if len(messages) >= BULK_TABLE_WRITE_SIZE:
                    break
            if not messages:
                time.sleep(WAIT_LOG_RECORDS_TIMEOUT)
            yt_client.write_table(
                yt.TablePath(self._log_path, append=True),
                messages,
            )


def check_lib_versions(local_lib_versions: LibVersions) -> None:
    remote_lib_versions = LibVersions.create()
    diff = {}
    if local_lib_versions.tractorun != remote_lib_versions.tractorun:
        diff["tractorun"] = {
            "local": local_lib_versions.tractorun,
            "remote": remote_lib_versions.tractorun,
        }
    if local_lib_versions.ytsaurus_client != remote_lib_versions.ytsaurus_client:
        diff["ytsaurus-client"] = {
            "local": local_lib_versions.ytsaurus_client,
            "remote": remote_lib_versions.ytsaurus_client,
        }
    if diff:
        warnings.warn(f"Local and remote libraries has different versions: {diff}")


def set_nonblocking(fd: Any) -> None:
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)


def bootstrap(
    mesh: Mesh,
    training_dir: TrainingDir,
    yt_client_config: str,
    command: list[str],
    sidecars: list[Sidecar],
    env: list[EnvVariable],
    tensorproxy: Optional[TensorproxyBootstrap],
    lib_versions: LibVersions,
    cluster_config: TractorunClusterConfig,
) -> None:
    # Runs inside a job

    check_lib_versions(local_lib_versions=lib_versions)

    processes = []
    yt_config = pickle.loads(base64.b64decode(yt_client_config))
    yt_client = yt.YtClient(config=yt_config)

    tp_sidecars, tp_env = [], {}
    if tensorproxy is not None:
        total_ports = mesh.process_per_node + tensorproxy.ports_count
        tp_grpc_port, tp_mon_port = range(total_ports)[mesh.process_per_node :]

        tp_sidecars = tensorproxy.prepare_and_get_sidecars(
            yt_proxy=yt_config["proxy"]["url"],
            grpc_port=tp_grpc_port,
            monitoring_port=tp_mon_port,
        )
        tp_env = tensorproxy.get_environment(grpc_port=tp_grpc_port)

    spec_env = {}
    for var in env:
        if var.cypress_path is not None:
            spec_env[var.name] = yt_client.get(var.cypress_path)
        else:
            spec_env[var.name] = var.value

    sidecars = sidecars + tp_sidecars

    selector = selectors.DefaultSelector()

    for i in range(mesh.process_per_node):
        update_inplace(
            yt_config,
            {
                "pickling": {
                    "module_filter": None,
                },
            },
        )

        node_index = int(os.environ["YT_JOB_COOKIE"])
        proc_index = i
        self_index = node_index * mesh.process_per_node + proc_index

        proc_config = ProcConfig(
            mesh=mesh,
            node_index=node_index,
            proc_index=proc_index,
            self_index=self_index,
            port=int(os.environ[f"YT_PORT_{i}"]),
            training_dir=training_dir,
            yt_client_config=base64.b64encode(pickle.dumps(yt_config)).decode("utf-8"),
            cluster_config=cluster_config,
        )
        config_name = f"config_{i}.json"
        with open(config_name, "w") as f:
            serializer = AttrSerializer(ProcConfig)
            json.dump(serializer.serialize(proc_config), f)

        # TODO: torch multiprocessing is better (or another backend-specific tool),
        # but pickling does not work in this case.
        # torch.multiprocessing.spawn(wrapped_run_mp, nprocs=mesh.process_per_node, args=(f, c, path,), join=True)
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,
            env={
                **os.environ,
                TRACTO_CONFIG_ENV_VAR: config_name,
                "YT_PROXY": yt_config["proxy"]["url"],
                "YT_TOKEN": yt_config["token"],
                **tp_env,
                **spec_env,
            },
        )
        set_nonblocking(process.stdout.fileno())
        set_nonblocking(process.stderr.fileno())
        selector.register(process.stdout, selectors.EVENT_READ, data=proc_config)
        selector.register(process.stderr, selectors.EVENT_READ, data=proc_config)
        processes.append(process)

    sidecar_runs: list[SidecarRun] = []
    for sidecar in sidecars:
        sidecar_run = SidecarRun.run(
            sidecar=sidecar,
            env={
                **os.environ,
                "YT_PROXY": yt_config["proxy"]["url"],
                "YT_TOKEN": yt_config["token"],
                **spec_env,  # TODO(gritukan): Make separate env for sidecars
            },
        )
        set_nonblocking(sidecar_run.stdout().fileno())
        set_nonblocking(sidecar_run.stderr().fileno())
        selector.register(sidecar_run.stdout(), selectors.EVENT_READ, data=sidecar_run)
        selector.register(sidecar_run.stderr(), selectors.EVENT_READ, data=sidecar_run)
        sidecar_runs.append(sidecar_run)

    io_queue: multiprocessing.Queue = multiprocessing.Queue(maxsize=IO_QUEUE_MAXSIZE)
    yt_log_writer = multiprocessing.Process(
        target=YtLogWriter(
            yt_client_config=yt_client_config,
            queue=io_queue,
            log_path=f"{training_dir.raw_logs_path}/0",
        ).write_log
    )
    yt_log_writer.start()
    while True:
        time.sleep(PROCESSES_POLL_TIMEOUT)
        exit_codes = [process.poll() for process in processes]
        match check_status(exit_codes):
            case PoolStatus.failed:
                for run in sidecar_runs:
                    run.terminate()
                yt_log_writer.terminate()
                # yt_log_writer.join()
                yt_log_writer.close()
                # extract to a function and return status
                sys.exit(1)
            case PoolStatus.success:
                for run in sidecar_runs:
                    run.terminate()
                yt_log_writer.terminate()
                # yt_log_writer.join()
                yt_log_writer.close()
                return

        for run in sidecar_runs:
            match run.need_restart():
                case RestartVerdict.restart:
                    print(f"Restart sidecar {run.command}", file=sys.stderr)
                    run.restart()
                case RestartVerdict.fail:
                    print(f"Sidecar {run.command} has been failed", file=sys.stderr)
                    sys.exit(1)
                case RestartVerdict.skip:
                    pass
                case RestartVerdict.unknown:
                    print(f"Warning: unknown restart policy for {run.command}", file=sys.stderr)
                case _:
                    print(f"Warning: unknown restart verdict for {run.command}", file=sys.stderr)

        for key, _ in selector.select(timeout=-1):
            if TYPE_CHECKING:
                assert isinstance(key, SelectorKey)
            lines = key.fileobj.readlines()
            if not lines:
                break
            for line in lines:
                line = line.rstrip()
                print(line, file=sys.stderr)
                record = LogRecord(
                    message=line,
                    self_index=0,
                    fd="stdout",
                    source="foo",
                )
                try:
                    io_queue.put_nowait(record)
                except queue.Full:
                    print("io queue is full, can't write data on YT", file=sys.stderr)
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
