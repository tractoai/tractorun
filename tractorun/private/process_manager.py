import base64
import contextlib
import datetime
import enum
import logging
from pathlib import Path
import pickle
import selectors
import sys
from typing import (
    TYPE_CHECKING,
    Generator,
    Mapping,
    Optional,
)

import attrs
from yt.common import update_inplace

from tractorun.mesh import Mesh
from tractorun.private.operation_log import (
    LogHandler,
    LogHandlerFactory,
    LogRecord,
)
from tractorun.private.sidecar import (
    RestartVerdict,
    SidecarIndex,
    SidecarRun,
)
from tractorun.private.training_dir import TrainingDir
from tractorun.private.worker import (
    WorkerIndex,
    WorkerRun,
)
from tractorun.private.yt_cluster import TractorunClusterConfig
from tractorun.sidecar import Sidecar


_LOGGER = logging.getLogger(__name__)


class OutputType(str, enum.Enum):
    stdout: str = "stdout"
    stderr: str = "stderr"


class ProcessManagerPollStatus(enum.IntEnum):
    success: int = 0
    fail: int = 1
    running: int = -1


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
        sandbox_path: Path,
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
            sandbox_path=sandbox_path,
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
        sandbox_path: Path,
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
            _LOGGER.debug("Starting worker %s: %s", self_index, command)
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
                sandbox_path=sandbox_path,
            )
            _LOGGER.debug("Worker %s started", self_index)
            worker_index = WorkerIndex(self_index)
            worker_runs[worker_index] = worker_run

        for local_index, sidecar in enumerate(sidecars):
            sidecar_index = SidecarIndex(node_index * len(sidecars) + local_index)
            _LOGGER.debug("Starting sidecar %s: %s", local_index, sidecar.command)
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
        status, sidecars_to_restart = self._poll_processes()
        self._process_logs()
        for sidecar_index in sidecars_to_restart:
            self._restart_sidecar(sidecar_index)
        return status

    def _poll_processes(self) -> tuple[ProcessManagerPollStatus, list[SidecarIndex]]:
        exit_codes = [worker_run.poll() for worker_run in self._worker_runs.values()]
        match check_status(exit_codes):
            case PoolStatus.failed:
                return ProcessManagerPollStatus.fail, []
            case PoolStatus.success:
                return ProcessManagerPollStatus.success, []

        sidecars_to_restart: list[SidecarIndex] = []
        for sidecar_index, sidecar_run_meta in self._sidecar_runs.items():
            sidecar_run = sidecar_run_meta.sidecar_run
            match sidecar_run.need_restart():
                case RestartVerdict.restart:
                    _LOGGER.info("Restarting sidecar %s", sidecar_run.command)
                    sidecars_to_restart.append(sidecar_index)
                case RestartVerdict.fail:
                    _LOGGER.info("Sidecar %s failed", sidecar_run.command)
                    return ProcessManagerPollStatus.fail, []
                case RestartVerdict.skip:
                    pass
                case RestartVerdict.unknown:
                    _LOGGER.warning("Unknown restart policy for %s", sidecar_run.command)
                case _:
                    _LOGGER.warning("Unknown restart verdict for %s", sidecar_run.command)
        return ProcessManagerPollStatus.running, sidecars_to_restart

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
                    fd=meta.output_type.value,
                    datetime=datetime.datetime.now(),
                )
                for handler in meta.log_handlers:
                    handler.process(record)

    def _stop(self) -> None:
        for worker_index, worker_run in self._worker_runs.items():
            _LOGGER.debug("Stopping worker %s: %s", worker_index, worker_run.config.command)
            worker_run.terminate()
            _LOGGER.debug("Worker %s stopped", worker_index)
        for sidecar_index, sidecar_run_meta in self._sidecar_runs.items():
            _LOGGER.debug("Stopping sidecar %s: %s", sidecar_index, sidecar_run_meta.sidecar_run.command)
            sidecar_run_meta.sidecar_run.terminate()
            _LOGGER.debug("Sidecar %s stopped", sidecar_index)
        _LOGGER.debug("Stopping _io_selector")
        self._io_selector.close()
        _LOGGER.debug("_io_selector stopped")
        for handler in self._log_handlers:
            _LOGGER.debug("Stopping log handler %s", handler)
            handler.stop()
            _LOGGER.debug("Log handler %s stopped", handler)


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
