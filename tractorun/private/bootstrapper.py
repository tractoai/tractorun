import base64
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Optional
import warnings

import attrs
import yt.wrapper as yt

from tractorun import __version__
from tractorun.env import EnvVariable
from tractorun.exception import TractorunBootstrapError
from tractorun.mesh import Mesh
from tractorun.operation_log import OperationLogMode
from tractorun.private.logging import setup_logging
from tractorun.private.operation_log import (
    LogHandlerFactory,
    YTLogHandlerFactory,
)
from tractorun.private.process_manager import (
    ProcessManager,
    ProcessManagerPollStatus,
)
from tractorun.private.tensorproxy import TensorproxyBootstrap
from tractorun.private.training_dir import TrainingDir
from tractorun.private.yt_cluster import TractorunClusterConfig
from tractorun.sidecar import Sidecar


PROCESSES_POLL_TIMEOUT = 5


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
class BootstrapConfig:
    mesh: Mesh
    sidecars: list[Sidecar]
    env: list[EnvVariable]
    training_dir: TrainingDir
    yt_client_config: str
    tensorproxy: Optional[TensorproxyBootstrap]
    lib_versions: LibVersions
    cluster_config: TractorunClusterConfig
    operation_log_mode: OperationLogMode


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
    operation_log_mode: OperationLogMode,
    sandbox_path: Path,
) -> None:
    # Runs inside a job

    check_lib_versions(local_lib_versions=lib_versions)
    setup_logging()

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
            sandbox_path=sandbox_path,
        )
        tp_env = tensorproxy.get_environment(grpc_port=tp_grpc_port)

    spec_env = {}
    for var in env:
        if var.cypress_path is not None:
            spec_env[var.name] = yt_client.get(var.cypress_path)
        else:
            spec_env[var.name] = var.value

    sidecars = sidecars + tp_sidecars

    log_handlers: list[LogHandlerFactory] = []
    if operation_log_mode == OperationLogMode.realtime_yt_table:
        log_handlers.append(
            YTLogHandlerFactory(yt_client_config=yt_client_config, training_dir=training_dir),
        )

    status = ProcessManagerPollStatus.running
    with ProcessManager.start(
        command=command,
        sidecars=sidecars,
        training_dir=training_dir,
        yt_client_config=yt_client_config,
        cluster_config=cluster_config,
        mesh=mesh,
        node_index=int(os.environ["YT_JOB_COOKIE"]),
        os_environ=os.environ,
        tp_env=tp_env,
        spec_env=spec_env,
        log_handler_factories=log_handlers,
        sandbox_path=sandbox_path,
    ) as process_manager:
        while status == ProcessManagerPollStatus.running:
            time.sleep(PROCESSES_POLL_TIMEOUT)
            status = process_manager.poll()
    match status:
        case ProcessManagerPollStatus.success:
            return
        case ProcessManagerPollStatus.fail:
            sys.exit(1)
        case _:
            raise TractorunBootstrapError(f"Unknown poll status {status}")
