import base64
import enum
import json
import os
import pickle
import subprocess
import sys
import time
from typing import Optional

import attrs
from yt.common import update_inplace

from tractorun.constants import TRACTO_CONFIG_ENV_VAR
from tractorun.helpers import AttrSerializer
from tractorun.mesh import Mesh
from tractorun.sidecar import (
    RestartVerdict,
    Sidecar,
    SidecarRun,
)
from tractorun.tensorproxy import TensorproxyBootstrap


TIMEOUT = 10


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class ProcConfig:
    mesh: Mesh
    node_index: int
    proc_index: int
    port: int
    path: str
    yt_client_config: str


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class BootstrapConfig:
    mesh: Mesh
    sidecars: list[Sidecar]
    path: str
    yt_client_config: str
    tensorproxy: Optional[TensorproxyBootstrap]


def bootstrap(
    mesh: Mesh,
    path: str,
    yt_client_config: str,
    command: list[str],
    sidecars: list[Sidecar],
    tensorproxy: Optional[TensorproxyBootstrap],
) -> None:
    # Runs in a job

    processes = []
    yt_config = pickle.loads(base64.b64decode(yt_client_config))

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

    sidecars = sidecars + tp_sidecars

    for i in range(mesh.process_per_node):
        update_inplace(
            yt_config,
            {
                "pickling": {
                    "module_filter": None,
                },
            },
        )

        proc_config = ProcConfig(
            mesh=mesh,
            node_index=int(os.environ["YT_JOB_COOKIE"]),
            proc_index=i,
            port=int(os.environ[f"YT_PORT_{i}"]),
            path=path,
            yt_client_config=base64.b64encode(pickle.dumps(yt_config)).decode("utf-8"),
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
            stdout=sys.stderr,
            stderr=sys.stderr,
            bufsize=1,
            universal_newlines=True,
            env={
                **os.environ,
                TRACTO_CONFIG_ENV_VAR: config_name,
                "YT_PROXY": yt_config["proxy"]["url"],
                "YT_TOKEN": yt_config["token"],
                **tp_env,
            },
        )
        processes.append(process)

    sidecar_runs: list[SidecarRun] = []
    for sidecar in sidecars:
        sidecar_run = SidecarRun.run(
            sidecar=sidecar,
            env={
                **os.environ,
                "YT_PROXY": yt_config["proxy"]["url"],
                "YT_TOKEN": yt_config["token"],
            },
        )
        sidecar_runs.append(sidecar_run)

    while True:
        time.sleep(TIMEOUT)
        exit_codes = [process.poll() for process in processes]
        match check_status(exit_codes):
            case PoolStatus.failed:
                sys.exit(1)
            case PoolStatus.success:
                for run in sidecar_runs:
                    run.terminate()
                return

        for run in sidecar_runs:
            match run.need_restart():
                case RestartVerdict.restart:
                    run.restart()
                case RestartVerdict.fail:
                    print("Sidecar has been failed", file=sys.stderr)
                    sys.exit(1)
                case RestartVerdict.skip:
                    pass
                case RestartVerdict.unknown:
                    print("Warning: unknown restart policy", file=sys.stderr)
                    pass
                case _:
                    print("Warning: unknown restart verdict", file=sys.stderr)
                    pass


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
