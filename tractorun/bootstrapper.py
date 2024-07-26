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
    SidecarRunner,
)


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


def bootstrap(mesh: Mesh, path: str, yt_client_config: str, command: list[str], sidecars: list[Sidecar]) -> None:
    # Runs in a job

    processes = []

    yt_config = pickle.loads(base64.b64decode(yt_client_config))

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
            },
        )
        processes.append(process)

    sidecar_runs: list[SidecarRun] = []
    for sidecar in sidecars:
        runner = SidecarRunner(
            command=sidecar.command,
            env={
                **os.environ,
                "YT_PROXY": yt_config["proxy"]["url"],
                "YT_TOKEN": yt_config["token"],
            },
        )
        process = runner.run()
        sidecar_runs.append(
            SidecarRun(
                process=process,
                runner=runner,
                restart_policy=sidecar.restart_policy,
            )
        )

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
