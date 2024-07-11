import base64
import pickle
import sys
from typing import (
    Dict,
    List,
)

import attrs
from yt.common import update_inplace

from tractorun.constants import TRACTO_CONFIG_ENV_VAR
from tractorun.helpers import AttrSerializer
from tractorun.mesh import Mesh


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
    path: str
    yt_client_config: str
    command: List[str]


def bootstrap(mesh: Mesh, path: str, yt_client_config: Dict, command: List[str]) -> None:
    # Runs in a job

    import json
    import os
    import subprocess

    processes = []

    for i in range(mesh.process_per_node):
        update_inplace(
            yt_client_config,
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
            yt_client_config=base64.b64encode(pickle.dumps(yt_client_config)).decode("utf-8"),
        )
        with open(f"config_{i}.json", "w") as f:
            serializer = AttrSerializer(ProcConfig)
            json.dump(serializer.serialize(proc_config), f)

        process = subprocess.Popen(
            command,
            stdout=sys.stderr,
            stderr=sys.stderr,
            bufsize=1,
            universal_newlines=True,
            env={
                **os.environ,
                TRACTO_CONFIG_ENV_VAR: f"config_{i}.json",
                "YT_PROXY": yt_client_config["proxy"]["url"],
                "YT_TOKEN": yt_client_config["token"],
            },
        )
        processes.append(process)

    for process in processes:
        exit_code = process.wait()
        if exit_code != 0:
            sys.exit(exit_code)

    # TODO: torch multiprocessing is better, but pickling does not work.
    # torch.multiprocessing.spawn(wrapped_run_mp, nprocs=mesh.process_per_node, args=(f, c, path,), join=True)
