import yt.wrapper as yt

import typing as tp

from copy import deepcopy

import sys

from pathlib import Path

from coordinator import Coordinator
from job_client import JobClient

import torch.distributed as dist


def run(f: tp.Callable, path: str, peer_count: int, client: yt.YtClient = None) -> None:
    yt.create("map_node", path, attributes={"epoch_id": -1}, ignore_existing=True)
    yt.create("map_node", path + "/primary_lock", ignore_existing=True)
    yt.create("map_node", path + "/epochs", ignore_existing=True)

    print(yt.config.get_config(client))
    c = yt.YtClient(config=deepcopy(yt.config.get_config(client)))
    def wrapped() -> None:
        import os
        import socket

        port = int(os.environ['YT_PORT_0'])
        self_endpoint = socket.gethostname() + ":" + str(port)
        coordinator = Coordinator(c, path, peer_count, self_endpoint)
        coordinator.prepare()

        dist.init_process_group(
            backend='gloo',
            init_method='tcp://' + coordinator.get_primary_endpoint(),
            rank=coordinator.get_self_index(),
            world_size=coordinator._peer_count,
        )

        job_client = JobClient(coordinator)
        f(job_client)

    def _module_filter(module):
        if not hasattr(module, '__file__'):
            return False

        # This is really bad.
        system_paths = [Path(p) for p in sys.path[2:]]
        for path in system_paths:
            if path in Path(module.__file__).parents:
                return False

        return True

    yt.update_config({
        "pickling": {
            #"python_binary": "/opt/conda/bin/python3.11",
            "force_using_py_instead_of_pyc": True,
            "module_filter": _module_filter,
        },
    })

    op = yt.run_operation(
        yt.VanillaSpecBuilder()
            .begin_task("task")
                .command(wrapped)
                .job_count(peer_count)
                .port_count(1)
                .docker_image("cr.nemax.nebius.cloud/crnf2coti090683j5ssi/gritukan_ml:5")
                .environment({"YT_ALLOW_HTTP_REQUESTS_TO_YT_FROM_JOB": "1"})
            .end_task()
    )
