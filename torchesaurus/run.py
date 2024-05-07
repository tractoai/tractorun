import yt.wrapper as yt

import typing as tp

from copy import deepcopy

import torch.multiprocessing

import sys

from pathlib import Path

from .coordinator import Coordinator
from .job_client import JobClient
from .mesh import Mesh


def wrapped_run_mp(i, f, c, path) -> None:
    import os
    import json
    import socket

    with open(f"config_{i}.json", 'r') as ff:
        config = json.load(ff)

    port = int(config['port'])
    self_endpoint = socket.gethostname() + ":" + str(port)
    mesh = Mesh(int(config['nnodes']), int(config['nproc']), int(config['ngpu_per_proc']))
    coordinator = Coordinator(c, path, self_endpoint, mesh, int(config['node_index']), int(config['proc_index']))
    #TOOD: coordinator should be with prerequisites
    job_client = JobClient(coordinator, c)
    job_client.initialize()

    f(job_client)


def run(f: tp.Callable, path: str, mesh: Mesh, client: yt.YtClient = None) -> None:
    yt.create("map_node", path, attributes={"epoch_id": -1}, ignore_existing=True)
    yt.create("map_node", path + "/primary_lock", ignore_existing=True)
    yt.create("map_node", path + "/epochs", ignore_existing=True)

    print(yt.config.get_config(client))
    c = yt.YtClient(config=deepcopy(yt.config.get_config(client)))

    def wrapped_bootstrap(mesh: Mesh) -> None:
        import os
        import sys
        import json
        import subprocess

        processes = []

        for i in range(mesh.process_per_node):
            proc_config = {}
            proc_config['nnodes'] = mesh.node_count
            proc_config['nproc'] = mesh.process_per_node
            proc_config['ngpu_per_proc'] = mesh.gpu_per_process
            proc_config['node_index'] = os.environ['YT_JOB_COOKIE']
            proc_config['proc_index'] = i
            proc_config['port'] = os.environ[f'YT_PORT_{i}']
            with open(f'config_{i}.json', 'w') as ff:
                json.dump(proc_config, ff)

            command = ['python3'] + list(sys.argv)
            process = subprocess.Popen(
                command,
                stdout=sys.stderr,
                stderr=sys.stderr,
                bufsize=1,
                universal_newlines=True,
                env={**os.environ, 'TRACTO_CONFIG': f'config_{i}.json'},
            )
            processes.append(process)

        for process in processes:   
            exit_code = process.wait()
            if exit_code != 0:
                sys.exit(exit_code)

        # TODO: torch multiprocessing is better, but pickling does not work.
        #torch.multiprocessing.spawn(wrapped_run_mp, nprocs=mesh.process_per_node, args=(f, c, path,), join=True)


    def wrapped_run() -> None:
        import os
        import json
        import socket

        config_path = os.environ['TRACTO_CONFIG']
        with open(config_path, 'r') as ff:
            config = json.load(ff)

        port = int(config['port'])
        self_endpoint = socket.gethostname() + ":" + str(port)
        mesh = Mesh(int(config['nnodes']), int(config['nproc']), int(config['ngpu_per_proc']))
        coordinator = Coordinator(c, path, self_endpoint, mesh, int(config['node_index']), int(config['proc_index']))
        #TOOD: coordinator should be with prerequisites
        job_client = JobClient(coordinator, c)
        job_client.initialize()

        f(job_client)


    def wrapped() -> None:
        import os
        if 'TRACTO_CONFIG' in os.environ:
            wrapped_run()
        else:
            wrapped_bootstrap(mesh)


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
                .job_count(mesh.node_count)
                .gpu_limit(mesh.gpu_per_process * mesh.process_per_node)
                .port_count(mesh.process_per_node)
                .memory_limit(10*(1024**3))
                .docker_image("cr.nemax.nebius.cloud/crnf2coti090683j5ssi/gritukan_ml:6")
                .environment({"YT_ALLOW_HTTP_REQUESTS_TO_YT_FROM_JOB": "1"})
            .end_task()
    )
