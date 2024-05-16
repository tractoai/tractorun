# TODO: kill with fire!
import pickle
import base64
import sys
import typing as tp
from pathlib import Path
from copy import deepcopy

import yt.wrapper as yt
from yt.wrapper.common import update_inplace

from .coordinator import Coordinator
from .checkpoints import CheckpointManager
from .job_client import JobClient
from .mesh import Mesh
from .resources import Resources


""" def wrapped_run_mp(i, f, c, path) -> None:
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

    f(job_client) """


def initialize() -> JobClient:
    import os
    import json
    import socket

    config_path = os.environ['TRACTO_CONFIG']
    with open(config_path, 'r') as ff:
        config = json.load(ff)

    port = int(config['port'])
    path = config['path']
    self_endpoint = socket.gethostname() + ":" + str(port)
    mesh = Mesh(int(config['nnodes']), int(config['nproc']), int(config['ngpu_per_proc']))
    c = yt.YtClient(config=pickle.loads(base64.b64decode(config['yt_client_config'])))
    coordinator = Coordinator(c, path, self_endpoint, mesh, int(config['node_index']), int(config['proc_index']))
    checkpoint_manager = CheckpointManager(path + "/checkpoints", c)
    #TOOD: coordinator should be with prerequisites
    job_client = JobClient(coordinator, checkpoint_manager, c)
    job_client.initialize()

    ep = coordinator.get_primary_endpoint()
    os.environ['MASTER_ADDR'] = ep.split(':')[0]
    os.environ['MASTER_PORT'] = ep.split(':')[1]
    os.environ['WORLD_SIZE'] = str(coordinator.get_total_peer_count())
    os.environ['NODE_RANK'] = str(coordinator.get_self_index() // mesh.process_per_node)
    os.environ['LOCAL_RANK'] = str(coordinator.get_self_index() % mesh.process_per_node)

    return job_client


def bootstrap(mesh: Mesh, path: str, c: yt.YtClient, pyargs = None) -> None:
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
        proc_config['path'] = path

        conf = yt.config.get_config(c)
        update_inplace(conf, {
            "pickling": {
                "module_filter": None,
            },
        })

        proc_config['yt_client_config'] = base64.b64encode(pickle.dumps(conf)).decode()
        with open(f'config_{i}.json', 'w') as ff:
            json.dump(proc_config, ff)

        if pyargs:
            command = ['python3'] + pyargs
        else:
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
    # torch.multiprocessing.spawn(wrapped_run_mp, nprocs=mesh.process_per_node, args=(f, c, path,), join=True)


def fix_module_import() -> None:
    def _module_filter(module):
        if not hasattr(module, '__file__'):
            return False
        
        if "torchesaurus" in str(module.__file__):
            return True

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


def run(f: tp.Callable, path: str, mesh: Mesh, resources: Resources = Resources(), client: yt.YtClient = None) -> None:
    yt.create("map_node", path, attributes={"epoch_id": -1}, ignore_existing=True)
    yt.create("map_node", path + "/primary_lock", ignore_existing=True)
    yt.create("map_node", path + "/epochs", ignore_existing=True)

    c = yt.YtClient(config=deepcopy(yt.config.get_config(client)))

    def wrapped() -> None:
        import os
        if 'TRACTO_CONFIG' in os.environ:
            job_client = initialize()
            f(job_client)
        else:
            bootstrap(mesh, path, c)


    # antiaffinity! =)
    cpu_limit = resources.cpu_limit or 150
    memory_limit = resources.memory_limit or 300*(1024**3)

    fix_module_import()

    op = yt.run_operation(
        yt.VanillaSpecBuilder()
            .begin_task("task")
                .command(wrapped)
                .job_count(mesh.node_count)
                .gpu_limit(mesh.gpu_per_process * mesh.process_per_node)
                .port_count(mesh.process_per_node)
                .cpu_limit(cpu_limit)
                .memory_limit(memory_limit)
                .docker_image("cr.nemax.nebius.cloud/crnf2coti090683j5ssi/gritukan_ml:7")
                .environment({"YT_ALLOW_HTTP_REQUESTS_TO_YT_FROM_JOB": "1"})
            .end_task()
    )


def run_script(args, script_name):
    # Pickling fix.
    import torchesaurus.dataset

    def wrapped() -> None:
        mesh = Mesh(args.nnodes, args.nproc_per_node, args.ngpu_per_proc)
        return bootstrap(mesh, args.path, c=None, pyargs=[script_name])
    
    fix_module_import()

    # TODO: parse from args.
    resources = Resources()
    cpu_limit = resources.cpu_limit or 150
    memory_limit = resources.memory_limit or 300*(1024**3)

    op = yt.run_operation(
        yt.VanillaSpecBuilder()
            .begin_task("task")
                .command(wrapped)
                .job_count(args.nnodes)
                .gpu_limit(args.nproc_per_node * args.ngpu_per_proc)
                .port_count(args.nproc_per_node)
                .cpu_limit(cpu_limit)
                .memory_limit(memory_limit)
                .docker_image("cr.nemax.nebius.cloud/crnf2coti090683j5ssi/gritukan_ml:7")
                .environment({"YT_ALLOW_HTTP_REQUESTS_TO_YT_FROM_JOB": "1"})
                .file_paths(yt.LocalFile(args.training_script, file_name=script_name))
            .end_task()
            .max_failed_job_count(1)
    )
