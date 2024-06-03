import argparse
import base64
import pickle  # TODO: kill with fire!
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
)

import yt.wrapper as yt
from yt.wrapper.common import update_inplace

from tractorun.checkpoints import CheckpointManager
from tractorun.coordinator import Coordinator
from tractorun.job_client import JobClient
from tractorun.mesh import Mesh
from tractorun.resources import Resources


DEFAULT_DOCKER_IMAGE = "cr.ai.nebius.cloud/crnf2coti090683j5ssi/tractorun/torchesaurus_runtime:2024-05-31-12-26-50"


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


def get_job_client(user_config: Dict[Any, Any]) -> JobClient:
    # Runs in a job
    import json
    import os
    import socket

    config_path = os.environ["TRACTO_CONFIG"]
    with open(config_path, "r") as ff:
        config = json.load(ff)

    port = int(config["port"])
    path = config["path"]
    self_endpoint = socket.gethostname() + ":" + str(port)
    mesh = Mesh(int(config["nnodes"]), int(config["nproc"]), int(config["ngpu_per_proc"]))
    yt_cli = yt.YtClient(config=pickle.loads(base64.b64decode(config["yt_client_config"])))
    coordinator = Coordinator(
        yt_cli=yt_cli,
        path=path,
        self_endpoint=self_endpoint,
        mesh=mesh,
        node_index=int(config["node_index"]),
        process_index=int(config["proc_index"]),
    )
    checkpoint_manager = CheckpointManager(path + "/checkpoints", yt_cli)
    # TOOD: coordinator should be with prerequisites
    job_client = JobClient(coordinator, checkpoint_manager, yt_cli, user_config=user_config)
    job_client.initialize()

    ep = coordinator.get_primary_endpoint()
    os.environ["MASTER_ADDR"] = ep.split(":")[0]
    os.environ["MASTER_PORT"] = ep.split(":")[1]
    os.environ["WORLD_SIZE"] = str(coordinator.get_total_peer_count())
    os.environ["NODE_RANK"] = str(coordinator.get_self_index() // mesh.process_per_node)
    os.environ["LOCAL_RANK"] = str(coordinator.get_self_index() % mesh.process_per_node)

    return job_client


def bootstrap(mesh: Mesh, path: str, yt_cli: yt.YtClient, pyargs: Optional[list] = None) -> None:
    # Runs in a job

    import json
    import os
    import subprocess
    import sys

    processes = []

    for i in range(mesh.process_per_node):
        from typing import (
            Dict,
            Union,
        )

        proc_config: Dict[str, Union[str, int]] = {
            "nnodes": mesh.node_count,
            "nproc": mesh.process_per_node,
            "ngpu_per_proc": mesh.gpu_per_process,
            "node_index": os.environ["YT_JOB_COOKIE"],
            "proc_index": i,
            "port": os.environ[f"YT_PORT_{i}"],
            "path": path,
        }

        conf = yt.config.get_config(yt_cli)
        update_inplace(
            conf,
            {
                "pickling": {
                    "module_filter": None,
                },
            },
        )

        proc_config["yt_client_config"] = base64.b64encode(pickle.dumps(conf)).decode()
        with open(f"config_{i}.json", "w") as ff:
            json.dump(proc_config, ff)

        if pyargs:
            command = ["python3"] + pyargs
        else:
            command = ["python3"] + list(sys.argv)

        process = subprocess.Popen(
            command,
            stdout=sys.stderr,
            stderr=sys.stderr,
            bufsize=1,
            universal_newlines=True,
            env={**os.environ, "TRACTO_CONFIG": f"config_{i}.json", "NCCL_DEBUG": "TRACE", "NCCL_SHM_DISABLE": "1"},
        )
        processes.append(process)

    for process in processes:
        exit_code = process.wait()
        if exit_code != 0:
            sys.exit(exit_code)

    # TODO: torch multiprocessing is better, but pickling does not work.
    # torch.multiprocessing.spawn(wrapped_run_mp, nprocs=mesh.process_per_node, args=(f, c, path,), join=True)


def run(
    user_function: Callable,
    path: str,
    mesh: Mesh,
    user_config: Optional[Dict[Any, Any]] = None,
    resources: Optional[Resources] = None,
    yt_cli: Optional[yt.YtClient] = None,
    docker_image: Optional[str] = None,
) -> None:
    docker_image = docker_image or DEFAULT_DOCKER_IMAGE
    resources = resources if resources is not None else Resources()
    user_config = user_config or {}

    yt_cli: yt.YtClient = yt_cli or yt.YtClient()
    yt_cli.config["pickling"]["ignore_system_modules"] = True

    yt_cli.create("map_node", path, attributes={"epoch_id": -1}, ignore_existing=True)
    yt_cli.create("map_node", path + "/primary_lock", ignore_existing=True)
    yt_cli.create("map_node", path + "/epochs", ignore_existing=True)

    def wrapped() -> None:
        import os

        if "TRACTO_CONFIG" in os.environ:
            job_client = get_job_client(user_config=user_config)
            user_function(job_client)
        else:
            bootstrap(mesh, path, yt_cli)

    # antiaffinity! =)
    cpu_limit = resources.cpu_limit or 50
    memory_limit = resources.memory_limit or 300 * (1024**3)

    yt_cli.run_operation(
        yt.VanillaSpecBuilder()
        .begin_task("task")
        .command(wrapped)
        .job_count(mesh.node_count)
        .gpu_limit(mesh.gpu_per_process * mesh.process_per_node)
        .port_count(mesh.process_per_node)
        .cpu_limit(cpu_limit)
        .memory_limit(memory_limit)
        .docker_image(docker_image)
        .environment({"YT_ALLOW_HTTP_REQUESTS_TO_YT_FROM_JOB": "1"})
        .end_task()
    )


def run_script(args: argparse.Namespace, script_name: str) -> None:
    # we shouldn't use argparse.Namespace here
    # Pickling fix.

    yt_cli = yt.YtClient()

    def wrapped() -> None:
        mesh = Mesh(args.nnodes, args.nproc_per_node, args.ngpu_per_proc)
        return bootstrap(mesh, args.path, yt_cli=yt_cli, pyargs=[script_name])

    # TODO: parse from args.
    resources = Resources()
    cpu_limit = resources.cpu_limit or 50
    memory_limit = resources.memory_limit or 300 * (1024**3)

    yt_cli.run_operation(
        yt.VanillaSpecBuilder()
        .begin_task("task")
        .command(wrapped)
        .job_count(args.nnodes)
        .gpu_limit(args.nproc_per_node * args.ngpu_per_proc)
        .port_count(args.nproc_per_node)
        .cpu_limit(cpu_limit)
        .memory_limit(memory_limit)
        .docker_image(DEFAULT_DOCKER_IMAGE)
        .environment({"YT_ALLOW_HTTP_REQUESTS_TO_YT_FROM_JOB": "1"})
        .file_paths(yt.LocalFile(args.training_script, file_name=script_name))
        .end_task()
        .max_failed_job_count(1)
    )
