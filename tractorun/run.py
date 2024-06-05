import base64
import json
import os
import pickle  # TODO: kill with fire!
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Union,
)

import yt.wrapper as yt
from yt.wrapper.common import update_inplace

from tractorun import constants as const
from tractorun.backend.tractorch.environment import prepare_environment
from tractorun.closet import get_closet
from tractorun.environment import get_toolbox
from tractorun.mesh import Mesh
from tractorun.resources import Resources
from tractorun.toolbox import Toolbox


DEFAULT_DOCKER_IMAGE = "cr.ai.nebius.cloud/crnf2coti090683j5ssi/tractorun/torchesaurus_runtime:2024-05-31-12-26-50"


def _bootstrap(mesh: Mesh, path: str, yt_cli: yt.YtClient, pyargs: Optional[list] = None) -> None:
    # Runs in a job

    import json
    import os
    import subprocess
    import sys

    processes = []

    for i in range(mesh.process_per_node):
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
            env={
                **os.environ,
                "TRACTO_CONFIG": f"config_{i}.json",
                "NCCL_DEBUG": "TRACE",
                "NCCL_SHM_DISABLE": "1",
            },
        )
        processes.append(process)

    for process in processes:
        exit_code = process.wait()
        if exit_code != 0:
            sys.exit(exit_code)

    # TODO: torch multiprocessing is better, but pickling does not work.
    # torch.multiprocessing.spawn(wrapped_run_mp, nprocs=mesh.process_per_node, args=(f, c, path,), join=True)


def prepare_and_get_toolbox() -> Toolbox:
    closet = get_closet()
    prepare_environment(closet)
    return get_toolbox(closet)


def run(
    user_function: Callable,
    yt_path: str,
    mesh: Mesh,
    user_config: Optional[Dict[Any, Any]] = None,
    resources: Optional[Resources] = None,
    yt_cli: Optional[yt.YtClient] = None,
    docker_image: Optional[str] = None,
) -> None:
    docker_image = docker_image or DEFAULT_DOCKER_IMAGE
    resources = resources if resources is not None else Resources()

    yt_cli = yt_cli or yt.YtClient(config=yt.default_config.get_config_from_env())
    yt_cli.config["pickling"]["ignore_system_modules"] = True

    yt_cli.create("map_node", yt_path, attributes={"incarnation_id": -1}, ignore_existing=True)
    yt_cli.create("map_node", yt_path + "/primary_lock", ignore_existing=True)
    yt_cli.create("map_node", yt_path + "/incarnations", ignore_existing=True)

    def wrapped() -> None:
        if "TRACTO_CONFIG" in os.environ:
            toolbox = prepare_and_get_toolbox()
            user_function(toolbox)
        else:
            _bootstrap(mesh, yt_path, yt_cli)

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
        .environment(
            {
                "YT_ALLOW_HTTP_REQUESTS_TO_YT_FROM_JOB": "1",
                const.YT_USER_CONFIG_ENV_VAR: json.dumps(user_config),
            }
        )
        .end_task()
    )


def run_script(
    mesh: Mesh,
    training_script: str,
    yt_path: str,
    docker_image: Optional[str] = None,
    user_config: Optional[Dict[Any, Any]] = None,
) -> None:
    docker_image = docker_image or DEFAULT_DOCKER_IMAGE
    user_config = user_config or {}

    yt_cli = yt.YtClient(config=yt.default_config.get_config_from_env())
    yt_cli.config["pickling"]["ignore_system_modules"] = True
    script_name = training_script.split("/")[-1]

    def wrapped() -> None:
        _bootstrap(mesh, yt_path, yt_cli=yt_cli, pyargs=[script_name])

    # TODO: parse from args.
    resources = Resources()
    cpu_limit = resources.cpu_limit or 50
    memory_limit = resources.memory_limit or 300 * (1024**3)

    yt_cli.run_operation(
        yt.VanillaSpecBuilder()
        .begin_task("task")
        .command(wrapped)
        .job_count(mesh.node_count)
        .gpu_limit(mesh.process_per_node * mesh.gpu_per_process)
        .port_count(mesh.process_per_node)
        .cpu_limit(cpu_limit)
        .memory_limit(memory_limit)
        .docker_image(docker_image)
        .environment(
            {
                "YT_ALLOW_HTTP_REQUESTS_TO_YT_FROM_JOB": "1",
                const.YT_USER_CONFIG_ENV_VAR: json.dumps(user_config),
            }
        )
        .file_paths(yt.LocalFile(training_script, file_name=script_name))
        .end_task()
    )
