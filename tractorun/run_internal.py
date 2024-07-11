import abc
import base64
import json
import os
import pickle
import random
import shutil
import sys
import tarfile
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

import attrs
from yt import wrapper as yt
from yt.common import update_inplace

from tractorun import constants as const
from tractorun.base_backend import BackendBase
from tractorun.bind import Bind
from tractorun.closet import get_closet
from tractorun.environment import get_toolbox
from tractorun.exceptions import TractorunInvalidConfiguration
from tractorun.mesh import (
    Mesh,
    MeshSerializer,
)
from tractorun.resources import Resources
from tractorun.toolbox import Toolbox


class Runnable(abc.ABC):
    @abc.abstractmethod
    def modify_task(self, task: yt.TaskSpecBuilder) -> yt.TaskSpecBuilder:
        pass

    @abc.abstractmethod
    def modify_operation(self, operation: yt.VanillaSpecBuilder) -> yt.VanillaSpecBuilder:
        pass

    @abc.abstractmethod
    def get_wrapped_job_function(
        self,
        mesh: Mesh,
        yt_path: str,
        yt_client: yt.YtClient,
    ) -> Callable:
        pass


@attrs.define
class Command(Runnable):
    command: List[str]

    def modify_task(self, task: yt.TaskSpecBuilder) -> yt.TaskSpecBuilder:
        return task

    def modify_operation(self, operation: yt.VanillaSpecBuilder) -> yt.VanillaSpecBuilder:
        return operation

    def get_wrapped_job_function(
        self,
        mesh: Mesh,
        yt_path: str,
        yt_client: yt.YtClient,
    ) -> Callable:
        def wrapped() -> None:
            _bootstrap(mesh, yt_path, yt_client, self.command)

        return wrapped


@attrs.define
class UserFunction(Runnable):
    function: Callable
    _backend: BackendBase

    def modify_task(self, task: yt.TaskSpecBuilder) -> yt.TaskSpecBuilder:
        return task

    def modify_operation(self, operation: yt.VanillaSpecBuilder) -> yt.VanillaSpecBuilder:
        return operation

    def get_wrapped_job_function(
        self,
        mesh: Mesh,
        yt_path: str,
        yt_client: yt.YtClient,
    ) -> Callable:
        def wrapped() -> None:
            if "TRACTO_CONFIG" in os.environ:
                toolbox = _prepare_and_get_toolbox(backend=self._backend)
                self.function(toolbox)
            else:
                command = ["python3"] + sys.argv
                _bootstrap(
                    mesh=mesh,
                    path=yt_path,
                    yt_client=yt_client,
                    command=command,
                )

        return wrapped


def _run_tracto(
    runnable: Runnable,
    *,
    docker_image: str,
    yt_path: str,
    mesh: Mesh,
    user_config: Optional[Dict[Any, Any]] = None,
    binds: Optional[List[Bind]] = None,
    resources: Optional[Resources] = None,
    yt_client: Optional[yt.YtClient] = None,
    wandb_enabled: bool = False,
    wandb_api_key: Optional[str] = None,
    yt_operation_spec: Optional[Dict[Any, Any]] = None,
    yt_task_spec: Optional[Dict[Any, Any]] = None,
) -> None:
    resources = resources if resources is not None else Resources()
    binds = binds if binds is not None else []
    yt_operation_spec = yt_operation_spec if yt_operation_spec is not None else {}
    yt_task_spec = yt_task_spec if yt_task_spec is not None else {}

    # if mesh.node_count > 1 and mesh.gpu_per_process * mesh.process_per_node not in (0, 8):
    #     raise exc.TractorunInvalidConfiguration("gpu per node can only be 0 or 8")

    yt_client = yt_client or yt.YtClient(config=yt.default_config.get_config_from_env())
    yt_client.config["pickling"]["ignore_system_modules"] = True

    wrapped = runnable.get_wrapped_job_function(mesh=mesh, yt_path=yt_path, yt_client=yt_client)

    # TODO: do it normally
    if os.path.exists(".binds"):
        shutil.rmtree(".binds")
    os.mkdir(".binds")

    task_spec = yt.VanillaSpecBuilder().begin_task("task")

    for idx, bind in enumerate(binds):
        path = f".binds/{idx}.tar"
        with tarfile.open(path, "w:gz") as tar:
            tar.add(bind.source, arcname=os.path.basename(bind.source))
        task_spec = task_spec.file_paths(yt.LocalFile(path, path))

    def unpack_wrapper(func: Callable) -> Callable:
        def wrapper() -> None:
            for idx_w, bind_w in enumerate(binds):
                path_w = f".binds/{idx_w}.tar"
                print(f"Extract {path_w} to {bind_w.destination}", file=sys.stderr)
                with tarfile.open(path_w, "r:gz") as tar_w:
                    tar_w.extractall(path=bind_w.destination)
                print(f"Extracted {os.listdir(bind_w.destination)}", file=sys.stderr)
            func()

        return wrapper

    task_spec = runnable.modify_task(
        task_spec.command(unpack_wrapper(wrapped))
        .job_count(mesh.node_count)
        .gpu_limit(mesh.gpu_per_process * mesh.process_per_node)
        .port_count(mesh.process_per_node)
        .cpu_limit(resources.cpu_limit)
        .memory_limit(resources.memory_limit)
        .docker_image(docker_image)
        .spec(yt_task_spec)
        .environment(
            {
                "YT_ALLOW_HTTP_REQUESTS_TO_YT_FROM_JOB": "1",
                const.YT_USER_CONFIG_ENV_VAR: json.dumps(user_config),
                "WANDB_ENABLED": str(int(wandb_enabled)),
                # Sometimes we can't read compiled bytecode in forks on yt.
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
    )

    operation_spec = task_spec.end_task()

    if mesh.pool_trees is not None:
        operation_spec = operation_spec.pool_trees(mesh.pool_trees)

    if wandb_enabled:
        operation_spec = operation_spec.secure_vault(
            {
                "WANDB_API_KEY": wandb_api_key,
            }
        )

    operation_spec = operation_spec.spec(yt_operation_spec)
    operation_spec = runnable.modify_operation(operation_spec)

    yt_client.run_operation(operation_spec)


def _run_local(
    runnable: Runnable,
    *,
    yt_path: str,
    mesh: Mesh,
    yt_client: Optional[yt.YtClient] = None,
    wandb_enabled: bool = False,
    wandb_api_key: Optional[str] = None,
) -> None:
    if mesh.node_count != 1:
        raise TractorunInvalidConfiguration("local mode only supports 1 node")

    # TODO: respawn in docker

    # Fake YT job environment.
    os.environ["YT_OPERATION_ID"] = "1-2-3-4"
    os.environ["YT_JOB_ID"] = "a-b-c-d"
    os.environ["YT_JOB_COOKIE"] = "0"
    os.environ["YT_ALLOW_HTTP_REQUESTS_TO_YT_FROM_JOB"] = "1"
    os.environ["WANDB_ENABLED"] = str(int(wandb_enabled))
    if wandb_api_key:
        os.environ["YT_SECURE_VAULT_WANDB_API_KEY"] = wandb_api_key

    # TODO: look for free ports

    start_port = random.randint(10000, 20000)
    for i in range(mesh.process_per_node):
        os.environ[f"YT_PORT_{i}"] = str(start_port + i)

    wrapped = runnable.get_wrapped_job_function(mesh=mesh, yt_path=yt_path, yt_client=yt_client)
    return wrapped()


def _bootstrap(mesh: Mesh, path: str, yt_client: yt.YtClient, command: List[str]) -> None:
    # Runs in a job

    import json
    import os
    import subprocess

    processes = []

    for i in range(mesh.process_per_node):
        proc_config: Dict[str, Union[str, int]] = {
            "mesh": MeshSerializer.serialize(mesh),
            "node_index": os.environ["YT_JOB_COOKIE"],
            "proc_index": i,
            "port": os.environ[f"YT_PORT_{i}"],
            "path": path,
        }

        conf = yt.config.get_config(yt_client)
        update_inplace(
            conf,
            {
                "pickling": {
                    "module_filter": None,
                },
            },
        )

        proc_config["yt_client_config"] = base64.b64encode(pickle.dumps(conf)).decode()
        with open(f"config_{i}.json", "w") as f:
            json.dump(proc_config, f)

        process = subprocess.Popen(
            command,
            stdout=sys.stderr,
            stderr=sys.stderr,
            bufsize=1,
            universal_newlines=True,
            env={
                **os.environ,
                "TRACTO_CONFIG": f"config_{i}.json",
                "YT_PROXY": conf["proxy"]["url"],
                "YT_TOKEN": conf["token"],
            },
        )
        processes.append(process)

    for process in processes:
        exit_code = process.wait()
        if exit_code != 0:
            sys.exit(exit_code)

    # TODO: torch multiprocessing is better, but pickling does not work.
    # torch.multiprocessing.spawn(wrapped_run_mp, nprocs=mesh.process_per_node, args=(f, c, path,), join=True)


def _prepare_and_get_toolbox(backend: BackendBase) -> Toolbox:
    # Runs in a job
    closet = get_closet()
    backend.environment.prepare(closet)
    return get_toolbox(closet)
