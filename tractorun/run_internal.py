import abc
import base64
import json
import os
import pickle
import random
import shlex
import sys
import tempfile
from typing import (
    Any,
    Callable,
    Optional,
)

import attrs
from yt import wrapper as yt

from tractorun import constants as const
from tractorun.base_backend import BackendBase
from tractorun.bind import (
    Bind,
    BindsPacker,
)
from tractorun.bootstrapper import (
    BOOTSTRAP_CONFIG_YT_PATH,
    BootstrapConfig,
    bootstrap,
)
from tractorun.closet import get_closet
from tractorun.constants import (
    BIND_PATHS_ENV_VAR,
    BOOTSTRAP_CONFIG_FILENAME_ENV_VAR,
)
from tractorun.environment import get_toolbox
from tractorun.exceptions import TractorunInvalidConfiguration
from tractorun.helpers import AttrSerializer
from tractorun.mesh import Mesh
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
    def get_bootstrap_command(self) -> list[str]:
        pass

    @abc.abstractmethod
    def make_yt_command(self) -> Callable | bytes:
        pass

    @abc.abstractmethod
    def make_local_command(
        self,
        mesh: Mesh,
        yt_path: str,
        yt_client_config: str,
    ) -> Callable:
        pass


@attrs.define
class Command(Runnable):
    command: list[str]

    def modify_task(self, task: yt.TaskSpecBuilder) -> yt.TaskSpecBuilder:
        return task

    def modify_operation(self, operation: yt.VanillaSpecBuilder) -> yt.VanillaSpecBuilder:
        return operation

    def get_bootstrap_command(self) -> list[str]:
        return self.command

    def make_yt_command(self) -> bytes:
        escaped_command = " ".join([shlex.quote(arg) for arg in self.command])
        return f"_tractorun_bootstrap {escaped_command}".encode("utf-8")

    def make_local_command(
        self,
        mesh: Mesh,
        yt_path: str,
        yt_client_config: str,
    ) -> Callable:
        def wrapped() -> None:
            bootstrap(
                mesh=mesh,
                path=yt_path,
                yt_client_config=yt_client_config,
                command=self.get_bootstrap_command(),
            )

        return wrapped


@attrs.define
class UserFunction(Runnable):
    function: Callable
    _backend: BackendBase

    def modify_task(self, task: yt.TaskSpecBuilder) -> yt.TaskSpecBuilder:
        return task

    def modify_operation(self, operation: yt.VanillaSpecBuilder) -> yt.VanillaSpecBuilder:
        return operation

    def get_bootstrap_command(self) -> list[str]:
        # on YT it should be native YT-wrapper command like
        # python3 _py_runner.py wrapped.pickle config_dump _modules_info _main_module.py _main_module PY_SOURCE
        return ["python3"] + sys.argv

    def make_yt_command(self) -> Callable:
        def wrapped() -> None:
            # run on YT
            if "TRACTO_CONFIG" in os.environ:
                toolbox = _prepare_and_get_toolbox(backend=self._backend)
                self.function(toolbox)
            else:
                binds_packer = BindsPacker.from_env(os.environ[BIND_PATHS_ENV_VAR])
                binds_packer.unpack()
                bootstrap_config_path = os.path.join(
                    BOOTSTRAP_CONFIG_YT_PATH,
                    os.environ[BOOTSTRAP_CONFIG_FILENAME_ENV_VAR],
                )
                with open(bootstrap_config_path, "r") as f:
                    content = f.read()
                    deserializer = AttrSerializer(BootstrapConfig)
                    config: BootstrapConfig = deserializer.deserialize(data=content)
                bootstrap(
                    mesh=config.mesh,
                    path=config.path,
                    yt_client_config=config.yt_client_config,
                    command=self.get_bootstrap_command(),
                )

        return wrapped

    def make_local_command(
        self,
        mesh: Mesh,
        yt_path: str,
        yt_client_config: str,
    ) -> Callable:
        def wrapped() -> None:
            # run on YT
            if "TRACTO_CONFIG" in os.environ:
                toolbox = _prepare_and_get_toolbox(backend=self._backend)
                self.function(toolbox)
            else:
                bootstrap(
                    mesh=mesh,
                    path=yt_path,
                    yt_client_config=yt_client_config,
                    command=self.get_bootstrap_command(),
                )

        return wrapped


def _run_tracto(
    runnable: Runnable,
    *,
    docker_image: str,
    yt_path: str,
    mesh: Mesh,
    user_config: Optional[dict[Any, Any]] = None,
    binds: Optional[list[Bind]] = None,
    resources: Optional[Resources] = None,
    yt_client: Optional[yt.YtClient] = None,
    wandb_enabled: bool = False,
    wandb_api_key: Optional[str] = None,
    yt_operation_spec: Optional[dict[Any, Any]] = None,
    yt_task_spec: Optional[dict[Any, Any]] = None,
) -> None:
    resources = resources if resources is not None else Resources()
    binds = binds if binds is not None else []
    yt_operation_spec = yt_operation_spec if yt_operation_spec is not None else {}
    yt_task_spec = yt_task_spec if yt_task_spec is not None else {}

    # if mesh.node_count > 1 and mesh.gpu_per_process * mesh.process_per_node not in (0, 8):
    #     raise exc.TractorunInvalidConfiguration("gpu per node can only be 0 or 8")

    yt_client = yt_client or yt.YtClient(config=yt.default_config.get_config_from_env())
    yt_client.config["pickling"]["ignore_system_modules"] = True

    yt_client_config: dict = yt.config.get_config(yt_client)

    yt_command = runnable.make_yt_command()

    task_spec = yt.VanillaSpecBuilder().begin_task("task")

    config = BootstrapConfig(
        mesh=mesh,
        path=yt_path,
        yt_client_config=base64.b64encode(pickle.dumps(yt_client_config)).decode("utf-8"),
    )
    tmp_file = tempfile.NamedTemporaryFile()
    tmp_file.write(AttrSerializer(BootstrapConfig).serialize(config).encode("utf-8"))

    binds_packer = BindsPacker(
        binds=binds + [Bind(source=tmp_file.name, destination=BOOTSTRAP_CONFIG_YT_PATH)],
    )
    bind_paths = binds_packer.pack()
    task_spec = task_spec.file_paths([yt.LocalFile(path, path) for path in bind_paths])

    task_spec = runnable.modify_task(
        task_spec.command(yt_command)
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
                BIND_PATHS_ENV_VAR: binds_packer.to_env(),
                BOOTSTRAP_CONFIG_FILENAME_ENV_VAR: os.path.split(tmp_file.name)[1],
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
    tmp_file.close()


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

    yt_client = yt_client or yt.YtClient(config=yt.default_config.get_config_from_env())

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

    wrapped = runnable.make_local_command(
        mesh=mesh,
        yt_path=yt_path,
        yt_client_config=base64.b64encode(pickle.dumps(yt_client.config)).decode("utf-8"),
    )
    return wrapped()


def _prepare_and_get_toolbox(backend: BackendBase) -> Toolbox:
    # Runs in a job
    closet = get_closet()
    backend.environment.prepare(closet)
    return get_toolbox(closet)
