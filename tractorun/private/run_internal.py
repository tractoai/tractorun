import abc
import base64
import copy
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
from yt.wrapper import TaskSpecBuilder

from tractorun import __version__
from tractorun.base_backend import BackendBase
from tractorun.bind import BindLocal
from tractorun.docker_auth import DockerAuthData
from tractorun.env import EnvVariable
from tractorun.exception import TractorunConfigurationError
from tractorun.mesh import Mesh
from tractorun.private import constants as const
from tractorun.private.bind import (
    BindsLibPacker,
    BindsPacker,
)
from tractorun.private.bootstrapper import (
    BootstrapConfig,
    LibVersions,
    bootstrap,
)
from tractorun.private.closet import get_closet
from tractorun.private.constants import (
    BIND_PATHS_ENV_VAR,
    BOOTSTRAP_CONFIG_FILENAME_ENV_VAR,
    BOOTSTRAP_CONFIG_NAME,
)
from tractorun.private.coordinator import get_incarnation_id
from tractorun.private.docker_auth import DockerAuthDataExtractor
from tractorun.private.environment import get_toolbox
from tractorun.private.helpers import AttrSerializer
from tractorun.private.stderr_reader import StderrReaderWorker
from tractorun.private.tensorproxy import (
    TensorproxyBootstrap,
    TensorproxyConfigurator,
)
from tractorun.private.training_dir import (
    TrainingDir,
    prepare_training_dir,
)
from tractorun.resources import Resources
from tractorun.run_info import (
    LocalRunInfo,
    YtRunInfo,
)
from tractorun.sidecar import Sidecar
from tractorun.stderr_reader import StderrMode
from tractorun.tensorproxy import TensorproxySidecar
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
        sidecars: list[Sidecar],
        env: list[EnvVariable],
        training_dir: TrainingDir,
        yt_client_config: str,
        tensorproxy: Optional[TensorproxyBootstrap],
        lib_versions: LibVersions,
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
        return f"python3 -m tractorun.cli.tractorun_bootstrap {escaped_command}".encode("utf-8")

    def make_local_command(
        self,
        mesh: Mesh,
        sidecars: list[Sidecar],
        env: list[EnvVariable],
        training_dir: TrainingDir,
        yt_client_config: str,
        tensorproxy: Optional[TensorproxyBootstrap],
        lib_versions: LibVersions,
    ) -> Callable:
        def wrapped() -> None:
            bootstrap(
                mesh=mesh,
                training_dir=training_dir,
                yt_client_config=yt_client_config,
                command=self.get_bootstrap_command(),
                sidecars=sidecars,
                env=env,
                tensorproxy=tensorproxy,
                lib_versions=lib_versions,
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
                toolbox = prepare_and_get_toolbox(backend=self._backend)
                self.function(toolbox)
            else:
                binds_packer = BindsPacker.from_env(os.environ[BIND_PATHS_ENV_VAR])
                binds_packer.unpack()
                bootstrap_config_path = os.environ[BOOTSTRAP_CONFIG_FILENAME_ENV_VAR]
                with open(bootstrap_config_path, "r") as f:
                    content = f.read()
                    deserializer = AttrSerializer(BootstrapConfig)
                    config: BootstrapConfig = deserializer.deserialize(data=content)
                bootstrap(
                    mesh=config.mesh,
                    training_dir=config.training_dir,
                    yt_client_config=config.yt_client_config,
                    command=self.get_bootstrap_command(),
                    sidecars=config.sidecars,
                    env=config.env,
                    tensorproxy=config.tensorproxy,
                    lib_versions=config.lib_versions,
                )

        return wrapped

    def make_local_command(
        self,
        mesh: Mesh,
        sidecars: list[Sidecar],
        env: list[EnvVariable],
        training_dir: TrainingDir,
        yt_client_config: str,
        tensorproxy: Optional[TensorproxyBootstrap],
        lib_versions: LibVersions,
    ) -> Callable:
        def wrapped() -> None:
            # run on YT
            if "TRACTO_CONFIG" in os.environ:
                toolbox = prepare_and_get_toolbox(backend=self._backend)
                self.function(toolbox)
            else:
                bootstrap(
                    mesh=mesh,
                    training_dir=training_dir,
                    yt_client_config=yt_client_config,
                    command=self.get_bootstrap_command(),
                    sidecars=sidecars,
                    env=env,
                    tensorproxy=tensorproxy,
                    lib_versions=lib_versions,
                )

        return wrapped


def run_tracto(
    runnable: Runnable,
    *,
    docker_image: str,
    yt_path: str,
    mesh: Mesh,
    proxy_stderr_mode: StderrMode,
    user_config: dict[Any, Any] | None = None,
    binds_local: list[BindLocal] | None = None,
    binds_local_lib: list[str] | None = None,
    tensorproxy: TensorproxySidecar | None = None,
    no_wait: bool = False,
    sidecars: list[Sidecar] | None = None,
    env: list[EnvVariable] | None = None,
    resources: Resources | None = None,
    yt_client: yt.YtClient | None = None,
    yt_operation_spec: dict[Any, Any] | None = None,
    yt_task_spec: dict[Any, Any] | None = None,
    docker_auth: DockerAuthData | None = None,
    attach_external_libs: bool = False,
    dry_run: bool = False,
) -> YtRunInfo:
    resources = resources if resources is not None else Resources()
    binds_local = binds_local if binds_local is not None else []
    binds_local_lib = binds_local_lib if binds_local_lib is not None else []
    sidecars = sidecars if sidecars is not None else []
    env = env if env is not None else []
    yt_operation_spec = yt_operation_spec if yt_operation_spec is not None else {}
    yt_task_spec = yt_task_spec if yt_task_spec is not None else {}

    # if mesh.node_count > 1 and mesh.gpu_per_process * mesh.process_per_node not in (0, 8):
    #     raise exc.TractorunInvalidConfiguration("gpu per node can only be 0 or 8")

    yt_client = yt_client or yt.YtClient(config=yt.default_config.get_config_from_env())
    yt_client.config["pickling"]["ignore_system_modules"] = False if attach_external_libs else True

    yt_client_config = yt.config.get_config(yt_client)
    yt_client_config_pickled = base64.b64encode(pickle.dumps(yt_client_config)).decode("utf-8")

    yt_client_config_for_job: dict = copy.deepcopy(yt_client_config)

    # for tests only
    yt_config_for_job_patch_yson_string = os.environ.get("TRACTORUN_YT_CONFIG_FOR_JOB_PATCH")
    if yt_config_for_job_patch_yson_string:
        patch = yt.yson.loads(yt_config_for_job_patch_yson_string.encode())
        yt.common.update_inplace(yt_client_config_for_job, patch)

    yt_client_config_for_job_pickled = base64.b64encode(pickle.dumps(yt_client_config_for_job)).decode("utf-8")

    # detached mode only applies to the local client
    yt_client.config["detached"] = True if no_wait else False

    tmp_dir = tempfile.TemporaryDirectory()
    training_dir = TrainingDir.create(yt_path)

    tp_bootstrap, tp_yt_files, tp_ports = TensorproxyConfigurator(tensorproxy=tensorproxy).generate_configuration()

    # "fail_on_job_restart" is useful for gang operations,
    # so let's turn it on unless disabled explicitly.
    if "fail_on_job_restart" not in yt_operation_spec:
        yt_operation_spec["fail_on_job_restart"] = True

    bootstrap_config = BootstrapConfig(
        mesh=mesh,
        sidecars=sidecars,
        env=env,
        training_dir=training_dir,
        yt_client_config=yt_client_config_for_job_pickled,
        tensorproxy=tp_bootstrap,
        lib_versions=LibVersions.create(),
    )

    bootstrap_config_path = os.path.join(tmp_dir.name, BOOTSTRAP_CONFIG_NAME)
    with open(bootstrap_config_path, "w") as f:
        f.write(AttrSerializer(BootstrapConfig).serialize(bootstrap_config))

    binds_packer = BindsPacker.from_binds(
        binds=binds_local,
    )
    packed_binds = binds_packer.pack(tmp_dir.name)
    bind_libs_packer = BindsLibPacker(
        paths=binds_local_lib,
    )
    packed_libs = bind_libs_packer.pack(tmp_dir.name)
    new_pythonpath = ":".join(["./" + packed_lib.archive_name for packed_lib in packed_libs])

    yt_file_bindings = []
    yt_file_bindings.extend(
        [yt.LocalFile(packed_bind.local_path, packed_bind.yt_path) for packed_bind in packed_binds],
    )
    yt_file_bindings.extend([yt.LocalFile(packed_lib.path, packed_lib.archive_name) for packed_lib in packed_libs])
    yt_file_bindings.append(
        yt.LocalFile(bootstrap_config_path, BOOTSTRAP_CONFIG_NAME),
    )

    yt_command = runnable.make_yt_command()
    task_spec: TaskSpecBuilder = yt.VanillaSpecBuilder().begin_task("task")

    task_spec = task_spec.file_paths(yt_file_bindings + tp_yt_files)

    task_spec = runnable.modify_task(
        task_spec.command(yt_command)
        .job_count(mesh.node_count)
        .gpu_limit(mesh.gpu_per_process * mesh.process_per_node)
        .port_count(mesh.process_per_node + tp_ports)
        .cpu_limit(resources.cpu_limit)
        .memory_limit(resources.memory_limit)
        .docker_image(docker_image)
        .spec(yt_task_spec)
        .environment(
            {
                "YT_ALLOW_HTTP_REQUESTS_TO_YT_FROM_JOB": "1",
                const.YT_USER_CONFIG_ENV_VAR: json.dumps(user_config),
                # Sometimes we can't read compiled bytecode in forks on yt.
                "PYTHONDONTWRITEBYTECODE": "1",
                BIND_PATHS_ENV_VAR: binds_packer.to_env(),
                "PYTHONPATH": f"$PYTHONPATH:{new_pythonpath}" if new_pythonpath else "$PYTHONPATH",
                BOOTSTRAP_CONFIG_FILENAME_ENV_VAR: BOOTSTRAP_CONFIG_NAME,
            },
        )
    )

    operation_spec = task_spec.end_task()

    if mesh.pool_trees is not None:
        operation_spec = operation_spec.pool_trees(mesh.pool_trees)

    secure_vault: dict[str, Any] = {}
    if docker_auth:
        secure_vault["docker_auth"] = DockerAuthDataExtractor(yt_client=yt_client).extract(docker_auth).to_spec()

    if secure_vault:
        operation_spec.secure_vault(secure_vault)

    operation_spec = operation_spec.spec(yt_operation_spec)
    operation_spec = runnable.modify_operation(operation_spec)

    prev_incarnation_id = get_incarnation_id(yt_client, training_dir)

    operation_id = None
    is_sync = not no_wait
    if not dry_run:
        prepare_training_dir(yt_client=yt_client, training_dir=training_dir)
        with StderrReaderWorker(
            prev_incarnation_id=prev_incarnation_id,
            training_dir=training_dir,
            yt_client_config_pickled=yt_client_config_pickled,
            mode=proxy_stderr_mode,
            mesh=mesh,
        ):
            operation = yt_client.run_operation(operation_spec, sync=is_sync)
            assert isinstance(operation, yt.Operation)
            operation_id = operation.id

    run_info = YtRunInfo(
        operation_spec=operation_spec.build(client=yt_client),
        operation_id=operation_id,
        operation_attributes=operation.get_attributes(),
    )

    tmp_dir.cleanup()

    return run_info


def run_local(
    runnable: Runnable,
    *,
    yt_path: str,
    mesh: Mesh,
    sidecars: Optional[list[Sidecar]] = None,
    env: Optional[list[EnvVariable]] = None,
    tensorproxy: Optional[TensorproxySidecar] = None,
    yt_client: Optional[yt.YtClient] = None,
    dry_run: bool = False,
) -> LocalRunInfo:
    sidecars = sidecars if sidecars is not None else []

    if mesh.node_count != 1:
        raise TractorunConfigurationError("local mode only supports 1 node")

    yt_client = yt_client or yt.YtClient(config=yt.default_config.get_config_from_env())

    # TODO: respawn in docker
    # Fake YT job environment.
    os.environ["YT_OPERATION_ID"] = "1-2-3-4"
    os.environ["YT_JOB_ID"] = "a-b-c-d"
    os.environ["YT_JOB_COOKIE"] = "0"
    os.environ["YT_ALLOW_HTTP_REQUESTS_TO_YT_FROM_JOB"] = "1"

    # TODO: look for free ports
    start_port = random.randint(10000, 20000)
    for i in range(mesh.process_per_node):
        os.environ[f"YT_PORT_{i}"] = str(start_port + i)

    training_dir = TrainingDir.create(yt_path)
    tp_bootstrap, _, _ = TensorproxyConfigurator(tensorproxy=tensorproxy).generate_configuration()

    wrapped = runnable.make_local_command(
        mesh=mesh,
        sidecars=sidecars,
        env=env or [],
        training_dir=training_dir,
        yt_client_config=base64.b64encode(pickle.dumps(yt_client.config)).decode("utf-8"),
        tensorproxy=tp_bootstrap,
        lib_versions=LibVersions(
            tractorun=__version__,
            ytsaurus_client=yt.__version__,
        ),
    )
    if not dry_run:
        prepare_training_dir(training_dir, yt_client)
        wrapped()
    return LocalRunInfo()


def prepare_and_get_toolbox(backend: BackendBase) -> Toolbox:
    # Runs in a job
    closet = get_closet()
    backend.environment.prepare(closet)
    return get_toolbox(closet)
