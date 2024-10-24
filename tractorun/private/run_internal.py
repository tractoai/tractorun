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
import cattrs.errors
from yt import wrapper as yt
from yt.wrapper import TaskSpecBuilder
import yt.yson as yson

from tractorun import __version__
from tractorun.base_backend import BackendBase
from tractorun.bind import (
    BindCypress,
    BindLocal,
)
from tractorun.docker_auth import DockerAuthData
from tractorun.env import EnvVariable
from tractorun.exception import (
    TractorunConfigurationError,
    TractorunVersionMismatchError,
)
from tractorun.mesh import Mesh
from tractorun.operation_log import OperationLogMode
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
from tractorun.private.helpers import (
    AttrSerializer,
    create_attrs_converter,
)
from tractorun.private.stderr_reader import StderrReaderWorker
from tractorun.private.tensorproxy import (
    TensorproxyBootstrap,
    TensorproxyConfigurator,
)
from tractorun.private.training_dir import (
    TrainingDir,
    prepare_training_dir,
)
from tractorun.private.yt_cluster import TractorunClusterConfig
from tractorun.resources import Resources
from tractorun.run_info import RunInfo
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
        cluster_config: TractorunClusterConfig,
        operation_log_mode: OperationLogMode,
    ) -> Callable:
        pass


@attrs.define
class CliCommand(Runnable):
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
        cluster_config: TractorunClusterConfig,
        operation_log_mode: OperationLogMode,
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
                cluster_config=cluster_config,
                operation_log_mode=operation_log_mode,
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
                    deserializer = AttrSerializer(
                        BootstrapConfig,
                        # forward compatibility
                        converter=create_attrs_converter(forbid_extra_keys=False),
                    )
                    try:
                        config: BootstrapConfig = deserializer.deserialize(data=content)
                    except cattrs.errors.BaseValidationError as e:
                        raise TractorunVersionMismatchError(
                            "Please check that the tractorun version locally and on YT are the same",
                        ) from e
                bootstrap(
                    mesh=config.mesh,
                    training_dir=config.training_dir,
                    yt_client_config=config.yt_client_config,
                    command=self.get_bootstrap_command(),
                    sidecars=config.sidecars,
                    env=config.env,
                    tensorproxy=config.tensorproxy,
                    lib_versions=config.lib_versions,
                    cluster_config=config.cluster_config,
                    operation_log_mode=config.operation_log_mode,
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
        cluster_config: TractorunClusterConfig,
        operation_log_mode: OperationLogMode,
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
                    cluster_config=cluster_config,
                    operation_log_mode=operation_log_mode,
                )

        return wrapped


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class TractorunParams:
    runnable: Runnable
    docker_image: str
    yt_path: str
    mesh: Mesh
    proxy_stderr_mode: StderrMode
    operation_log_mode: OperationLogMode
    cluster_config_path: str
    title: str | None = None
    user_config: dict[Any, Any]
    binds_local: list[BindLocal]
    binds_local_lib: list[str]
    binds_cypress: list[BindCypress]
    tensorproxy: TensorproxySidecar | None
    no_wait: bool
    sidecars: list[Sidecar]
    env: list[EnvVariable]
    resources: Resources
    yt_client: yt.YtClient | None
    yt_operation_spec: dict[Any, Any]
    yt_task_spec: dict[Any, Any]
    docker_auth: DockerAuthData | None
    attach_external_libs: bool
    dry_run: bool


def run_tracto(params: TractorunParams) -> RunInfo:
    # if mesh.node_count > 1 and mesh.gpu_per_process * mesh.process_per_node not in (0, 8):
    #     raise exc.TractorunInvalidConfiguration("gpu per node can only be 0 or 8")

    yt_client = params.yt_client or yt.YtClient(config=yt.default_config.get_config_from_env())
    yt_client.config["pickling"]["ignore_system_modules"] = False if params.attach_external_libs else True

    # we store it explicitly since locally it could have been read from ~/.yt/token
    yt_client.config["token"] = yt.http_helpers.get_token(client=yt_client) or ""

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
    yt_client.config["detached"] = True if params.no_wait else False

    tmp_dir = tempfile.TemporaryDirectory()
    training_dir = TrainingDir.create(params.yt_path)

    tp_bootstrap, tp_yt_files, tp_ports = TensorproxyConfigurator(
        tensorproxy=params.tensorproxy
    ).generate_configuration()

    cluster_config = TractorunClusterConfig.load_from_yt(yt_client=yt_client, path=params.cluster_config_path)

    bootstrap_config = BootstrapConfig(
        mesh=params.mesh,
        sidecars=params.sidecars,
        env=params.env,
        training_dir=training_dir,
        yt_client_config=yt_client_config_for_job_pickled,
        tensorproxy=tp_bootstrap,
        lib_versions=LibVersions.create(),
        cluster_config=cluster_config,
        operation_log_mode=params.operation_log_mode,
    )

    bootstrap_config_path = os.path.join(tmp_dir.name, BOOTSTRAP_CONFIG_NAME)
    with open(bootstrap_config_path, "w") as f:
        f.write(AttrSerializer(BootstrapConfig).serialize(bootstrap_config))

    binds_packer = BindsPacker.from_binds(
        binds=params.binds_local,
    )
    packed_binds = binds_packer.pack(tmp_dir.name)
    bind_libs_packer = BindsLibPacker(
        paths=params.binds_local_lib,
    )
    packed_libs = bind_libs_packer.pack(tmp_dir.name)
    new_pythonpath = ":".join(["./" + packed_lib.archive_name for packed_lib in packed_libs])

    yt_file_bindings = []
    yt_file_bindings.extend(
        [yt.LocalFile(packed_bind.local_path, packed_bind.yt_path) for packed_bind in packed_binds],
    )

    yt_file_bindings.extend(
        [yson.to_yson_type(cb.source, attributes={"file_name": cb.destination}) for cb in params.binds_cypress]
    )

    yt_file_bindings.extend([yt.LocalFile(packed_lib.path, packed_lib.archive_name) for packed_lib in packed_libs])
    yt_file_bindings.append(
        yt.LocalFile(bootstrap_config_path, BOOTSTRAP_CONFIG_NAME),
    )

    yt_command = params.runnable.make_yt_command()

    # prepare task spec
    task_spec: TaskSpecBuilder = yt.VanillaSpecBuilder().begin_task("task")

    task_spec = task_spec.file_paths(yt_file_bindings + tp_yt_files)

    task_spec = params.runnable.modify_task(
        task_spec.command(yt_command)
        .job_count(params.mesh.node_count)
        .gpu_limit(params.mesh.gpu_per_process * params.mesh.process_per_node)
        .port_count(params.mesh.process_per_node + tp_ports)
        .cpu_limit(params.resources.cpu_limit)
        .memory_limit(params.resources.memory_limit)
        .docker_image(params.docker_image)
        .spec(params.yt_task_spec)
        .environment(
            {
                "YT_ALLOW_HTTP_REQUESTS_TO_YT_FROM_JOB": "1",
                const.YT_USER_CONFIG_ENV_VAR: json.dumps(params.user_config),
                # Sometimes we can't read compiled bytecode in forks on yt.
                "PYTHONDONTWRITEBYTECODE": "1",
                BIND_PATHS_ENV_VAR: binds_packer.to_env(),
                "PYTHONPATH": f"$PYTHONPATH:{new_pythonpath}" if new_pythonpath else "$PYTHONPATH",
                BOOTSTRAP_CONFIG_FILENAME_ENV_VAR: BOOTSTRAP_CONFIG_NAME,
            },
        )
    )

    # prepare operation spec
    operation_spec = task_spec.end_task()
    additional_operation_spec = copy.deepcopy(params.yt_operation_spec)

    operation_spec = operation_spec.title(params.title)

    if params.mesh.pool_trees is not None:
        operation_spec = operation_spec.pool_trees(params.mesh.pool_trees)
    if params.mesh.pool is not None:
        operation_spec = operation_spec.pool(params.mesh.pool)

    secure_vault: dict[str, Any] = {}
    if params.docker_auth:
        secure_vault["docker_auth"] = DockerAuthDataExtractor(yt_client=yt_client).extract(params.docker_auth).to_spec()
        operation_spec.secure_vault(secure_vault)

    # "fail_on_job_restart" is useful for gang operations,
    # so let's turn it on unless disabled explicitly.
    if "fail_on_job_restart" not in additional_operation_spec:
        additional_operation_spec["fail_on_job_restart"] = True

    if "is_gang" not in additional_operation_spec:
        additional_operation_spec["is_gang"] = True

    additional_operation_spec["annotations"] = additional_operation_spec.get("annotations", {})
    additional_operation_spec["annotations"]["is_tractorun"] = True

    # save job stderr for 150 jobs
    # we can't make it bigger because 150 is a scheduler's limit
    operation_spec.max_stderr_count(150)

    operation_spec = operation_spec.spec(additional_operation_spec)
    operation_spec = params.runnable.modify_operation(operation_spec)

    prev_incarnation_id = get_incarnation_id(yt_client, training_dir)

    operation_id = None
    is_sync = not params.no_wait
    if not params.dry_run:
        prepare_training_dir(yt_client=yt_client, training_dir=training_dir)
        with StderrReaderWorker(
            prev_incarnation_id=prev_incarnation_id,
            training_dir=training_dir,
            yt_client_config_pickled=yt_client_config_pickled,
            mode=params.proxy_stderr_mode,
            mesh=params.mesh,
        ):
            operation = yt_client.run_operation(operation_spec, sync=is_sync)
            assert isinstance(operation, yt.Operation)
            operation_id = operation.id

    run_info = RunInfo(
        operation_spec=operation_spec.build(client=yt_client),
        operation_id=operation_id,
    )

    tmp_dir.cleanup()

    return run_info


def run_local(
    params: TractorunParams,
) -> RunInfo:
    if params.mesh.node_count != 1:
        raise TractorunConfigurationError("local mode only supports 1 node")

    yt_client = params.yt_client or yt.YtClient(config=yt.default_config.get_config_from_env())

    # TODO: respawn in docker
    # Fake YT job environment.
    os.environ["YT_OPERATION_ID"] = "1-2-3-4"
    os.environ["YT_JOB_ID"] = "a-b-c-d"
    os.environ["YT_JOB_COOKIE"] = "0"
    os.environ["YT_ALLOW_HTTP_REQUESTS_TO_YT_FROM_JOB"] = "1"

    cluster_config = TractorunClusterConfig.load_from_yt(yt_client=yt_client, path=params.cluster_config_path)

    # TODO: look for free ports
    start_port = random.randint(10000, 20000)
    for i in range(params.mesh.process_per_node):
        os.environ[f"YT_PORT_{i}"] = str(start_port + i)

    training_dir = TrainingDir.create(params.yt_path)
    tp_bootstrap, _, _ = TensorproxyConfigurator(tensorproxy=params.tensorproxy).generate_configuration()

    wrapped = params.runnable.make_local_command(
        mesh=params.mesh,
        sidecars=params.sidecars,
        env=params.env or [],
        training_dir=training_dir,
        yt_client_config=base64.b64encode(pickle.dumps(yt_client.config)).decode("utf-8"),
        tensorproxy=tp_bootstrap,
        lib_versions=LibVersions(
            tractorun=__version__,
            ytsaurus_client=yt.__version__,
        ),
        cluster_config=cluster_config,
        operation_log_mode=params.operation_log_mode,
    )
    if not params.dry_run:
        prepare_training_dir(training_dir, yt_client)
        wrapped()
    return RunInfo(operation_spec={}, operation_id=None)


def prepare_and_get_toolbox(backend: BackendBase) -> Toolbox:
    # Runs in a job
    closet = get_closet()
    backend.environment.prepare(closet)
    return get_toolbox(closet)
