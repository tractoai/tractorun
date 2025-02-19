from typing import (
    Any,
    Callable,
)
import warnings

import yt.wrapper as yt

from tractorun.base_backend import BackendBase
from tractorun.bind import (
    BindCypress,
    BindLocal,
)
from tractorun.docker_auth import DockerAuthData
from tractorun.env import EnvVariable
from tractorun.exception import TractorunConfigurationError
from tractorun.mesh import Mesh
from tractorun.operation_log import OperationLogMode
from tractorun.private.constants import DEFAULT_CLUSTER_CONFIG_PATH as _DEFAULT_CLUSTER_CONFIG_PATH
from tractorun.private.helpers import get_default_docker_image as _get_default_docker_image
from tractorun.private.logging import setup_logging as _setup_logging
from tractorun.private.run_internal import CliCommand as _CliCommand
from tractorun.private.run_internal import TractorunParams as _TractorunParams
from tractorun.private.run_internal import UserFunction as _UserFunction
from tractorun.private.run_internal import prepare_and_get_toolbox as _prepare_and_get_toolbox
from tractorun.private.run_internal import run_local as _run_local
from tractorun.private.run_internal import run_tracto as _run_tracto
from tractorun.resources import Resources
from tractorun.run_info import RunInfo
from tractorun.sidecar import Sidecar
from tractorun.stderr_reader import StderrMode
from tractorun.tensorproxy import TensorproxySidecar
from tractorun.toolbox import Toolbox


__all__ = ["run", "run_script", "prepare_and_get_toolbox"]


def run_script(
    user_command: list[str],
    *,
    mesh: Mesh,
    yt_path: str | None = None,
    title: str | None = None,
    user_config: dict[Any, Any] | None = None,
    cluster_config_path: str = _DEFAULT_CLUSTER_CONFIG_PATH,
    docker_image: str | None = None,
    resources: Resources | None = None,
    yt_client: yt.YtClient | None = None,
    binds_local: list[BindLocal] | None = None,
    binds_local_lib: list[str] | None = None,
    binds_cypress: list[BindCypress] | None = None,
    sidecars: list[Sidecar] | None = None,
    tensorproxy: TensorproxySidecar | None = None,
    env: list[EnvVariable] | None = None,
    no_wait: bool = False,
    yt_operation_spec: dict[Any, Any] | None = None,
    yt_task_spec: dict[Any, Any] | None = None,
    local: bool = False,
    proxy_stderr_mode: StderrMode = StderrMode.disabled,
    operation_log_mode: OperationLogMode = OperationLogMode.default,
    docker_auth: DockerAuthData | None = None,
    dry_run: bool = False,
) -> RunInfo:
    log_level = _setup_logging()

    docker_image = _get_docker_image(docker_image)

    sidecars = sidecars or []
    binds_local = binds_local or []
    binds_local_lib = binds_local_lib or []
    binds_cypress = binds_cypress or []
    env = env or []
    user_config = user_config or {}
    resources = resources or Resources()
    mesh = mesh or Mesh()
    yt_operation_spec = yt_operation_spec or {}
    yt_task_spec = yt_task_spec or {}

    params = _TractorunParams(
        runnable=_CliCommand(command=user_command),
        yt_path=yt_path,
        mesh=mesh,
        title=title,
        binds_local=binds_local,
        binds_local_lib=binds_local_lib,
        binds_cypress=binds_cypress,
        sidecars=sidecars,
        tensorproxy=tensorproxy,
        env=env,
        no_wait=no_wait,
        user_config=user_config,
        cluster_config_path=cluster_config_path,
        resources=resources,
        yt_client=yt_client,
        docker_image=docker_image,
        yt_operation_spec=yt_operation_spec,
        yt_task_spec=yt_task_spec,
        proxy_stderr_mode=proxy_stderr_mode,
        operation_log_mode=operation_log_mode,
        docker_auth=docker_auth,
        dry_run=dry_run,
        attach_external_libs=False,
        log_level=log_level,
    )
    if local:
        return _run_local(params=params)
    else:
        return _run_tracto(params=params)


def run(
    user_command: Callable,
    *,
    backend: BackendBase,
    yt_path: str | None = None,
    mesh: Mesh | None = None,
    title: str | None = None,
    user_config: dict[Any, Any] | None = None,
    cluster_config_path: str = _DEFAULT_CLUSTER_CONFIG_PATH,
    docker_image: str | None = None,
    resources: Resources | None = None,
    yt_client: yt.YtClient | None = None,
    binds_local: list[BindLocal] | None = None,
    binds_local_lib: list[str] | None = None,
    binds_cypress: list[BindCypress] | None = None,
    sidecars: list[Sidecar] | None = None,
    tensorproxy: TensorproxySidecar | None = None,
    env: list[EnvVariable] | None = None,
    no_wait: bool = False,
    yt_operation_spec: dict[Any, Any] | None = None,
    yt_task_spec: dict[Any, Any] | None = None,
    local: bool = False,
    proxy_stderr_mode: StderrMode = StderrMode.disabled,
    operation_log_mode: OperationLogMode = OperationLogMode.default,
    docker_auth: DockerAuthData | None = None,
    attach_external_libs: bool = False,
    dry_run: bool = False,
) -> RunInfo:
    log_level = _setup_logging()

    if attach_external_libs:
        warnings.warn("Use attach_external_libs=True only in adhoc scripts. Don't use it in production.")
    docker_image = _get_docker_image(docker_image)

    sidecars = sidecars or []
    binds_local = binds_local or []
    binds_local_lib = binds_local_lib or []
    binds_cypress = binds_cypress or []
    env = env or []
    user_config = user_config or {}
    mesh = mesh or Mesh()
    resources = resources or Resources()
    yt_operation_spec = yt_operation_spec or {}
    yt_task_spec = yt_task_spec or {}

    params = _TractorunParams(
        runnable=_UserFunction(function=user_command, backend=backend),
        yt_path=yt_path,
        mesh=mesh,
        title=title,
        binds_local=binds_local,
        binds_local_lib=binds_local_lib,
        binds_cypress=binds_cypress,
        sidecars=sidecars,
        tensorproxy=tensorproxy,
        env=env,
        no_wait=no_wait,
        user_config=user_config,
        cluster_config_path=cluster_config_path,
        resources=resources,
        yt_client=yt_client,
        docker_image=docker_image,
        yt_operation_spec=yt_operation_spec,
        yt_task_spec=yt_task_spec,
        proxy_stderr_mode=proxy_stderr_mode,
        operation_log_mode=operation_log_mode,
        docker_auth=docker_auth,
        attach_external_libs=attach_external_libs,
        dry_run=dry_run,
        log_level=log_level,
    )
    if local:
        return _run_local(params=params)
    else:
        return _run_tracto(params=params)


def _get_docker_image(docker_image: str | None) -> str:
    if docker_image is not None:
        return docker_image
    docker_image = _get_default_docker_image()
    if docker_image is None:
        raise TractorunConfigurationError("docker_image should be specified")
    return docker_image


def prepare_and_get_toolbox(backend: BackendBase) -> Toolbox:
    return _prepare_and_get_toolbox(backend)
