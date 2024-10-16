from typing import (
    Any,
    Callable,
)
import warnings

import attrs
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
from tractorun.private.constants import DEFAULT_CLUSTER_CONFIG_PATH as _DEFAULT_CLUSTER_CONFIG_PATH
from tractorun.private.helpers import get_default_docker_image as _get_default_docker_image
from tractorun.private.run_internal import Runnable
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


__all__ = ["run", "prepare_and_get_toolbox"]


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class TractorunParams:
    runnable: Runnable
    docker_image: str
    yt_path: str
    mesh: Mesh
    proxy_stderr_mode: StderrMode
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


def run(
    user_function: Callable,
    *,
    backend: BackendBase,
    yt_path: str,
    mesh: Mesh,
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
    docker_auth: DockerAuthData | None = None,
    attach_external_libs: bool = False,
    dry_run: bool = False,
) -> RunInfo:
    sidecars = sidecars or []
    binds_local = binds_local or []
    binds_local_lib = binds_local_lib or []
    binds_cypress = binds_cypress or []
    env = env or []
    user_config = user_config or {}
    resources = resources or Resources()
    yt_operation_spec = yt_operation_spec or {}
    yt_task_spec = yt_task_spec or {}

    if attach_external_libs:
        warnings.warn("Use attach_external_libs=True only in adhoc scripts. Don't use in production.")
    if docker_image is None:
        docker_image = _get_default_docker_image()
        if docker_image is None:
            raise TractorunConfigurationError("docker_image should be specified")
    params = TractorunParams(
        runnable=_UserFunction(
            function=user_function,
            backend=backend,
        ),
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
        docker_auth=docker_auth,
        attach_external_libs=attach_external_libs,
        dry_run=dry_run,
    )
    if local:
        return _run_local(params=params)
    else:
        return _run_tracto(params=params)


def prepare_and_get_toolbox(backend: BackendBase) -> Toolbox:
    return _prepare_and_get_toolbox(backend)
