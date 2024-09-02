from typing import (
    Any,
    Callable,
)

from typing_extensions import (
    Literal,
    overload,
)
import yt.wrapper as yt

from tractorun.base_backend import BackendBase
from tractorun.bind import BindLocal
from tractorun.docker_auth import DockerAuthData
from tractorun.env import EnvVariable
from tractorun.mesh import Mesh
from tractorun.private.helpers import get_default_docker_image as _get_default_docker_image
from tractorun.private.run_internal import UserFunction as _UserFunction
from tractorun.private.run_internal import prepare_and_get_toolbox as _prepare_and_get_toolbox
from tractorun.private.run_internal import run_local as _run_local
from tractorun.private.run_internal import run_tracto as _run_tracto
from tractorun.resources import Resources
from tractorun.run_info import (
    LocalRunInfo,
    YtRunInfo,
)
from tractorun.sidecar import Sidecar
from tractorun.stderr_reader import StderrMode
from tractorun.toolbox import Toolbox


__all__ = ["run", "prepare_and_get_toolbox"]


@overload
def run(
    user_function: Callable,
    *,
    backend: BackendBase,
    yt_path: str,
    mesh: Mesh,
    user_config: dict[Any, Any] | None = ...,
    docker_image: str | None = ...,
    resources: Resources | None = ...,
    yt_client: yt.YtClient | None = ...,
    binds_local: list[BindLocal] | None = ...,
    binds_local_lib: list[str] | None = ...,
    sidecars: list[Sidecar] | None = ...,
    env: list[EnvVariable] | None = ...,
    wandb_enabled: bool = ...,
    wandb_api_key: str | None = ...,
    yt_operation_spec: dict[Any, Any] | None = ...,
    yt_task_spec: dict[Any, Any] | None = ...,
    local: Literal[True],
    proxy_stderr_mode: StderrMode = ...,
    docker_auth: DockerAuthData | None = ...,
    dry_run: bool = ...,
) -> LocalRunInfo: ...


@overload
def run(
    user_function: Callable,
    *,
    backend: BackendBase,
    yt_path: str,
    mesh: Mesh,
    user_config: dict[Any, Any] | None = ...,
    docker_image: str | None = ...,
    resources: Resources | None = ...,
    yt_client: yt.YtClient | None = ...,
    binds_local: list[BindLocal] | None = ...,
    binds_local_lib: list[str] | None = ...,
    sidecars: list[Sidecar] | None = ...,
    env: list[EnvVariable] | None = ...,
    wandb_enabled: bool = ...,
    wandb_api_key: str | None = ...,
    yt_operation_spec: dict[Any, Any] | None = ...,
    yt_task_spec: dict[Any, Any] | None = ...,
    local: Literal[False] = False,
    proxy_stderr_mode: StderrMode = ...,
    docker_auth: DockerAuthData | None = ...,
    dry_run: bool = ...,
) -> YtRunInfo: ...


def run(
    user_function: Callable,
    *,
    backend: BackendBase,
    yt_path: str,
    mesh: Mesh,
    user_config: dict[Any, Any] | None = None,
    docker_image: str | None = None,
    resources: Resources | None = None,
    yt_client: yt.YtClient | None = None,
    binds_local: list[BindLocal] | None = None,
    binds_local_lib: list[str] | None = None,
    sidecars: list[Sidecar] | None = None,
    env: list[EnvVariable] | None = None,
    wandb_enabled: bool = False,
    wandb_api_key: str | None = None,
    yt_operation_spec: dict[Any, Any] | None = None,
    yt_task_spec: dict[Any, Any] | None = None,
    local: bool = False,
    proxy_stderr_mode: StderrMode = StderrMode.disabled,
    docker_auth: DockerAuthData | None = None,
    dry_run: bool = False,
) -> YtRunInfo | LocalRunInfo:
    if docker_image is None:
        docker_image = _get_default_docker_image()
    if local:
        return _run_local(
            _UserFunction(
                function=user_function,
                backend=backend,
            ),
            yt_path=yt_path,
            mesh=mesh,
            sidecars=sidecars,
            env=env,
            yt_client=yt_client,
            wandb_enabled=wandb_enabled,
            wandb_api_key=wandb_api_key,
            dry_run=dry_run,
        )
    else:
        return _run_tracto(
            _UserFunction(
                function=user_function,
                backend=backend,
            ),
            yt_path=yt_path,
            mesh=mesh,
            binds_local=binds_local,
            binds_local_lib=binds_local_lib,
            sidecars=sidecars,
            env=env,
            user_config=user_config,
            resources=resources,
            yt_client=yt_client,
            docker_image=docker_image,
            wandb_enabled=wandb_enabled,
            wandb_api_key=wandb_api_key,
            yt_operation_spec=yt_operation_spec,
            yt_task_spec=yt_task_spec,
            proxy_stderr_mode=proxy_stderr_mode,
            docker_auth=docker_auth,
            dry_run=dry_run,
        )


def prepare_and_get_toolbox(backend: BackendBase) -> Toolbox:
    return _prepare_and_get_toolbox(backend)
