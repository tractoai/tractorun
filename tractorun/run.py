from typing import (
    Any,
    Callable,
    Optional,
)

import yt.wrapper as yt

from tractorun.base_backend import BackendBase
from tractorun.bind import Bind
from tractorun.constants import DEFAULT_DOCKER_IMAGE
from tractorun.mesh import Mesh
from tractorun.resources import Resources
from tractorun.run_internal import (
    Command,
    UserFunction,
    _prepare_and_get_toolbox,
    _run_local,
    _run_tracto,
)
from tractorun.sidecar import Sidecar
from tractorun.toolbox import Toolbox


def run(
    user_function: Callable,
    *,
    backend: BackendBase,
    yt_path: str,
    mesh: Mesh,
    user_config: Optional[dict[Any, Any]] = None,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    resources: Optional[Resources] = None,
    yt_client: Optional[yt.YtClient] = None,
    sidecars: Optional[list[Sidecar]] = None,
    wandb_enabled: bool = False,
    wandb_api_key: Optional[str] = None,
    yt_operation_spec: Optional[dict[Any, Any]] = None,
    yt_task_spec: Optional[dict[Any, Any]] = None,
    local: bool = False,
) -> None:
    if local:
        return _run_local(
            UserFunction(
                function=user_function,
                backend=backend,
            ),
            yt_path=yt_path,
            mesh=mesh,
            sidecars=sidecars,
            yt_client=yt_client,
            wandb_enabled=wandb_enabled,
            wandb_api_key=wandb_api_key,
        )
    else:
        return _run_tracto(
            UserFunction(
                function=user_function,
                backend=backend,
            ),
            yt_path=yt_path,
            mesh=mesh,
            sidecars=sidecars,
            user_config=user_config,
            resources=resources,
            yt_client=yt_client,
            docker_image=docker_image,
            wandb_enabled=wandb_enabled,
            wandb_api_key=wandb_api_key,
            yt_operation_spec=yt_operation_spec,
            yt_task_spec=yt_task_spec,
        )


def run_script(
    command: list[str],
    *,
    yt_path: str,
    mesh: Mesh,
    docker_image: str,
    resources: Resources,
    user_config: Optional[dict[Any, Any]] = None,
    binds: Optional[list[Bind]] = None,
    bind_libs: Optional[list[str]] = None,
    sidecars: Optional[list[Sidecar]] = None,
    local: bool = False,
    yt_operation_spec: Optional[dict[Any, Any]] = None,
    yt_task_spec: Optional[dict[Any, Any]] = None,
) -> None:
    if binds is None:
        binds = []
    if local:
        return _run_local(
            runnable=Command(command=command),
            yt_path=yt_path,
            mesh=mesh,
            sidecars=sidecars,
            yt_client=None,
        )
    else:
        return _run_tracto(
            runnable=Command(command=command),
            yt_path=yt_path,
            mesh=mesh,
            user_config=user_config,
            resources=resources,
            yt_client=None,
            docker_image=docker_image,
            binds=binds,
            bind_libs=bind_libs,
            sidecars=sidecars,
            yt_operation_spec=yt_operation_spec,
            yt_task_spec=yt_task_spec,
        )


def prepare_and_get_toolbox(backend: BackendBase) -> Toolbox:
    return _prepare_and_get_toolbox(backend)
