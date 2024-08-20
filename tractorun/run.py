from typing import (
    Any,
    Callable,
    Optional,
)

import yt.wrapper as yt

from tractorun.base_backend import BackendBase
from tractorun.bind import BindLocal
from tractorun.env import EnvVariable
from tractorun.mesh import Mesh
from tractorun.private.constants import DEFAULT_DOCKER_IMAGE as _DEFAULT_DOCKER_IMAGE
from tractorun.private.run_internal import UserFunction as _UserFunction
from tractorun.private.run_internal import prepare_and_get_toolbox as _prepare_and_get_toolbox
from tractorun.private.run_internal import run_local as _run_local
from tractorun.private.run_internal import run_tracto as _run_tracto
from tractorun.resources import Resources
from tractorun.sidecar import Sidecar
from tractorun.stderr_reader import StderrMode
from tractorun.toolbox import Toolbox


def run(
    user_function: Callable,
    *,
    backend: BackendBase,
    yt_path: str,
    mesh: Mesh,
    user_config: Optional[dict[Any, Any]] = None,
    docker_image: str = _DEFAULT_DOCKER_IMAGE,
    resources: Optional[Resources] = None,
    yt_client: Optional[yt.YtClient] = None,
    binds_local: Optional[list[BindLocal]] = None,
    binds_local_lib: Optional[list[str]] = None,
    sidecars: Optional[list[Sidecar]] = None,
    env: Optional[list[EnvVariable]] = None,
    wandb_enabled: bool = False,
    wandb_api_key: Optional[str] = None,
    yt_operation_spec: Optional[dict[Any, Any]] = None,
    yt_task_spec: Optional[dict[Any, Any]] = None,
    local: bool = False,
    proxy_stderr_mode: StderrMode = StderrMode.disabled,
) -> None:
    if local:
        _run_local(
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
        )
    else:
        _run_tracto(
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
        )


def prepare_and_get_toolbox(backend: BackendBase) -> Toolbox:
    return _prepare_and_get_toolbox(backend)
