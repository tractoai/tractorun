from typing import Any as _Any
from typing import Callable as _Callable
from typing import Optional as _Optional

import yt.wrapper as _yt

from tractorun.base_backend import BackendBase as _BackendBase
from tractorun.bind import BindLocal as _BindLocal
from tractorun.env import EnvVariable as _EnvVariable
from tractorun.mesh import Mesh as _Mesh
from tractorun.private.constants import DEFAULT_DOCKER_IMAGE as _DEFAULT_DOCKER_IMAGE
from tractorun.private.run_internal import UserFunction as _UserFunction
from tractorun.private.run_internal import prepare_and_get_toolbox as _prepare_and_get_toolbox
from tractorun.private.run_internal import run_local as _run_local
from tractorun.private.run_internal import run_tracto as _run_tracto
from tractorun.resources import Resources as _Resources
from tractorun.sidecar import Sidecar as _Sidecar
from tractorun.stderr_reader import StderrMode as _StderrMode
from tractorun.toolbox import Toolbox as _Toolbox


def run(
    user_function: _Callable,
    *,
    backend: _BackendBase,
    yt_path: str,
    mesh: _Mesh,
    user_config: _Optional[dict[_Any, _Any]] = None,
    docker_image: str = _DEFAULT_DOCKER_IMAGE,
    resources: _Optional[_Resources] = None,
    yt_client: _Optional[_yt.YtClient] = None,
    binds_local: _Optional[list[_BindLocal]] = None,
    binds_local_lib: _Optional[list[str]] = None,
    sidecars: _Optional[list[_Sidecar]] = None,
    env: _Optional[list[_EnvVariable]] = None,
    wandb_enabled: bool = False,
    wandb_api_key: _Optional[str] = None,
    yt_operation_spec: _Optional[dict[_Any, _Any]] = None,
    yt_task_spec: _Optional[dict[_Any, _Any]] = None,
    local: bool = False,
    proxy_stderr_mode: _StderrMode = _StderrMode.disabled,
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


def prepare_and_get_toolbox(backend: _BackendBase) -> _Toolbox:
    return _prepare_and_get_toolbox(backend)
