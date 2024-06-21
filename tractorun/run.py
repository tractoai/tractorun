from typing import (
    Any,
    Callable,
    Dict,
    Optional,
)

import yt.wrapper as yt

from tractorun.mesh import Mesh
from tractorun.resources import Resources
from tractorun.run_internal import (
    TrainingScript,
    UserFunction,
    _prepare_and_get_toolbox,
    _run_tracto,
    _run_local,
)
from tractorun.toolbox import Toolbox


def run(
    user_function: Callable,
    *,
    yt_path: str,
    mesh: Mesh,
    user_config: Optional[Dict[Any, Any]] = None,
    docker_image: Optional[str] = None,
    resources: Optional[Resources] = None,
    yt_client: Optional[yt.YtClient] = None,
    wandb_enabled: bool = False,
    wandb_api_key: Optional[str] = None,
    local: bool = False,
) -> None:
    if local:
        return _run_local(
            UserFunction(function=user_function),
            yt_path=yt_path,
            mesh=mesh,
            user_config=user_config,
            resources=resources,
            yt_client=yt_client,
            docker_image=docker_image,
            wandb_enabled=wandb_enabled,
            wandb_api_key=wandb_api_key,
        )
    else:
        return _run_tracto(
            UserFunction(function=user_function),
            yt_path=yt_path,
            mesh=mesh,
            user_config=user_config,
            resources=resources,
            yt_client=yt_client,
            docker_image=docker_image,
            wandb_enabled=wandb_enabled,
            wandb_api_key=wandb_api_key,
        )


def run_script(
    training_script: str,
    *,
    yt_path: str,
    mesh: Mesh,
    user_config: Optional[Dict[Any, Any]] = None,
    docker_image: Optional[str] = None,
    local: bool = False,
) -> None:
    if local:
        return _run_local(
            runnable=TrainingScript(script_path=training_script),
            yt_path=yt_path,
            mesh=mesh,
            user_config=user_config,
            yt_client=None,
            resources=None,
            docker_image=docker_image,
        )
    else:
        return _run_tracto(
            runnable=TrainingScript(script_path=training_script),
            yt_path=yt_path,
            mesh=mesh,
            user_config=user_config,
            resources=None,
            yt_client=None,
            docker_image=docker_image,
        )


def prepare_and_get_toolbox() -> Toolbox:
    return _prepare_and_get_toolbox()
