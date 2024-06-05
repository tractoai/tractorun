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
    _run,
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
    yt_cli: Optional[yt.YtClient] = None,
) -> None:
    _run(
        UserFunction(function=user_function),
        yt_path=yt_path,
        mesh=mesh,
        user_config=user_config,
        resources=resources,
        yt_cli=yt_cli,
        docker_image=docker_image,
    )


def run_script(
    training_script: str,
    *,
    yt_path: str,
    mesh: Mesh,
    user_config: Optional[Dict[Any, Any]] = None,
    docker_image: Optional[str] = None,
) -> None:
    _run(
        runnable=TrainingScript(script_path=training_script),
        yt_path=yt_path,
        mesh=mesh,
        user_config=user_config,
        resources=None,
        yt_cli=None,
        docker_image=docker_image,
    )


def prepare_and_get_toolbox() -> Toolbox:
    return _prepare_and_get_toolbox()
