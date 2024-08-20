from typing import (
    Any,
    Optional,
)

from tractorun.bind import BindLocal
from tractorun.env import EnvVariable
from tractorun.mesh import Mesh
from tractorun.private.run_internal import (
    Command,
    run_local,
    run_tracto,
)
from tractorun.resources import Resources
from tractorun.sidecar import Sidecar
from tractorun.stderr_reader import StderrMode
from tractorun.tensorproxy import TensorproxySidecar


def run_script(
    command: list[str],
    *,
    yt_path: str,
    mesh: Mesh,
    docker_image: str,
    resources: Resources,
    tensorproxy: TensorproxySidecar,
    user_config: Optional[dict[Any, Any]],
    binds_local: list[BindLocal],
    binds_local_lib: list[str],
    sidecars: list[Sidecar],
    env: list[EnvVariable],
    local: bool,
    yt_operation_spec: Optional[dict[Any, Any]],
    yt_task_spec: Optional[dict[Any, Any]],
    proxy_stderr_mode: StderrMode,
) -> None:
    if binds_local is None:
        binds_local = []
    if local:
        run_local(
            runnable=Command(command=command),
            yt_path=yt_path,
            mesh=mesh,
            sidecars=sidecars,
            env=env,
            yt_client=None,
            tensorproxy=tensorproxy,
        )
    else:
        run_tracto(
            runnable=Command(command=command),
            yt_path=yt_path,
            mesh=mesh,
            user_config=user_config,
            resources=resources,
            yt_client=None,
            docker_image=docker_image,
            binds_local=binds_local,
            binds_local_lib=binds_local_lib,
            tensorproxy=tensorproxy,
            sidecars=sidecars,
            env=env,
            proxy_stderr_mode=proxy_stderr_mode,
            yt_operation_spec=yt_operation_spec,
            yt_task_spec=yt_task_spec,
        )
