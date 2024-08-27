#!/usr/bin/env python3

import argparse
import json
import sys
from typing import (
    Any,
    Optional,
    TypeVar,
)

import attrs
import yaml

from tractorun.bind import BindLocal
from tractorun.docker_auth import (
    DockerAuthData,
    DockerAuthPlainText,
    DockerAuthSecret,
)
from tractorun.env import EnvVariable
from tractorun.exception import TractorunConfigError
from tractorun.mesh import Mesh
from tractorun.private.constants import DEFAULT_DOCKER_IMAGE
from tractorun.private.helpers import AttrSerializer
from tractorun.private.run import run_script
from tractorun.resources import Resources
from tractorun.sidecar import (
    RestartPolicy,
    Sidecar,
)
from tractorun.stderr_reader import StderrMode
from tractorun.tensorproxy import TensorproxySidecar


MESH_NODE_COUNT_DEFAULT = 1
MESH_PROCESS_PER_NODE_DEFAULT = 1
MESH_GPU_PER_PROCESS_DEFAULT = 0
TENSORPROXY_ENABLED_DEFAULT = False
TENSORPROXY_YT_PATH_DEFAULT = "//home/tractorun/tensorproxy"
TENSORPROXY_RESTART_POLICY_DEFAULT = RestartPolicy.ALWAYS
LOCAL_DEFAULT = False
PROXY_STDERR_MODE_DEFAULT = StderrMode.disabled


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class MeshConfig:
    node_count: Optional[int] = attrs.field(default=None)
    process_per_node: Optional[int] = attrs.field(default=None)
    gpu_per_process: Optional[int] = attrs.field(default=None)
    pool_trees: Optional[list[str]] = attrs.field(default=None)


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class ResourcesConfig:
    cpu_limit: Optional[float] = attrs.field(default=None)
    memory_limit: Optional[int] = attrs.field(default=None)


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class SidecarConfig:
    command: list[str]
    restart_policy: RestartPolicy


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class TensorproxyConfig:
    enabled: Optional[bool] = attrs.field(default=None)
    restart_policy: Optional[RestartPolicy] = attrs.field(default=None)
    yt_path: Optional[str] = attrs.field(default=None)


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class EnvVariableConfig:
    name: str
    value: Optional[str] = None
    cypress_path: Optional[str] = None


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class DockerAuthSecretConfig:
    cypress_path: str


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class DockerAuthPlainTextConfig:
    username: str | None
    password: str | None
    auth: str | None


DockerAuthConfig = DockerAuthSecretConfig | DockerAuthPlainTextConfig


_T = TypeVar("_T")


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class Config:
    """yaml config representation"""

    yt_path: Optional[str] = attrs.field(default=None)
    docker_image: Optional[str] = attrs.field(default=None)
    user_config: Optional[dict[str, Any]] = attrs.field(default=None)
    yt_operation_spec: Optional[dict[str, Any]] = attrs.field(default=None)
    yt_task_spec: Optional[dict[str, Any]] = attrs.field(default=None)
    local: Optional[bool] = attrs.field(default=None)
    bind_local: Optional[list[str]] = attrs.field(default=None)
    bind_local_lib: Optional[list[str]] = attrs.field(default=None)
    proxy_stderr_mode: Optional[StderrMode] = attrs.field(default=None)
    command: Optional[list[str]] = attrs.field(default=None)

    mesh: MeshConfig = attrs.field(default=MeshConfig())
    resources: ResourcesConfig = attrs.field(default=ResourcesConfig())
    sidecars: Optional[list[SidecarConfig]] = attrs.field(default=None)
    env: Optional[list[EnvVariableConfig]] = attrs.field(default=None)
    tensorproxy: TensorproxyConfig = attrs.field(default=TensorproxyConfig())
    docker_auth: Optional[DockerAuthConfig] = attrs.field(default=None)

    @classmethod
    def load_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return AttrSerializer(Config).deserialize(config)


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class EffectiveConfig:
    yt_path: str
    docker_image: str
    user_config: Optional[dict[str, Any]]
    yt_operation_spec: Optional[dict[str, Any]]
    yt_task_spec: Optional[dict[str, Any]]
    local: bool
    bind_local: list[BindLocal]
    bind_local_lib: list[str]
    proxy_stderr_mode: StderrMode
    command: list[str]

    mesh: Mesh
    resources: Resources
    sidecars: list[Sidecar]
    env: list[EnvVariable]
    tensorproxy: TensorproxySidecar
    docker_auth: DockerAuthData | None

    @classmethod
    def configure(cls, args: dict[str, Any], config: Config) -> "EffectiveConfig":
        # TODO: transform args into some special canonized object
        def _choose_value(args_value: _T, config_value: _T, default: Optional[_T] = None) -> _T:
            result = args_value if args_value is not None else config_value
            if result is None and default is not None:
                return default
            return result

        user_config = json.loads(args["user_config"]) if args["user_config"] is not None else None
        yt_operation_spec = json.loads(args["yt_operation_spec"]) if args["yt_operation_spec"] is not None else None
        yt_task_spec = json.loads(args["yt_task_spec"]) if args["yt_task_spec"] is not None else None

        # here is `args["command"] or None` as a special hack
        # because argparse can't use default=None here
        command = _choose_value(args["command"] or None, config.command)
        if not command:
            raise TractorunConfigError("Command should be set in config or by cli param")
        yt_path = _choose_value(args["yt_path"], config.yt_path)
        if yt_path is None:
            raise TractorunConfigError("Command should be set in config or by cli param --yt-path")
        binds = _choose_value(args_value=args["bind_local"], config_value=config.bind_local)
        if binds is None:
            binds = []
        effective_binds: list[BindLocal] = []
        for bind in binds:
            source, destination = bind.split(":")
            effective_binds.append(
                BindLocal(
                    source=source,
                    destination=destination,
                ),
            )

        bind_lib = _choose_value(args_value=args["bind_local_lib"], config_value=config.bind_local_lib)
        if bind_lib is None:
            bind_lib = []

        sidecars = config.sidecars
        if args["sidecar"] is not None:
            raw_sidecars = [json.loads(sidecar) for sidecar in args["sidecar"]]
            sidecars = [
                SidecarConfig(
                    command=sidecar["command"],
                    restart_policy=sidecar["restart_policy"],
                )
                for sidecar in raw_sidecars
            ]
        if sidecars is None:
            sidecars = []

        env = config.env
        if args["env"] is not None:
            raw_env = [json.loads(e) for e in args["env"]]
            env = [
                EnvVariableConfig(
                    name=e["name"],
                    cypress_path=e.get("cypress_path"),
                    value=e.get("value"),
                )
                for e in raw_env
            ]
        if env is None:
            env = []

        docker_auth_config = config.docker_auth
        if args["docker_auth"] is not None:
            TT = TypeVar("TT", bound=DockerAuthConfig)
            def _hack(t: TT)
            docker_auth_config = AttrSerializer[DockerAuthConfig](
                DockerAuthSecretConfig | DockerAuthSecretConfig
            ).deserialize(
                args["docker_auth"],
            )
        docker_auth: DockerAuthData
        match docker_auth_config:
            case DockerAuthSecretConfig():
                assert isinstance(docker_auth_config, DockerAuthSecret)  # skip type warning in pycharm
                docker_auth = DockerAuthSecret(cypress_path=docker_auth_config.cypress_path)
            case DockerAuthPlainTextConfig():
                assert isinstance(docker_auth_config, DockerAuthPlainTextConfig)  # skip type warning in pycharm
                docker_auth = DockerAuthPlainText(
                    username=docker_auth_config.username,
                    password=docker_auth_config.password,
                    auth=docker_auth_config.auth,
                )
            case _:
                raise TractorunConfigError("Unknown docker auth type")

        new_config = EffectiveConfig(
            yt_path=_choose_value(args_value=args["yt_path"], config_value=config.yt_path),
            docker_image=_choose_value(args_value=args["docker_image"], config_value=config.docker_image),
            user_config=_choose_value(args_value=user_config, config_value=config.user_config),
            yt_operation_spec=_choose_value(args_value=yt_operation_spec, config_value=config.yt_operation_spec),
            yt_task_spec=_choose_value(args_value=yt_task_spec, config_value=config.yt_task_spec),
            local=_choose_value(args_value=args["local"], config_value=config.local, default=LOCAL_DEFAULT),
            bind_local=effective_binds,
            bind_local_lib=bind_lib,
            proxy_stderr_mode=_choose_value(
                args_value=args["proxy_stderr_mode"],
                config_value=config.proxy_stderr_mode,
                default=PROXY_STDERR_MODE_DEFAULT,
            ),
            sidecars=[
                Sidecar(
                    command=sidecar.command,
                    restart_policy=sidecar.restart_policy,
                )
                for sidecar in sidecars
            ],
            env=[
                EnvVariable(
                    name=e.name,
                    value=e.value,
                    cypress_path=e.cypress_path,
                )
                for e in env
            ],
            command=command,
            mesh=Mesh(
                node_count=_choose_value(
                    args_value=args["mesh.node_count"],
                    config_value=config.mesh.node_count,
                    default=MESH_NODE_COUNT_DEFAULT,
                ),
                process_per_node=_choose_value(
                    args_value=args["mesh.process_per_node"],
                    config_value=config.mesh.process_per_node,
                    default=MESH_PROCESS_PER_NODE_DEFAULT,
                ),
                gpu_per_process=_choose_value(
                    args_value=args["mesh.gpu_per_process"],
                    config_value=config.mesh.gpu_per_process,
                    default=MESH_GPU_PER_PROCESS_DEFAULT,
                ),
                pool_trees=_choose_value(
                    args_value=args["mesh.pool_trees"],
                    config_value=config.mesh.pool_trees,
                ),
            ),
            resources=Resources(
                cpu_limit=_choose_value(
                    args_value=args["resources.cpu_limit"],
                    config_value=config.resources.cpu_limit,
                ),
                memory_limit=_choose_value(
                    args_value=args["resources.memory_limit"],
                    config_value=config.resources.memory_limit,
                ),
            ),
            tensorproxy=TensorproxySidecar(
                enabled=_choose_value(
                    args_value=args["tensorproxy.enabled"],
                    config_value=config.tensorproxy.enabled,
                    default=TENSORPROXY_ENABLED_DEFAULT,
                ),
                yt_path=_choose_value(
                    args_value=args["tensorproxy.yt_path"],
                    config_value=config.tensorproxy.yt_path,
                    default=TENSORPROXY_YT_PATH_DEFAULT,
                ),
                restart_policy=_choose_value(
                    args_value=args["tensorproxy.restart_policy"],
                    config_value=config.tensorproxy.restart_policy,
                    default=TENSORPROXY_RESTART_POLICY_DEFAULT,
                ),
            ),
            docker_auth=docker_auth,
        )
        return new_config


def make_cli_parser() -> argparse.ArgumentParser:
    # Defaults shouldn't be set in argparse
    # because it interferes with merging with configs.
    # We don't want to use config values as defaults too.
    parser = argparse.ArgumentParser(
        description="Tractorun",
    )
    parser.add_argument("--yt-path", help="YT workdir", type=str)
    parser.add_argument("--run-config-path", type=str, help="path to tractorun config")
    parser.add_argument("--docker-image", type=str, help=f"docker image name. Default: {DEFAULT_DOCKER_IMAGE}")
    parser.add_argument(
        "--docker-auth",
        type=str,
        help="auth data for docker registry in json: {} or {}".format(
            json.dumps(
                attrs.asdict(
                    DockerAuthPlainText(password="placeholder", username="placeholder", auth="placeholder"),  # type: ignore
                ),
            ),
            json.dumps(
                attrs.asdict(
                    DockerAuthSecret(cypress_path="placeholder"),  # type: ignore
                ),
            ),
        ),
    )
    parser.add_argument("--mesh.node-count", type=int, help=f"mesh node count. Default: {MESH_NODE_COUNT_DEFAULT}")
    parser.add_argument(
        "--mesh.process-per-node", type=int, help=f"mesh process per node. Default: {MESH_PROCESS_PER_NODE_DEFAULT}"
    )
    parser.add_argument(
        "--mesh.gpu-per-process", type=int, help=f"mesh gpu per process. Default: {MESH_GPU_PER_PROCESS_DEFAULT}"
    )
    parser.add_argument("--mesh.pool-trees", help="mesh pool trees", action="append")
    parser.add_argument("--resources.cpu-limit", type=int, help="cpu limit")
    parser.add_argument("--resources.memory-limit", type=int, help="mem limit")
    parser.add_argument("--user-config", type=str, help="json config that will be passed to the jobs")
    parser.add_argument(
        "--tensorproxy.enabled",
        type=bool,
        help=f"Enable tensorproxy sidecar. Default {TENSORPROXY_ENABLED_DEFAULT}",
    )
    parser.add_argument(
        "--tensorproxy.yt_path",
        type=str,
        help=f"YT path to tensorproxy binary. Default {TENSORPROXY_YT_PATH_DEFAULT}",
    )
    parser.add_argument(
        "--tensorproxy.restart_policy",
        type=str,
        help=f"Tensorflow's sidecar restart policy. Default {TENSORPROXY_RESTART_POLICY_DEFAULT}",
    )
    parser.add_argument("--yt-operation-spec", help="yt operation spec", type=str)
    parser.add_argument("--yt-task-spec", help="yt task spec", type=str)
    # TODO: rename to run mode
    parser.add_argument("--local", type=bool, help=f"enable local run mode. Default {LOCAL_DEFAULT}")
    parser.add_argument(
        "--bind-local",
        type=str,
        action="append",
        default=None,
        help="bind local file or folder to be passed to the docker container. Format: `local_path:remote_path`",
    )
    parser.add_argument(
        "--bind-local-lib",
        type=str,
        action="append",
        default=None,
        help="bind local python libraries to the docker container and remote PYTHONPATH",
    )
    parser.add_argument(
        "--sidecar",
        action="append",
        help='sidecar in json format `{"command": ["command"], "restart_policy: "always"}`. Restart policy: '
        + ", ".join(p for p in RestartPolicy),
    )
    parser.add_argument(
        "--proxy-stderr-mode",
        type=StderrMode,
        help="Proxy jobs stderr to terminal. Mode: " + ", ".join(m for m in StderrMode),
    )
    parser.add_argument(
        "--env",
        action="append",
        help='set env variable by value or from cypress node. JSON message like `{"name": "foo", "value: "real value", "cypress_path": "//tmp/foo"}`',
    )
    parser.add_argument("--dump-effective-config", help="print effective configuration", action="store_true")
    parser.add_argument("command", nargs="*", help="command to run")
    return parser


def main() -> None:
    parser = make_cli_parser()
    args = vars(parser.parse_args(sys.argv[1:]))
    file_config_content = Config.load_yaml(args["run_config_path"]) if args["run_config_path"] else Config()
    effective_config = EffectiveConfig.configure(
        config=file_config_content,
        args=args,
    )

    if args["dump_effective_config"]:
        print("Parsed args:")
        print(json.dumps(args, indent=4))
        print("\nConfig from file:")
        print(
            json.dumps(
                attrs.asdict(file_config_content),  # type: ignore
                indent=4,
            ),
        )
        print("\nEffective config:")
        print(
            json.dumps(
                attrs.asdict(effective_config),  # type: ignore
                indent=4,
            ),
        )
        return

    run_script(
        command=effective_config.command,
        mesh=effective_config.mesh,
        resources=effective_config.resources,
        yt_path=effective_config.yt_path,
        docker_image=effective_config.docker_image,
        binds_local=effective_config.bind_local,
        binds_local_lib=effective_config.bind_local_lib,
        tensorproxy=effective_config.tensorproxy,
        proxy_stderr_mode=effective_config.proxy_stderr_mode,
        sidecars=effective_config.sidecars,
        env=effective_config.env,
        user_config=effective_config.user_config,
        yt_operation_spec=effective_config.yt_operation_spec,
        yt_task_spec=effective_config.yt_task_spec,
        local=effective_config.local,
        docker_auth=effective_config.docker_auth,
    )


if __name__ == "__main__":
    main()
