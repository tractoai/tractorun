#!/usr/bin/env python3

import argparse
import json
import sys
import traceback
from typing import (
    Any,
    Optional,
    TypeVar,
)

import attrs
from cattrs import ClassValidationError
import yaml

from tractorun.bind import BindLocal
from tractorun.docker_auth import (
    DockerAuthData,
    DockerAuthSecret,
)
from tractorun.env import EnvVariable
from tractorun.exception import TractorunConfigError
from tractorun.mesh import Mesh
from tractorun.private.docker_auth import DockerAuthInternal
from tractorun.private.helpers import (
    create_attrs_converter,
    get_default_docker_image,
)
from tractorun.private.run import run_script
from tractorun.resources import Resources
from tractorun.run_info import RunInfo
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
    docker_auth_secret: Optional[DockerAuthSecretConfig] = attrs.field(default=None)

    @classmethod
    def load_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        converter = create_attrs_converter()
        try:
            return converter.structure(config, Config)
        except ClassValidationError as e:
            raise TractorunConfigError from e


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
    docker_auth_secret: DockerAuthData | None
    dry_run: bool

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

        docker_auth_secret = None
        if config.docker_auth_secret is not None:
            docker_auth_secret = DockerAuthSecret(cypress_path=config.docker_auth_secret.cypress_path)
        if args["docker_auth_secret.cypress_path"] is not None:
            docker_auth_secret = DockerAuthSecret(
                cypress_path=args["docker_auth_secret.cypress_path"],
            )

        new_config = EffectiveConfig(
            yt_path=_choose_value(args_value=args["yt_path"], config_value=config.yt_path),
            docker_image=_choose_value(
                args_value=args["docker_image"],
                config_value=config.docker_image,
                default=get_default_docker_image(),
            ),
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
            docker_auth_secret=docker_auth_secret,
            dry_run=args["dry_run"],
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
    parser.add_argument("--docker-image", type=str, help=f"docker image name. Default: {get_default_docker_image()}")
    parser.add_argument(
        "--docker-auth-secret.cypress-path",
        type=str,
        help="Path to the cypress node with format {}".format(
            json.dumps(
                attrs.asdict(DockerAuthInternal(username="placeholder", password="placeholder", auth="placeholder")),  # type: ignore
            ),
        ),
        default=None,
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
        help="set env variable by value or from cypress node. JSON message like {} or {}".format(
            EnvVariable(
                name="placeholder",
                value="placeholder",
            ),
            EnvVariable(
                name="placeholder",
                cypress_path="placeholder",
            ),
        ),
    )
    parser.add_argument("--dry-run", help="get internal information without running an operation", action="store_true")
    parser.add_argument("command", nargs="*", help="command to run")
    return parser


def make_configuration(cli_args: list) -> tuple[dict, Config, EffectiveConfig]:
    parser = make_cli_parser()
    args = vars(parser.parse_args(cli_args))
    file_config_content = Config.load_yaml(args["run_config_path"]) if args["run_config_path"] else Config()
    effective_config = EffectiveConfig.configure(
        config=file_config_content,
        args=args,
    )
    return args, file_config_content, effective_config


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class ConfigurationDebug:
    file_config: Config
    effective_config: EffectiveConfig
    cli_args: dict


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class CliRunInfo:
    configuration: ConfigurationDebug
    run_info: RunInfo | None
    errors: list[Any]


def main() -> None:
    args, file_config_content, effective_config = make_configuration(sys.argv[1:])

    run_info = None
    errors = []
    try:
        run_info = run_script(
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
            docker_auth=effective_config.docker_auth_secret,
            dry_run=effective_config.dry_run,
        )
    except Exception:
        errors.append(traceback.format_exc())
        if not effective_config.dry_run:
            raise
    if effective_config.dry_run:
        cli_run_info = CliRunInfo(
            configuration=ConfigurationDebug(
                file_config=file_config_content,
                cli_args=args,
                effective_config=effective_config,
            ),
            run_info=run_info,
            errors=errors,
        )

        print(
            json.dumps(
                attrs.asdict(cli_run_info),  # type: ignore
                indent=4,
                default=_json_encoder,
            ),
        )


def _json_encoder(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


if __name__ == "__main__":
    main()
