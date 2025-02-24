#!/usr/bin/env python3

import argparse
import json
import sys
import traceback
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    cast,
    overload,
)

from attr import AttrsInstance
import attrs
from cattrs import ClassValidationError
import yaml

from tractorun import __version__
from tractorun.bind import (
    BindAttributes,
    BindCypress,
    BindLocal,
)
from tractorun.docker_auth import (
    DockerAuthData,
    DockerAuthSecret,
)
from tractorun.env import EnvVariable
from tractorun.exception import TractorunConfigError
from tractorun.mesh import Mesh
from tractorun.operation_log import OperationLogMode
from tractorun.private.constants import (
    DEFAULT_CLUSTER_CONFIG_PATH,
    DEFAULT_TENSORPROXY_PATH,
)
from tractorun.private.docker_auth import DockerAuthInternal
from tractorun.private.helpers import (
    create_attrs_converter,
    get_default_docker_image,
)
from tractorun.resources import Resources
from tractorun.run import run_script
from tractorun.run_info import RunInfo
from tractorun.sidecar import (
    RestartPolicy,
    Sidecar,
)
from tractorun.stderr_reader import StderrMode
from tractorun.tensorproxy import TensorproxySidecar


__default_mesh = Mesh()

MESH_NODE_COUNT_DEFAULT = __default_mesh.node_count
MESH_PROCESS_PER_NODE_DEFAULT = __default_mesh.process_per_node
MESH_GPU_PER_PROCESS_DEFAULT = __default_mesh.gpu_per_process
TENSORPROXY_ENABLED_DEFAULT = False
CLUSTER_CONFIG_PATH_DEFAULT = DEFAULT_CLUSTER_CONFIG_PATH
TENSORPROXY_RESTART_POLICY_DEFAULT = RestartPolicy.ALWAYS
LOCAL_DEFAULT = False
NO_WAIT_DEFAULT = False
PROXY_STDERR_MODE_DEFAULT = StderrMode.disabled
OPERATION_LOG_MODE_DEFAULT = OperationLogMode.default


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class MeshConfig:
    node_count: int | None = attrs.field(default=None)
    process_per_node: int | None = attrs.field(default=None)
    gpu_per_process: int | None = attrs.field(default=None)
    pool_trees: list[str] | None = attrs.field(default=None)
    pool: str | None = attrs.field(default=None)


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class ResourcesConfig:
    cpu_limit: float | None = attrs.field(default=None)
    memory_limit: int | None = attrs.field(default=None)


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class TensorproxyConfig:
    enabled: bool | None = attrs.field(default=None)
    restart_policy: RestartPolicy | None = attrs.field(default=None)
    yt_path: str | None = attrs.field(default=None)


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class DockerAuthSecretConfig:
    cypress_path: str


_T = TypeVar("_T")


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class Config:
    """yaml config representation"""

    yt_path: str | None = attrs.field(default=None)
    docker_image: str | None = attrs.field(default=None)
    title: str | None = attrs.field(default=None)
    user_config: dict[str, Any] | None = attrs.field(default=None)
    yt_operation_spec: dict[str, Any] | None = attrs.field(default=None)
    yt_task_spec: dict[str, Any] | None = attrs.field(default=None)
    local: bool | None = attrs.field(default=None)
    no_wait: bool | None = attrs.field(default=None)
    bind_local: list[BindLocal] | None = attrs.field(default=None)
    bind_local_lib: list[str] | None = attrs.field(default=None)
    bind_cypress: list[BindCypress] | None = attrs.field(default=None)
    proxy_stderr_mode: StderrMode | None = attrs.field(default=None)
    operation_log_mode: OperationLogMode | None = attrs.field(default=None)
    cluster_config_path: str | None = attrs.field(default=None)
    command: list[str] | None = attrs.field(default=None)

    mesh: MeshConfig = attrs.field(default=MeshConfig())
    resources: ResourcesConfig = attrs.field(default=ResourcesConfig())
    sidecars: list[Sidecar] | None = attrs.field(default=None)
    env: list[EnvVariable] | None = attrs.field(default=None)
    tensorproxy: TensorproxyConfig = attrs.field(default=TensorproxyConfig())
    docker_auth_secret: DockerAuthSecretConfig | None = attrs.field(default=None)

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
    yt_path: str | None
    docker_image: str
    title: str | None
    user_config: dict[str, Any]
    yt_operation_spec: dict[str, Any]
    yt_task_spec: dict[str, Any]
    local: bool
    no_wait: bool
    bind_local: list[BindLocal]
    bind_local_lib: list[str]
    bind_cypress: list[BindCypress]
    proxy_stderr_mode: StderrMode
    operation_log_mode: OperationLogMode
    cluster_config_path: str
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
        @overload
        def _choose_value(args_value: _T | None, config_value: _T | None, default: None = None) -> _T | None: ...

        @overload
        def _choose_value(args_value: _T | None, config_value: _T | None, default: _T) -> _T: ...

        def _choose_value(args_value: _T | None, config_value: _T | None, default: _T | None = None) -> _T | None:
            result = args_value if args_value is not None else config_value
            if result is None and default is not None:
                return default
            return result

        command = _choose_value(args["command"] or None, config.command)
        if not command:
            raise TractorunConfigError("Command should be set in config or by cli param")
        yt_path = _choose_value(args["yt_path"], config.yt_path)
        docker_image = _choose_value(
            args_value=args["docker_image"],
            config_value=config.docker_image,
            default=get_default_docker_image(),
        )
        if docker_image is None:
            raise TractorunConfigError("docker_image should be set in config or by cli param")

        new_config = EffectiveConfig(
            yt_path=yt_path,
            docker_image=docker_image,
            title=_choose_value(args_value=args["title"], config_value=config.title),
            user_config=_choose_value(args_value=args["user_config"], config_value=config.user_config, default={}),
            yt_operation_spec=_choose_value(
                args_value=args["yt_operation_spec"],
                config_value=config.yt_operation_spec,
                default={},
            ),
            yt_task_spec=_choose_value(args_value=args["yt_task_spec"], config_value=config.yt_task_spec, default={}),
            local=_choose_value(args_value=args["local"], config_value=config.local, default=LOCAL_DEFAULT),
            no_wait=_choose_value(args_value=args["no_wait"], config_value=config.no_wait, default=NO_WAIT_DEFAULT),
            cluster_config_path=_choose_value(
                args_value=args["cluster_config_path"],
                config_value=config.cluster_config_path,
                default=CLUSTER_CONFIG_PATH_DEFAULT,
            ),
            bind_local=_choose_value(
                args_value=args["bind_local"],
                config_value=config.bind_local,
                default=[],
            ),
            bind_local_lib=_choose_value(
                args_value=args["bind_local_lib"], config_value=config.bind_local_lib, default=[]
            ),
            bind_cypress=_choose_value(
                args_value=args["bind_cypress"],
                config_value=config.bind_cypress,
                default=[],
            ),
            proxy_stderr_mode=_choose_value(
                args_value=args["proxy_stderr_mode"],
                config_value=config.proxy_stderr_mode,
                default=PROXY_STDERR_MODE_DEFAULT,
            ),
            operation_log_mode=_choose_value(
                args_value=args["operation_log_mode"],
                config_value=config.operation_log_mode,
                default=OPERATION_LOG_MODE_DEFAULT,
            ),
            sidecars=_choose_value(args_value=args["sidecar"], config_value=config.sidecars, default=[]),
            env=_choose_value(args_value=args["env"], config_value=config.env, default=[]),
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
                pool=_choose_value(
                    args_value=args["mesh.pool"],
                    config_value=config.mesh.pool,
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
                    default=DEFAULT_TENSORPROXY_PATH,
                ),
                restart_policy=_choose_value(
                    args_value=args["tensorproxy.restart_policy"],
                    config_value=config.tensorproxy.restart_policy,
                    default=TENSORPROXY_RESTART_POLICY_DEFAULT,
                ),
            ),
            docker_auth_secret=cls._make_docker_auth_secret(
                _choose_value(
                    args_value=args["docker_auth_secret.cypress_path"],
                    # TODO: make some helper
                    config_value=getattr(config.docker_auth_secret, "cypress_path", None),
                ),
            ),
            dry_run=args["dry_run"],
        )
        return new_config

    @classmethod
    def _make_docker_auth_secret(cls, docker_auth_cypress_path: str | None) -> DockerAuthSecret | None:
        if docker_auth_cypress_path is None:
            return None
        return DockerAuthSecret(cypress_path=docker_auth_cypress_path)


def _load_json(value: str | None) -> dict | None:
    if value is None:
        return None
    return json.loads(value)


def _parse_bind_local_arg(value: str | None) -> BindLocal | None:
    if value is None:
        return None
    if value.startswith("{"):
        return BindLocal(**json.loads(value))
    source, destination = value.split(":")
    return BindLocal(source=source, destination=destination)


def _parse_bind_cypress_arg(value: str | None) -> BindCypress | None:
    if value is None:
        return None
    if value.startswith("{"):
        converter = create_attrs_converter()
        try:
            return converter.structure(json.loads(value), BindCypress)
        except ClassValidationError as e:
            raise TractorunConfigError from e
    source, destination = value.split(":")
    return BindCypress(source=source, destination=destination)


def _attr_to_json(value: object) -> str:
    if TYPE_CHECKING:
        value = cast(AttrsInstance, value)
    return json.dumps(attrs.asdict(value))


def make_cli_parser() -> argparse.ArgumentParser:
    # Defaults shouldn't be set in argparse
    # because it interferes with merging with configs.
    # We don't want to use config values as defaults too.
    parser = argparse.ArgumentParser(
        description="Tractorun",
    )
    parser.add_argument(
        "--yt-path", type=str, help="base directory for tractorun data. Default: //tmp/tractorun/{uuid}"
    )
    parser.add_argument("--run-config-path", type=str, help="path to tractorun config")
    parser.add_argument("--docker-image", type=str, help=f"docker image name. Default: {get_default_docker_image()}")
    parser.add_argument(
        "--docker-auth-secret.cypress-path",
        type=str,
        help="path to the cypress node with format {}".format(
            json.dumps(
                _attr_to_json(DockerAuthInternal(username="placeholder", password="placeholder", auth="placeholder")),  # type: ignore
            ),
        ),
        default=None,
    )
    parser.add_argument("--title", type=str, help="title of the operation")
    parser.add_argument("--mesh.node-count", type=int, help=f"mesh node count. Default: {MESH_NODE_COUNT_DEFAULT}")
    parser.add_argument(
        "--mesh.process-per-node", type=int, help=f"mesh process per node. Default: {MESH_PROCESS_PER_NODE_DEFAULT}"
    )
    parser.add_argument(
        "--mesh.gpu-per-process", type=int, help=f"mesh gpu per process. Default: {MESH_GPU_PER_PROCESS_DEFAULT}"
    )
    parser.add_argument("--mesh.pool-trees", help="mesh pool trees", action="append")
    parser.add_argument("--mesh.pool", help="mesh pool", type=str)
    parser.add_argument("--resources.cpu-limit", type=int, help="cpu limit")
    parser.add_argument("--resources.memory-limit", type=int, help="mem limit")
    parser.add_argument("--user-config", type=_load_json, help="json config that will be passed to the jobs")
    parser.add_argument(
        "--cluster-config-path",
        type=str,
        help=f"path to the global tractorun config on YTSaurus cluster. Default: {CLUSTER_CONFIG_PATH_DEFAULT}",
    )
    parser.add_argument(
        "--tensorproxy.enabled",
        type=bool,
        help=f"enable tensorproxy sidecar. Default {TENSORPROXY_ENABLED_DEFAULT}",
    )
    parser.add_argument(
        "--tensorproxy.yt_path",
        type=str,
        help=f"YTsaurus path to tensorproxy binary. Default {DEFAULT_TENSORPROXY_PATH}",
    )
    parser.add_argument(
        "--tensorproxy.restart_policy",
        type=str,
        help=f"Tensorflow's sidecar restart policy. Default {TENSORPROXY_RESTART_POLICY_DEFAULT}",
    )
    parser.add_argument("--yt-operation-spec", help="yt operation spec", type=_load_json)
    parser.add_argument("--yt-task-spec", help="yt task spec", type=_load_json)
    # TODO: rename to run mode
    parser.add_argument("--local", type=bool, help=f"enable local run mode. Default {LOCAL_DEFAULT}")
    parser.add_argument(
        "--bind-local",
        type=_parse_bind_local_arg,
        action="append",
        default=None,
        help="bind local file or folder to be passed to the docker container. Format: `local_path:remote_path` or {}".format(
            _attr_to_json(
                BindLocal(
                    source="placeholder",
                    destination="placeholder",
                ),
            ),
        ),
    )
    parser.add_argument(
        "--bind-local-lib",
        type=str,
        action="append",
        default=None,
        help="path to local python library to bind it to the remote container and remote PYTHONPATH",
    )
    parser.add_argument(
        "--bind-cypress",
        type=_parse_bind_cypress_arg,
        action="append",
        default=None,
        help="bind cypress file to be passed to the docker container. Format: `local_path:remote_path` or {0}".format(
            _attr_to_json(
                BindCypress(
                    source="yt path",
                    destination="path inside job",
                    attributes=BindAttributes(),
                ),
            ),
        ),
    )
    parser.add_argument(
        "--sidecar",
        action="append",
        type=Sidecar.from_args,
        help="sidecar in json format `{example}`. Restart policy: {policy}".format(
            policy=", ".join(p for p in RestartPolicy),
            example=_attr_to_json(Sidecar(command=["placeholder"], restart_policy=RestartPolicy.ALWAYS)),
        ),
    )
    parser.add_argument(
        "--proxy-stderr-mode",
        type=StderrMode,
        help="proxy jobs stderr to terminal. Mode: "
        + ", ".join(m for m in StderrMode)
        + f". Default: {PROXY_STDERR_MODE_DEFAULT}",
    )
    parser.add_argument(
        "--operation-log-mode",
        type=OperationLogMode,
        help="store operation log mode. Mode: "
        + ", ".join(m for m in OperationLogMode)
        + f". Default: {OPERATION_LOG_MODE_DEFAULT}",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        default=None,
        help=f"don't create transaction and don't wait for operation to complete. Default {NO_WAIT_DEFAULT}",
    )
    parser.add_argument(
        "--env",
        action="append",
        type=EnvVariable.from_args,
        help="set env variable by value or from cypress node. JSON message like {} or {}".format(
            _attr_to_json(
                EnvVariable(
                    name="placeholder",
                    value="placeholder",
                ),
            ),
            _attr_to_json(
                EnvVariable(
                    name="placeholder",
                    cypress_path="placeholder",
                ),
            ),
        ),
    )
    parser.add_argument("--dry-run", help="get internal information without running an operation", action="store_true")
    parser.add_argument("--version", help="show version", action="version", version=f"tractorun {__version__}")
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
            user_command=effective_config.command,
            mesh=effective_config.mesh,
            title=effective_config.title,
            resources=effective_config.resources,
            yt_path=effective_config.yt_path,
            docker_image=effective_config.docker_image,
            binds_local=effective_config.bind_local,
            binds_local_lib=effective_config.bind_local_lib,
            binds_cypress=effective_config.bind_cypress,
            tensorproxy=effective_config.tensorproxy,
            proxy_stderr_mode=effective_config.proxy_stderr_mode,
            operation_log_mode=effective_config.operation_log_mode,
            sidecars=effective_config.sidecars,
            env=effective_config.env,
            user_config=effective_config.user_config,
            cluster_config_path=effective_config.cluster_config_path,
            yt_operation_spec=effective_config.yt_operation_spec,
            yt_task_spec=effective_config.yt_task_spec,
            no_wait=effective_config.no_wait,
            docker_auth=effective_config.docker_auth_secret,
            dry_run=effective_config.dry_run,
            yt_client=None,
            local=effective_config.local,
        )
    except Exception:
        errors.append(traceback.format_exc())
        if not effective_config.dry_run:
            raise
    if effective_config.dry_run or effective_config.no_wait:
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
