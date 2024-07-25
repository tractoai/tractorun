#!/usr/bin/env python3

import argparse
import json
import pprint
from typing import (
    Any,
    Optional,
    TypeVar,
)

import attrs
import cattr
import yaml

from tractorun.bind import Bind
from tractorun.constants import DEFAULT_DOCKER_IMAGE
from tractorun.exceptions import TractorunConfigError
from tractorun.mesh import Mesh
from tractorun.resources import Resources
from tractorun.run import run_script
from tractorun.sidecar import (
    RestartPolicy,
    Sidecar,
)


MESH_NODE_COUNT_DEFAULT = 1
MESH_PROCESS_PER_NODE_DEFAULT = 1
MESH_GPU_PER_PROCESS_DEFAULT = 0
LOCAL_DEFAULT = False


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
    bind: Optional[list[str]] = attrs.field(default=None)
    bind_lib: Optional[list[str]] = attrs.field(default=None)
    sidecar: list[Sidecar] = attrs.field(default=None)
    command: Optional[list[str]] = attrs.field(default=None)

    mesh: MeshConfig = attrs.field(default=MeshConfig())
    resources: ResourcesConfig = attrs.field(default=ResourcesConfig())

    @classmethod
    def load_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cattr.structure(config, Config)


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class EffectiveMeshConfig:
    node_count: int
    process_per_node: int
    gpu_per_process: int
    pool_trees: Optional[list[str]]


@attrs.define(kw_only=True, slots=True)
class EffectiveResourcesConfig:
    cpu_limit: Optional[float]
    memory_limit: Optional[int]


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class EffectiveConfig:
    yt_path: str
    docker_image: str
    user_config: Optional[dict[str, Any]]
    yt_operation_spec: Optional[dict[str, Any]]
    yt_task_spec: Optional[dict[str, Any]]
    local: bool
    bind: list[str]
    bind_lib: list[str]
    sidecar: list[Sidecar]
    command: list[str]

    mesh: EffectiveMeshConfig
    resources: EffectiveResourcesConfig

    @classmethod
    def configure(cls, args: dict[str, Any], config: Config) -> "EffectiveConfig":
        def _choose_value(args_value: _T, config_value: _T, default: Optional[_T] = None) -> _T:
            result = args_value if args_value is not None else config_value
            if result is None and default is not None:
                return default
            return result

        user_config = json.loads(args["user_config"]) if args["user_config"] is not None else None
        yt_operation_spec = json.loads(args["yt_operation_spec"]) if args["yt_operation_spec"] is not None else None
        yt_task_spec = json.loads(args["yt_task_spec"]) if args["yt_task_spec"] is not None else None
        sidecar = json.loads(args["sidecar"]) if args["sidecar"] is not None else None

        # here is `args["command"] or None` as a special hack
        # because argparse can't use default=None here
        command = _choose_value(args["command"] or None, config.command)
        if not command:
            raise TractorunConfigError("Command should be set in config or by cli param")
        yt_path = _choose_value(args["yt_path"], config.yt_path)
        if yt_path is None:
            raise TractorunConfigError("Command should be set in config or by cli param --yt-path")
        bind = _choose_value(args_value=args["bind"], config_value=config.bind)
        if bind is None:
            bind = []

        bind_lib = _choose_value(args_value=args["bind_lib"], config_value=config.bind_lib)
        if bind_lib is None:
            bind_lib = []

        new_config = EffectiveConfig(
            yt_path=_choose_value(args_value=args["yt_path"], config_value=config.yt_path),
            docker_image=_choose_value(args_value=args["docker_image"], config_value=config.docker_image),
            user_config=_choose_value(args_value=user_config, config_value=config.user_config),
            yt_operation_spec=_choose_value(args_value=yt_operation_spec, config_value=config.yt_operation_spec),
            yt_task_spec=_choose_value(args_value=yt_task_spec, config_value=config.yt_task_spec),
            local=_choose_value(args_value=args["local"], config_value=config.local, default=LOCAL_DEFAULT),
            bind=bind,
            bind_lib=bind_lib,
            sidecar=_choose_value(args_value=sidecar, config_value=config.sidecar),
            command=command,
            mesh=EffectiveMeshConfig(
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
            resources=EffectiveResourcesConfig(
                cpu_limit=_choose_value(
                    args_value=args["resources.cpu_limit"],
                    config_value=config.resources.cpu_limit,
                ),
                memory_limit=_choose_value(
                    args_value=args["resources.mem_limit"],
                    config_value=config.resources.memory_limit,
                ),
            ),
        )
        return new_config


def main() -> None:
    # Defaults shouldn't be set in argparse
    # because it interferes with merging with configs.
    # We don't want to use config values as defaults too.
    parser = argparse.ArgumentParser(
        description="Tractorun",
    )
    parser.add_argument("--yt-path", help="YT workdir", type=str)
    parser.add_argument("--run-config-path", type=str, help="path to tractorun config")
    parser.add_argument("--docker-image", type=str, help=f"docker image name. Default: {DEFAULT_DOCKER_IMAGE}")
    parser.add_argument("--mesh.node-count", type=int, help=f"mesh node count. Default: {MESH_NODE_COUNT_DEFAULT}")
    parser.add_argument(
        "--mesh.process-per-node", type=int, help=f"mesh process per node. Default: {MESH_PROCESS_PER_NODE_DEFAULT}"
    )
    parser.add_argument(
        "--mesh.gpu-per-process", type=int, help=f"mesh gpu per process. Default: {MESH_GPU_PER_PROCESS_DEFAULT}"
    )
    parser.add_argument("--mesh.pool-trees", help="mesh pool trees", action="append")
    parser.add_argument("--resources.cpu-limit", type=int, help="cpu limit")
    parser.add_argument("--resources.mem-limit", type=int, help="mem limit")
    parser.add_argument("--user-config", type=str, help="json config that will be passed to the jobs")
    parser.add_argument("--yt-operation-spec", help="yt operation spec", type=str)
    parser.add_argument("--yt-task-spec", help="yt task spec", type=str)
    parser.add_argument("--local", type=bool, help=f"enable local run mode. Default {LOCAL_DEFAULT}")
    parser.add_argument(
        "--bind",
        type=str,
        action="append",
        default=None,
        help="bind mounts to be passed to the docker container. Format: `local_path:remote_path`",
    )
    parser.add_argument(
        "--bind-lib",
        type=str,
        action="append",
        default=None,
        help="bind python libraries to the docker container and remote PYTHONPATH",
    )
    parser.add_argument(
        "--sidecar",
        nargs="*",
        help=f'sidecar in json format `{"command": "shell command", "restart_policy: "always"}`. Restart policy: {[p for p in RestartPolicy]}',
    )
    parser.add_argument("--dump-effective-config", help="print effective configuration", action="store_true")
    parser.add_argument("command", nargs="*", help="command to run")

    args = vars(parser.parse_args())

    file_config_content = Config.load_yaml(args["run_config_path"]) if args["run_config_path"] else Config()
    effective_config = EffectiveConfig.configure(
        config=file_config_content,
        args=args,
    )

    binds: list[Bind] = []
    for bind in effective_config.bind:
        source, destination = bind.split(":")
        binds.append(Bind(source=source, destination=destination))

    if args["dump_effective_config"]:
        print("Parsed args:")
        pprint.pprint(args)
        print("\nConfig from file:")
        pprint.pprint(attrs.asdict(file_config_content))  # type: ignore
        print("\nEffective config:")
        pprint.pprint(attrs.asdict(effective_config))  # type: ignore
        return

    run_script(
        command=effective_config.command,
        mesh=Mesh(
            node_count=effective_config.mesh.node_count,
            process_per_node=effective_config.mesh.process_per_node,
            gpu_per_process=effective_config.mesh.gpu_per_process,
            pool_trees=effective_config.mesh.pool_trees,
        ),
        resources=Resources(
            memory_limit=effective_config.resources.memory_limit,
            cpu_limit=effective_config.resources.cpu_limit,
        ),
        yt_path=effective_config.yt_path,
        docker_image=effective_config.docker_image,
        binds=binds,
        bind_libs=effective_config.bind_lib,
        sidecars=effective_config.sidecar,
        user_config=effective_config.user_config,
        yt_operation_spec=effective_config.yt_operation_spec,
        yt_task_spec=effective_config.yt_task_spec,
        local=effective_config.local,
    )


if __name__ == "__main__":
    main()
