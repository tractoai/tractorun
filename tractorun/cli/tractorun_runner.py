#!/usr/bin/env python3

import argparse
import json
import pprint
from typing import (
    Any,
    Dict,
    List,
    Optional,
    TypeVar,
)

import attr
import cattr
import yaml

from tractorun.bind import Bind
from tractorun.constants import DEFAULT_DOCKER_IMAGE
from tractorun.exceptions import TractorunConfigError
from tractorun.mesh import Mesh
from tractorun.resources import Resources
from tractorun.run import run_script


CONFIG_PATH_DEFAULT = None
NNODES_DEFAULT = 1
NPROC_PER_NODE_DEFAULT = 0
NGPU_PER_PROC_DEFAULT = 0
LOCAL_DEFAULT = False
CPU_LIMIT_DEFAULT = None
MEM_LIMIT_DEFAULT = None


@attr.define(kw_only=True, slots=True, auto_attribs=True)
class MeshConfig:
    node_count: Optional[int] = attr.ib(default=None)
    process_per_node: Optional[int] = attr.ib(default=None)
    gpu_per_process: Optional[int] = attr.ib(default=None)
    pool_trees: Optional[List[str]] = attr.ib(default=None)


@attr.define(kw_only=True, slots=True, auto_attribs=True)
class ResourcesConfig:
    cpu_limit: Optional[float] = attr.ib(default=None)
    memory_limit: Optional[int] = attr.ib(default=None)


@attr.define(kw_only=True, slots=True, auto_attribs=True)
class EffectiveMeshConfig:
    node_count: int
    process_per_node: int
    gpu_per_process: int
    pool_trees: Optional[List[str]]


@attr.define(kw_only=True, slots=True)
class EffectiveResourcesConfig:
    cpu_limit: float
    memory_limit: int


_T = TypeVar("_T")


@attr.define(kw_only=True, slots=True, auto_attribs=True)
class Config:
    yt_path: Optional[str] = attr.field(default=None)
    docker_image: Optional[str] = attr.field(default=None)
    user_config: Optional[Dict[str, Any]] = attr.field(default=None)
    yt_operation_spec: Optional[Dict[str, Any]] = attr.field(default=None)
    yt_task_spec: Optional[Dict[str, Any]] = attr.field(default=None)
    local: Optional[bool] = attr.field(default=None)
    bind: Optional[List[str]] = attr.field(default=None)
    command: Optional[List[str]] = attr.field(default=None)

    mesh: MeshConfig = attr.field(default=MeshConfig())
    resources: ResourcesConfig = attr.field(default=ResourcesConfig())

    @classmethod
    def load_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cattr.structure(config, Config)


@attr.define(kw_only=True, slots=True, auto_attribs=True)
class EffectiveConfig:
    yt_path: str = attr.field(default=None)
    docker_image: str
    user_config: Optional[Dict[str, Any]]
    yt_operation_spec: Optional[Dict[str, Any]]
    yt_task_spec: Optional[Dict[str, Any]]
    local: bool
    bind: List[str]
    command: List[str]

    mesh: EffectiveMeshConfig
    resources: EffectiveResourcesConfig

    @classmethod
    def configure(cls, args: Dict[str, Any], config: Config) -> "EffectiveConfig":
        def _choose_value(args_value: _T, config_value: _T) -> _T:
            result = args_value if args_value is not None else config_value
            return result

        user_config = json.loads(args["user_config"]) if args["user_config"] is not None else None
        yt_operation_spec = json.loads(args["yt_operation_spec"]) if args["yt_operation_spec"] is not None else None
        yt_task_spec = json.loads(args["yt_task_spec"]) if args["yt_task_spec"] is not None else None
        bind = args["bind"] if args["bind"] is not None else []

        command = _choose_value(args["command"], config.command)
        if not command:
            raise TractorunConfigError("Command should be set in config or by cli param")

        yt_path = _choose_value(args["yt_path"], config.yt_path)
        if yt_path is None:
            raise TractorunConfigError("Command should be set in config or by cli param --yt-path")

        new_config = EffectiveConfig(
            yt_path=_choose_value(args_value=args["yt_path"], config_value=config.yt_path),
            docker_image=_choose_value(args_value=args["docker_image"], config_value=config.docker_image),
            user_config=_choose_value(args_value=user_config, config_value=user_config),
            yt_operation_spec=_choose_value(args_value=yt_operation_spec, config_value=yt_operation_spec),
            yt_task_spec=_choose_value(args_value=yt_task_spec, config_value=yt_task_spec),
            local=_choose_value(args_value=args["local"], config_value=config.local),
            bind=bind,
            command=command,
            mesh=EffectiveMeshConfig(
                node_count=_choose_value(
                    args_value=args["mesh.node_count"],
                    config_value=config.mesh.node_count,
                ),
                process_per_node=_choose_value(
                    args_value=args["mesh.process_per_node"],
                    config_value=config.mesh.process_per_node,
                ),
                gpu_per_process=_choose_value(
                    args_value=args["mesh.gpu_per_process"],
                    config_value=config.mesh.gpu_per_process,
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
    parser = argparse.ArgumentParser(
        description="Tractorun",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--yt-path", help="YT workdir", type=str)
    parser.add_argument("--run-config-path", type=str, help="path to tractorun config", default=CONFIG_PATH_DEFAULT)
    parser.add_argument("--docker-image", type=str, help="docker image name", default=DEFAULT_DOCKER_IMAGE)
    parser.add_argument("--mesh.node-count", type=int, help="mesh node count", default=NNODES_DEFAULT)
    parser.add_argument(
        "--mesh.process-per-node", type=int, help="mesh process per node", default=NPROC_PER_NODE_DEFAULT
    )
    parser.add_argument("--mesh.gpu_per_process", type=int, help="mesh gpu per process", default=NGPU_PER_PROC_DEFAULT)
    parser.add_argument("--mesh.pool-trees", help="mesh pool trees", action="append")
    parser.add_argument("--resources.cpu-limit", type=int, help="cpu limit", default=CPU_LIMIT_DEFAULT)
    parser.add_argument("--resources.mem-limit", type=int, help="mem limit", default=MEM_LIMIT_DEFAULT)
    parser.add_argument("--user-config", type=str, help="json config that will be passed to the jobs")
    parser.add_argument("--yt-operation-spec", help="yt operation spec", type=str)
    parser.add_argument("--yt-task-spec", help="yt task spec", type=str)
    parser.add_argument("--local", type=bool, help="enable local run mode", default=LOCAL_DEFAULT)
    parser.add_argument(
        "--bind",
        type=str,
        action="append",
        help="bind mounts to be passed to the docker container",
        default=[],
    )
    parser.add_argument("--dump-effective-config", help="print effective configuration", action="store_true")
    parser.add_argument("command", nargs="*", help="command to run")

    args = vars(parser.parse_args())

    file_config_content = Config.load_yaml(args["run_config_path"]) if args["run_config_path"] else Config()
    effective_config = EffectiveConfig.configure(
        config=file_config_content,
        args=args,
    )

    binds = []
    for bind in effective_config.bind:
        source, destination = bind.split(":")
        binds.append(Bind(source=source, destination=destination))

    if args["dump_effective_config"]:
        print("Parsed args:")
        pprint.pprint(args)
        print("\nConfig from file:")
        pprint.pprint(attr.asdict(file_config_content))  # type: ignore
        print("\nEffective config:")
        pprint.pprint(attr.asdict(effective_config))  # type: ignore
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
        user_config=effective_config.user_config,
        yt_operation_spec=effective_config.yt_operation_spec,
        yt_task_spec=effective_config.yt_task_spec,
        local=effective_config.local,
    )


if __name__ == "__main__":
    main()
