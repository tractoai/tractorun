#!/usr/bin/env python3

import argparse
import json

from tractorun.bind import Bind
from tractorun.mesh import Mesh
from tractorun.run import run_script


def main() -> None:
    parser = argparse.ArgumentParser(description="Tractorun")
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--nproc_per_node", type=int, default=1)
    parser.add_argument("--ngpu_per_proc", type=int, default=0)
    parser.add_argument("--yt-path", type=str, required=True)
    parser.add_argument("--docker-image", type=str)
    parser.add_argument("--user-config", type=str, help="json config that will be passed to the jobs")
    parser.add_argument("--pool-trees", action="append", required=False)
    parser.add_argument("--yt-operation-spec", type=str, required=False)
    parser.add_argument("--yt-task-spec", type=str, required=False)
    parser.add_argument("--local", type=bool, default=False)
    parser.add_argument("--bind", type=str, action="append", help="bind mounts to be passed to the docker container")
    parser.add_argument("command", nargs="+", help="command to run")

    args = parser.parse_args()

    mesh = Mesh(
        node_count=args.nnodes,
        process_per_node=args.nproc_per_node,
        gpu_per_process=args.ngpu_per_proc,
        pool_trees=args.pool_trees,
    )
    user_config = json.loads(args.user_config) if args.user_config is not None else None
    yt_operation_spec = json.loads(args.yt_operation_spec) if args.yt_operation_spec is not None else None
    yt_task_spec = json.loads(args.yt_task_spec) if args.yt_task_spec is not None else None
    raw_binds = args.bind if args.bind is not None else []

    binds = []
    for bind in raw_binds:
        source, destination = bind.split(":")
        binds.append(Bind(source=source, destination=destination))

    run_script(
        command=args.command,
        mesh=mesh,
        yt_path=args.yt_path,
        docker_image=args.docker_image,
        binds=binds,
        user_config=user_config,
        yt_operation_spec=yt_operation_spec,
        yt_task_spec=yt_task_spec,
        local=args.local,
    )


if __name__ == "__main__":
    main()
