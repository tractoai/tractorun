#!/usr/bin/env python3

import argparse
import json

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
    parser.add_argument("training_script")

    args = parser.parse_args()

    mesh = Mesh(node_count=args.nnodes, process_per_node=args.nproc_per_node, gpu_per_process=args.ngpu_per_proc)
    user_config = json.loads(args.user_config) if args.user_config is not None else None

    run_script(
        training_script=args.training_script,
        mesh=mesh,
        yt_path=args.yt_path,
        docker_image=args.docker_image,
        user_config=user_config,
    )


if __name__ == "__main__":
    main()
