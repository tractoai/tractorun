import argparse
import os
import sys

import jax

from tractorun.backend.tractorax import Tractorax
from tractorun.mesh import Mesh
from tractorun.resources import Resources
from tractorun.run import run
from tractorun.toolbox import Toolbox


def task(_: Toolbox) -> None:
    print("Total device count:", jax.device_count(), file=sys.stderr)
    print("Local device count:", jax.local_device_count(), file=sys.stderr)

    xs = jax.numpy.ones(jax.local_device_count())
    print(jax.pmap(lambda x: jax.lax.psum(x, "i"), axis_name="i")(xs), file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yt-home-dir", type=str, required=True)
    parser.add_argument("--pool-tree", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, default="//home/samples/mnist-torch-train")
    parser.add_argument("--docker-image", type=str, default=os.environ.get("DOCKER_IMAGE"))
    parser.add_argument("--gpu-per-process", type=int, default=0)
    args = parser.parse_args()

    mesh = Mesh(node_count=1, process_per_node=8, gpu_per_process=args.gpu_per_process, pool_trees=[args.pool_tree])
    run(
        task,
        backend=Tractorax(),
        yt_path=args.yt_home_dir,
        mesh=mesh,
        docker_image=args.docker_image,
        resources=Resources(cpu_limit=4.0, memory_limit=64 * 1024**3),
        binds_local_lib=["../../../tractorun"],
    )
