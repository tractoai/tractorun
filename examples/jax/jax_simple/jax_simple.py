import argparse
import os
from pathlib import Path
import random
import string
import sys

from jax import (
    grad,
    jit,
)

from tractorun.backend.tractorax import Tractorax
from tractorun.mesh import Mesh
from tractorun.resources import Resources
from tractorun.run import run
from tractorun.toolbox import Toolbox


def task(_: Toolbox) -> None:
    @jit
    def f(x: int) -> int:
        return x**2 + 3 * x + 1

    grad_f = grad(f)
    print(grad_f(1.0), file=sys.stderr)


def _default_home_dir() -> str:
    rnm = "".join(random.choices(string.ascii_letters + string.digits, k=8))
    return f"//tmp/tractorun_examples/{rnm}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yt-home-dir", type=str, default=_default_home_dir())
    parser.add_argument("--pool-tree", type=str, default="default")
    parser.add_argument("--dataset-path", type=str, default="//home/samples/mnist-torch-train")
    parser.add_argument("--docker-image", type=str, default=os.environ.get("DOCKER_IMAGE"))
    parser.add_argument("--gpu-per-process", type=int, default=0)
    args = parser.parse_args()

    mesh = Mesh(node_count=1, process_per_node=8, gpu_per_process=args.gpu_per_process, pool_trees=[args.pool_tree])

    tractorun_path = (Path(__file__).parent.parent.parent.parent / "tractorun").resolve()
    run(
        task,
        backend=Tractorax(),
        yt_path=args.yt_home_dir,
        mesh=mesh,
        docker_image=args.docker_image,
        binds_local_lib=[str(tractorun_path)],
        resources=Resources(cpu_limit=4.0, memory_limit=16 * 1024**3),
    )
