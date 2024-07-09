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


if __name__ == "__main__":
    mesh = Mesh(node_count=1, process_per_node=1, gpu_per_process=0, pool_trees=["gpu_h100"])
    run(
        task,
        backend=Tractorax(),
        yt_path="//home/gritukan/mnist/trainings/dense",
        mesh=mesh,
        docker_image="cr.ai.nebius.cloud/crnf2coti090683j5ssi/tractorun/tractorax_runtime:2024-06-18-16-13-57",
        resources=Resources(cpu_limit=4.0, memory_limit=16 * 1024**3),
        local=True,
    )
