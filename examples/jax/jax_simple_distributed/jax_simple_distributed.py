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
    mesh = Mesh(node_count=1, process_per_node=7, gpu_per_process=1, pool_trees=["gpu-h100"])
    run(
        task,
        backend=Tractorax(),
        yt_path="//home/gritukan/mnist/trainings/dense",
        mesh=mesh,
        docker_image="cr.ai.nebius.cloud/crnf2coti090683j5ssi/tractorun/tractorax_runtime:2024-06-18-17-42-54",
        resources=Resources(cpu_limit=4.0, memory_limit=64 * 1024**3),
    )
