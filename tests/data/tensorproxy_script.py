import sys
import time
import typing

import numpy as np
import orbax.checkpoint as ocp

# ytpath patches other libs
# we have to keep this import
import ytpath  # noqa

# try to use jax
from tractorun.backend.generic import GenericBackend
from tractorun.run import prepare_and_get_toolbox


def timed_run(action: typing.Callable) -> None:

    start = time.time()
    action()
    print(f"Time: {time.time() - start:.2f}s", file=sys.stderr)


def gen_tree() -> list:
    rand_tensor = np.random.rand(25 * 1024**2).astype(np.float32)
    return [{"t" + str(j): rand_tensor for j in range(4)} for _ in range(4)]


def main() -> None:
    toolbox = prepare_and_get_toolbox(backend=GenericBackend())

    user_config = toolbox.get_user_config()
    use_ocdbt = user_config["use_ocdbt"]
    use_zarr3 = user_config["use_zarr3"]

    checkpoint_path = f"yt://{user_config['checkpoint_path']}/tensorproxy"

    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler(use_ocdbt=use_ocdbt, use_zarr3=use_zarr3))
    print("Generating", file=sys.stderr)
    my_tree = gen_tree()
    print("Saving")
    timed_run(lambda: checkpointer.save(checkpoint_path, my_tree))

    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler(use_ocdbt=use_ocdbt, use_zarr3=use_zarr3))
    print("Restoring", file=sys.stderr)
    timed_run(lambda: checkpointer.restore(checkpoint_path))


if __name__ == "__main__":
    main()
