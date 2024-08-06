import os
import subprocess
import sys
import time
import typing

import ytpath  # type: ignore
import orbax.checkpoint as ocp
import numpy as np

# try to use jax
from tractorun.backend.generic import GenericBackend
from tractorun.run import prepare_and_get_toolbox

grpc_address = 'localhost:20402'


YSON_CONFIG = """
{
    "monitoring_port" = 20401;
    "grpc_port" = 20402;
    "store_kind" = "cypress";
    "cypress_store" = {
        "cluster_url" = "planck.yt.nemax.nebiuscloud.net";
        "file_writer" = {
            block_size = 262144;
        }
    };
    logging={
        "flush_period"=100;
        rules=[
            {
                "exclude_categories"=[
                    Bus;
                ];
                family="plain_text";
                "min_level"=debug;
                writers=[
                    debug;
                ];
            };
            {
                family="plain_text";
                "min_level"=info;
                writers=[
                    info;
                ];
            };
            {
                family="plain_text";
                "min_level"=error;
                writers=[
                    error;
                ];
            };
        ];
        writers={
            debug={
                "enable_system_messages"=%true;
                "file_name"="./tensorproxy.debug.log";
                format="plain_text";
                "rotation_policy"={
                    "max_segment_count_to_keep"=1000;
                    "max_total_size_to_keep"=130000000000;
                    "rotation_period"=900000;
                };
                type=file;
            };
            error={
                "enable_system_messages"=%true;
                format="plain_text";
                type=stderr;
            };
            info={
                "enable_system_messages"=%true;
                "file_name"="./tensorproxy.info.log";
                format="plain_text";
                "rotation_policy"={
                    "max_segment_count_to_keep"=1000;
                    "max_total_size_to_keep"=130000000000;
                    "rotation_period"=900000;
                };
                type=file;
            };
        };
    };
}

"""


def timed_run(action: typing.Callable) -> None:
    import time
    start = time.time()
    action()
    print(f"Time: {time.time() - start:.2f}s", file=sys.stderr)


def main() -> None:
    os.environ["TS_GRPC_ADDRESS"] = grpc_address
    toolbox = prepare_and_get_toolbox(backend=GenericBackend())
    yt_client = toolbox.yt_client
    tensorproxy_data = yt_client.read_file("//home/yt-team/chiffa/tensorproxy")
    with open('./tensorproxy', 'wb') as f:
        f.write(tensorproxy_data.read())
    os.chmod("./tensorproxy", 0o755)
    assert os.path.isfile("./tensorproxy")

    with open("config.yson", "w") as f:
        f.write(YSON_CONFIG)

    user_config = toolbox.get_user_config()
    use_ocdbt = user_config["use_ocdbt"]
    use_zarr3 = user_config["use_zarr3"]
    process = subprocess.Popen(
        ["./tensorproxy", "--config", "config.yson"],
        stdout=sys.stderr,
        stderr=sys.stderr,
        bufsize=1,
        universal_newlines=True,
        env={
            **os.environ,
        },
    )
    time.sleep(10)

    checkpoint_path = f"yt://{user_config['checkpoint_path']}/tensorproxy"

    rand_tensor = np.random.rand(25 * 1024**2).astype(np.float32)
    gen_tree = lambda: [{
        't' + str(j): rand_tensor for j in range(4)
    } for _ in range(4)]

    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler(use_ocdbt=use_ocdbt, use_zarr3=use_zarr3))
    print("Generating", file=sys.stderr)
    my_tree = gen_tree()
    print("Saving")
    timed_run(lambda: checkpointer.save(checkpoint_path, my_tree))

    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler(use_ocdbt=use_ocdbt, use_zarr3=use_zarr3))
    print("Restoring", file=sys.stderr)
    timed_run(lambda: checkpointer.restore(checkpoint_path))
    process.terminate()


if __name__ == "__main__":
    main()
