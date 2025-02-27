import sys

from tests.utils import RAY_DOCKER_IMAGE
from tests.yt_instances import YtInstance
from tractorun.backend.ray import Ray
from tractorun.mesh import Mesh
from tractorun.resources import Resources
from tractorun.run import run
from tractorun.toolbox import Toolbox


def test_run_pickle(yt_instance: YtInstance, yt_path: str, mnist_ds_path: str) -> None:
    yt_client = yt_instance.get_client()

    def train_func(toolbox: Toolbox) -> None:
        import time

        import ray

        print("Hello from train_func", file=sys.stderr)
        if toolbox.coordinator.is_primary():
            ray.init(address="auto", ignore_reinit_error=True, logging_level=1)
            print(ray.nodes(), file=sys.stderr)

            @ray.remote(scheduling_strategy="SPREAD")
            def remote_task(i: int) -> str:
                time.sleep(2)
                return f"Task {i} has been completed on host {ray.get_runtime_context().node_id}"

            tasks = [remote_task.remote(i) for i in range(10)]

            results = ray.get(tasks)
            for result in results:
                print(result, file=sys.stderr)
        else:
            ray.init(address="auto", ignore_reinit_error=True, logging_level=1)
        print("End", file=sys.stderr)

    mesh = Mesh(node_count=2, process_per_node=1, gpu_per_process=8, pool_trees=["gpu_h200"])
    # The operation did not fail => success!
    run(
        train_func,
        backend=Ray(),
        yt_path=yt_path,
        mesh=mesh,
        yt_client=yt_client,
        docker_image=RAY_DOCKER_IMAGE,
        resources=Resources(cpu_limit=100, memory_limit=17179869184),
        env=[
            # EnvVariable(
            #     name="RAY_BACKEND_LOG_LEVEL",
            #     value="debug",
            # ),
            # EnvVariable(
            #     name="GRPC_VERBOSITY",
            #     value="debug",
            # ),
        ],
    )


# def test_run_script(yt_instance: YtInstance, yt_path: str, mnist_ds_path: str) -> None:
#     yt_client = yt_instance.get_client()
#
#     tracto_cli = TractoCli(
#         command=["python3", "/tractorun_tests/torch_run_script.py"],
#         docker_image=TRACTORCH_DOCKER_IMAGE,
#         args=[
#             "--mesh.node-count",
#             "1",
#             "--mesh.process-per-node",
#             "1",
#             "--mesh.gpu-per-process",
#             "0",
#             "--yt-path",
#             yt_path,
#             "--user-config",
#             json.dumps({"MNIST_DS_PATH": mnist_ds_path}),
#             "--bind-local",
#             f"{get_data_path('../data/torch_run_script.py')}:/tractorun_tests/torch_run_script.py",
#         ],
#     )
#     op_run = tracto_cli.run()
#     assert op_run.is_exitcode_valid()
#     assert op_run.is_operation_state_valid(yt_client=yt_client, job_count=1)
