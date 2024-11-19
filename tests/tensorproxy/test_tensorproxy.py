import json

from tests.utils import (
    TENSORPROXY_DOCKER_IMAGE,
    TractoCli,
    get_data_path,
    run_config_file,
)
from tests.yt_instances import YtInstance


def test_run_script(yt_instance: YtInstance, yt_path: str) -> None:
    yt_client = yt_instance.get_client()

    tracto_cli = TractoCli(
        command=["python3", "/tractorun_tests/tensorproxy_script.py"],
        docker_image=TENSORPROXY_DOCKER_IMAGE,
        args=[
            "--mesh.gpu-per-process",
            "0",
            "--tensorproxy.enabled",
            "1",
            "--resources.memory-limit",
            "428000000000",
            "--yt-path",
            yt_path,
            "--user-config",
            json.dumps({"use_ocdbt": False, "use_zarr3": False, "checkpoint_path": yt_path}),
            "--bind-local",
            f"{get_data_path('../data/tensorproxy_script.py')}:/tractorun_tests/tensorproxy_script.py",
        ],
    )
    op_run = tracto_cli.run()
    assert op_run.is_exitcode_valid()
    assert op_run.is_operation_state_valid(yt_client=yt_client, job_count=1)


def test_run_script_with_config(yt_instance: YtInstance, yt_path: str) -> None:
    yt_client = yt_instance.get_client()

    run_config = {
        "mesh": {
            "node_count": 1,
            "process_per_node": 1,
            "gpu_per_process": 0,
        },
        "tensorproxy": {
            "enabled": True,
        },
        "user_config": {
            "use_ocdbt": False,
            "use_zarr3": False,
            "checkpoint_path": yt_path,
        },
    }
    with run_config_file(run_config) as run_config_path:
        tracto_cli = TractoCli(
            command=["python3", "/tractorun_tests/tensorproxy_script.py"],
            docker_image=TENSORPROXY_DOCKER_IMAGE,
            args=[
                "--run-config-path",
                run_config_path,
                "--mesh.gpu-per-process",
                "0",
                "--resources.memory-limit",
                "428000000000",
                "--yt-path",
                yt_path,
                "--user-config",
                json.dumps({"use_ocdbt": False, "use_zarr3": False, "checkpoint_path": yt_path}),
                "--bind-local",
                f"{get_data_path('../data/tensorproxy_script.py')}:/tractorun_tests/tensorproxy_script.py",
            ],
        )
        op_run = tracto_cli.run()
    assert op_run.is_exitcode_valid()
    assert op_run.is_operation_state_valid(yt_client=yt_client, job_count=1)
