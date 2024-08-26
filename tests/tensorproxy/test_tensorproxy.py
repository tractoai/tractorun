import json

from tests.utils import (
    TractoCli,
    get_data_path,
    run_config_file,
)
from tests.yt_instances import YtInstance
from tractorun.stderr_reader import StderrMode


DOCKER_IMAGE = "cr.ai.nebius.cloud/crnf2coti090683j5ssi/tractorun/tensorproxy_tests:2024-10-14-15-02-56"


def test_run_script(yt_instance_with_tensorproxy: YtInstance, yt_path_with_tensorproxy: str) -> None:
    yt_client = yt_instance_with_tensorproxy.get_client()

    tracto_cli = TractoCli(
        command=["python3", "/tractorun_tests/tensorproxy_script.py"],
        docker_image=DOCKER_IMAGE,
        args=[
            "--mesh.gpu-per-process",
            "0",
            "--tensorproxy.enabled",
            "1",
            "--resources.memory-limit",
            "820000000",
            "--yt-path",
            yt_path_with_tensorproxy,
            "--user-config",
            json.dumps({"use_ocdbt": False, "use_zarr3": False, "checkpoint_path": yt_path_with_tensorproxy}),
            "--bind-local",
            f"{get_data_path('../data/tensorproxy_script.py')}:/tractorun_tests/tensorproxy_script.py",
            "--proxy-stderr-mode",
            StderrMode.primary,
        ],
    )
    op_run = tracto_cli.run()
    assert op_run.is_exitcode_valid()
    assert op_run.is_operation_state_valid(yt_client=yt_client, job_count=1)


# trigger tests 2
def test_run_script_with_config(yt_instance_with_tensorproxy: YtInstance, yt_path_with_tensorproxy: str) -> None:
    yt_client = yt_instance_with_tensorproxy.get_client()

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
            "checkpoint_path": yt_path_with_tensorproxy,
        },
    }
    with run_config_file(run_config) as run_config_path:
        tracto_cli = TractoCli(
            command=["python3", "/tractorun_tests/tensorproxy_script.py"],
            docker_image=DOCKER_IMAGE,
            args=[
                "--run-config-path",
                run_config_path,
                "--mesh.gpu-per-process",
                "0",
                "--resources.memory-limit",
                "820000000",
                "--yt-path",
                yt_path_with_tensorproxy,
                "--user-config",
                json.dumps({"use_ocdbt": False, "use_zarr3": False, "checkpoint_path": yt_path_with_tensorproxy}),
                "--bind-local",
                f"{get_data_path('../data/tensorproxy_script.py')}:/tractorun_tests/tensorproxy_script.py",
                "--proxy-stderr-mode",
                StderrMode.primary,
            ],
        )
        op_run = tracto_cli.run()
    assert op_run.is_exitcode_valid()
    assert op_run.is_operation_state_valid(yt_client=yt_client, job_count=1)
