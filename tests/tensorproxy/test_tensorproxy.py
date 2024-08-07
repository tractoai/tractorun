import json

from tests.utils import (
    TractoCli,
    get_data_path,
)
from tests.yt_instances import YtInstance


DOCKER_IMAGE = "cr.ai.nebius.cloud/crnf2coti090683j5ssi/tractorun/tensorproxy_tests:2024-08-06-22-05-28"


def test_run_script(yt_instance: YtInstance, yt_path: str) -> None:
    yt_client = yt_instance.get_client()

    tracto_cli = TractoCli(
        command=["python3", "/tractorun_tests/tensorproxy_script.py"],
        docker_image=DOCKER_IMAGE,
        args=[
            "--mesh.node-count",
            "1",
            "--mesh.process-per-node",
            "1",
            "--mesh.gpu-per-process",
            "0",
            "--resources.memory-limit",
            "428000000000",
            "--tensorproxy.enabled",
            "1",
            "--yt-path",
            yt_path,
            "--user-config",
            json.dumps({"use_ocdbt": False, "use_zarr3": False, "checkpoint_path": yt_path}),
            "--bind-local",
            f"{get_data_path('../data/tensorproxy_script.py')}:/tractorun_tests",
        ],
    )
    op_run = tracto_cli.run()
    assert op_run.is_exitcode_valid()
    assert op_run.is_operation_state_valid(yt_client=yt_client, job_count=1)
