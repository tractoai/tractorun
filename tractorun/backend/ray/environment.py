import logging
import os
import subprocess
import time
from urllib.parse import urlparse

from tractorun.base_backend import EnvironmentBase
from tractorun.private.closet import Closet as _Closet


__all__ = ["Environment"]


_LOGGER = logging.getLogger(__name__)


class Environment(EnvironmentBase):
    def prepare(self, closet: _Closet) -> None:
        coordinator_address = closet.coordinator.get_primary_endpoint()
        logging.debug("Coordinator address: %s", coordinator_address)
        parsed_coordinator_address = urlparse(f"schema://{coordinator_address}")
        # because of overlay problems
        original_coordinator_address = coordinator_address
        if coordinator_address == closet.coordinator.get_self_endpoint():
            coordinator_address = f"127.0.0.1:{parsed_coordinator_address.port}"
            logging.debug("Replace coordinator address %s by %s", original_coordinator_address, coordinator_address)
        with open("/etc/hosts", "a") as f:
            f.write(f"127.0.0.1\t{closet.coordinator.get_self_endpoint().split(':')[0]}")
        if closet.coordinator.is_primary():
            _LOGGER.info("Starting ray main process")
            command = [
                "ray",
                "start",
                "--head",
                "--port",
                str(parsed_coordinator_address.port),
                # "--node-ip-address",
                # "127.0.0.1",
                "--node-manager-port",
                os.environ["YT_PORT_2"],
                "--object-manager-port",
                os.environ["YT_PORT_3"],
                "--runtime-env-agent-port",
                os.environ["YT_PORT_4"],
                "--dashboard-port",
                os.environ["YT_PORT_5"],
                "--dashboard-agent-grpc-port",
                os.environ["YT_PORT_6"],
                "--dashboard-agent-listen-port",
                os.environ["YT_PORT_7"],
                "--ray-client-server-port",
                os.environ["YT_PORT_8"],
                "--worker-port-list",
                ",".join([os.environ[f"YT_PORT_{i}"] for i in range(9, 50)]),
                "--include-dashboard",
                "false",
                "--num-cpus",
                str(int(closet.resources.cpu_limit)),
                # "--temp-dir", "/slot/sandbox",
                "-v",
            ]
            _LOGGER.info("Master command %s", command)
            process = subprocess.Popen(command)
            process.wait()
            _LOGGER.info("Ray main process started")
        else:
            _LOGGER.info("Starting ray worker process")
            run_ray_worker(
                coordinator_address,
                closet.coordinator.get_self_endpoint().split(":")[0],
                cpu_limit=closet.resources.cpu_limit,
            )
        # time.sleep(20)


def check_ray() -> bool:
    while True:
        process = subprocess.Popen(
            ["ray", "status"],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        stdout, stderr = process.communicate()
        _LOGGER.info("Ray status stdout: %s", stdout)
        _LOGGER.info("Ray status stderr: %s", stderr)
        if process.returncode != 0:
            _LOGGER.info("Ray status failed")
            return False
        if b"Node status" in stdout:
            _LOGGER.info("Ray status ok!")
            return True
        time.sleep(5)
        _LOGGER.info("Check ray again")


def run_ray_worker(coordinator_address: str, node_address: str, cpu_limit: float) -> None:
    while True:
        command = [
            "ray",
            "start",
            "--node-ip-address",
            node_address,
            "--address",
            coordinator_address,
            "--node-manager-port",
            os.environ["YT_PORT_2"],
            "--object-manager-port",
            os.environ["YT_PORT_3"],
            "--runtime-env-agent-port",
            os.environ["YT_PORT_4"],
            "--ray-client-server-port",
            os.environ["YT_PORT_5"],
            "--dashboard-port",
            os.environ["YT_PORT_6"],
            "--dashboard-agent-grpc-port",
            os.environ["YT_PORT_7"],
            "--dashboard-agent-listen-port",
            os.environ["YT_PORT_8"],
            "--worker-port-list",
            ",".join([os.environ[f"YT_PORT_{i}"] for i in range(9, 50)]),
            "--num-cpus",
            str(int(cpu_limit)),
            "-v",
        ]
        _LOGGER.info("Worker command %s", command)
        process = subprocess.Popen(command)
        process.wait()
        assert process.returncode == 0
        if check_ray():
            break
