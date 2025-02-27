import atexit
import logging
import subprocess
import time
from typing import Callable
from urllib.parse import urlparse

import attrs
import ray
from ray.util import state as ray_state
from tractorun.base_backend import EnvironmentBase
from tractorun.exception import (
    TractorunBootstrapError,
    TractorunConfigurationError,
)
from tractorun.private.closet import Closet as _Closet


__all__ = ["Environment"]


RAY_CHECK_TIMEOUT = 5
_LOGGER = logging.getLogger(__name__)


class Environment(EnvironmentBase):
    def prepare(self, closet: _Closet) -> None:
        if closet.mesh.process_per_node != 1:
            raise TractorunConfigurationError("process per node should be == 1 for ray")
        coordinator_address = closet.coordinator.get_primary_endpoint()
        logging.debug("Coordinator address: %s", coordinator_address)
        with open("/etc/hosts", "a") as f:
            f.write(f"127.0.0.1\t{closet.coordinator.get_self_endpoint().split(':')[0]}")
        if closet.coordinator.is_primary():
            _LOGGER.info("Starting ray main process")
            self._run_main(closet)
            _LOGGER.info("Ray main process started")
        else:
            _LOGGER.info("Starting ray worker process")
            self._run_worker(
                closet,
            )
            _LOGGER.info("Ray worker process started")

    def _run_main(self, closet: _Closet) -> None:
        parsed_coordinator_address = urlparse(f"schema://{closet.coordinator.get_primary_endpoint()}")
        command = [
            "ray",
            "start",
            "--head",
            "--port",
            str(parsed_coordinator_address.port),
            "--include-dashboard",
            "false",
            "--num-cpus",
            str(int(closet.resources.cpu_limit)),
            "-v",
        ]
        _LOGGER.info("Master command %s", command)
        process = subprocess.Popen(command)
        process.wait()
        if process.returncode != 0:
            raise TractorunBootstrapError("Can't start head node")
        ray.init(address="auto")
        self._wait_for_nodes(closet)
        # ray.shutdown()
        atexit.register(stop_main_callback)

    def _run_worker(self, closet: _Closet) -> None:
        parsed_worker_address = urlparse(f"schema://{closet.coordinator.get_self_endpoint()}")
        assert parsed_worker_address.hostname
        exit_code = -1
        while exit_code != 0:
            command = [
                "ray",
                "start",
                "--node-ip-address",
                parsed_worker_address.hostname,
                "--address",
                closet.coordinator.get_primary_endpoint(),
                "--num-cpus",
                str(int(closet.resources.cpu_limit)),
                "-v",
            ]
            _LOGGER.info("Run worker command %s", command)
            process = subprocess.Popen(command)
            process.wait()
            exit_code = process.returncode
        ray.init(address="auto")
        self._wait_for_nodes(closet)
        # ray.shutdown()
        atexit.register(StopWorkerCallback(node_count=closet.mesh.node_count))

    def _wait_for_nodes(self, closet: _Closet) -> None:
        assert ray.is_initialized()
        nodes = ray.nodes()
        _LOGGER.info("Waiting for all nodes to join, now %s", nodes)
        assert len(nodes) <= closet.mesh.node_count
        while len(nodes) != closet.mesh.node_count or not all(map(lambda x: x["Alive"], nodes)):
            _LOGGER.info("Waiting for all nodes to join, now %s", nodes)
            nodes = ray.nodes()
            time.sleep(RAY_CHECK_TIMEOUT)


def stop_main_callback() -> None:
    if ray.is_initialized():
        ray.shutdown()
    command = [
        "ray",
        "stop",
        "--force"
    ]
    _LOGGER.info("Stopping head node %s", command)
    process = subprocess.Popen(command)
    process.wait()
    if process.returncode != 0:
        raise TractorunBootstrapError("Can't stop head node")
    _LOGGER.info("Head node stopped")


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class StopWorkerCallback:
    _node_count: int

    def __call__(self) -> None:
        _LOGGER.info("Stopping worker node node %s")
        try:
            # don't write warnings
            if not ray.is_initialized():
                ray.init(address="auto", ignore_reinit_error=True)
        except ConnectionError:
            _LOGGER.info("Ray is not running")
            return
        nodes = ray.nodes()
        while len(nodes) == self._node_count and all(map(lambda x: x["Alive"], nodes)):
            _LOGGER.info("Waiting for all nodes to join, now %s", nodes)
            nodes = ray.nodes()
            time.sleep(RAY_CHECK_TIMEOUT)
        ray.shutdown()
        _LOGGER.info("Worker node stopped")
