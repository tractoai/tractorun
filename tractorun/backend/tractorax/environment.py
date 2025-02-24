import logging
from urllib.parse import urlparse

import jax

from tractorun.base_backend import EnvironmentBase
from tractorun.private.closet import Closet as _Closet


__all__ = ["Environment"]


_LOGGER = logging.getLogger(__name__)


class Environment(EnvironmentBase):
    def prepare(self, closet: _Closet) -> None:
        local_device_ids = None
        if closet.mesh.gpu_per_process > 0:
            process_id = closet.coordinator.get_process_index()
            first_device_index = process_id * closet.mesh.gpu_per_process
            local_device_ids = list(range(first_device_index, first_device_index + closet.mesh.gpu_per_process))

        coordinator_address = closet.coordinator.get_primary_endpoint()
        _LOGGER.debug("Coordinator address: %s", coordinator_address)
        parsed_coordinator_address = urlparse(f"schema://{coordinator_address}")
        # because of overlay problems
        if parsed_coordinator_address.hostname == closet.coordinator.get_self_endpoint().split(":")[0]:
            old_coordinator_address = coordinator_address
            coordinator_address = f"127.0.0.1:{parsed_coordinator_address.port}"
            logging.debug("Replace coordinator address %s by %s", old_coordinator_address, coordinator_address)

        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=closet.coordinator.get_total_peer_count(),
            process_id=closet.coordinator.get_self_index(),
            local_device_ids=local_device_ids,
        )
