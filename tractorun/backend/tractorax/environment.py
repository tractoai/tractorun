import jax

from tractorun.base_backend import EnvironmentBase
from tractorun.private.closet import Closet as _Closet


__all__ = ["Environment"]


class Environment(EnvironmentBase):
    def prepare(self, closet: _Closet) -> None:
        local_device_ids = None
        if closet.mesh.gpu_per_process > 0:
            process_id = closet.coordinator.get_process_index()
            first_device_index = process_id * closet.mesh.gpu_per_process
            local_device_ids = list(range(first_device_index, first_device_index + closet.mesh.gpu_per_process))

        jax.distributed.initialize(
            coordinator_address=closet.coordinator.get_primary_endpoint(),
            num_processes=closet.coordinator.get_total_peer_count(),
            process_id=closet.coordinator.get_self_index(),
            local_device_ids=local_device_ids,
        )
