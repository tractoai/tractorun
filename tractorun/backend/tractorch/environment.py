import torch
import torch.distributed

from tractorun.closet import Closet
from tractorun.environment import prepare_environment as common_prepare_environment


def prepare_environment(closet: Closet) -> None:
    common_prepare_environment(closet)

    if closet.mesh.gpu_per_process > 0:
        assert torch.cuda.is_available()

        if closet.mesh.gpu_per_process > 1:
            raise RuntimeError("not supported")

        device_index = closet.coordinator.get_process_index()
        assert device_index < torch.cuda.device_count()
        torch.cuda.set_device(closet.coordinator.get_process_index())

    backend = "gloo" if closet.mesh.gpu_per_process == 0 else "nccl"
    torch.distributed.init_process_group(
        backend=backend,
        init_method="tcp://" + closet.coordinator.get_primary_endpoint(),
        rank=closet.coordinator.get_self_index(),
        world_size=closet.coordinator.get_total_peer_count(),
    )
