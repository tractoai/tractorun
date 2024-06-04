from typing import (
    Any,
    Dict,
)

import torch
import torch.distributed

from tractorun.environment import prepare_environment as common_prepare_environment
from tractorun.job_client import JobClient


def prepare_environment(user_config: Dict[Any, Any]) -> JobClient:
    job_client = common_prepare_environment(user_config)

    mesh = job_client.get_mesh()
    coordinator = job_client.coordinator

    if mesh.gpu_per_process > 0:
        assert torch.cuda.is_available()

        if mesh.gpu_per_process > 1:
            raise RuntimeError("not supported")

        device_index = coordinator.get_process_index()
        assert device_index < torch.cuda.device_count()
        torch.cuda.set_device(coordinator.get_process_index())

    backend = "gloo" if mesh.gpu_per_process == 0 else "nccl"
    torch.distributed.init_process_group(
        backend=backend,
        init_method="tcp://" + coordinator.get_primary_endpoint(),
        rank=coordinator.get_self_index(),
        world_size=coordinator.get_total_peer_count(),
    )

    return job_client
