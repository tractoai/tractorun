from typing import (
    Any,
    Dict,
)

import torch

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

    return job_client
