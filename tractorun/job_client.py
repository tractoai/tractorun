from typing import (
    Any,
    Dict,
)

import attrs
import torch
import torch.cuda
import torch.distributed as dist
import yt.wrapper as yt

from tractorun.checkpoints import CheckpointManager
from tractorun.coordinator import Coordinator
from tractorun.mesh import Mesh


@attrs.define
class JobClient:
    coordinator: Coordinator
    checkpoint_manager: CheckpointManager
    yt_client: yt.YtClient
    user_config: Dict[Any, Any]

    def initialize(self) -> None:
        self.coordinator.prepare()
        mesh = self.coordinator.get_mesh()
        if mesh.gpu_per_process > 0:
            assert torch.cuda.is_available()

            if mesh.gpu_per_process > 1:
                raise RuntimeError("not supported")

            device_index = self.coordinator.get_process_index()
            assert device_index < torch.cuda.device_count()
            torch.cuda.set_device(self.coordinator.get_process_index())

        backend = "gloo" if mesh.gpu_per_process == 0 else "nccl"
        dist.init_process_group(
            backend=backend,
            init_method="tcp://" + self.coordinator.get_primary_endpoint(),
            rank=self.coordinator.get_self_index(),
            world_size=self.coordinator.get_total_peer_count(),
        )

        self.checkpoint_manager.initialize()

    def get_mesh(self) -> Mesh:
        return self.coordinator.get_mesh()
