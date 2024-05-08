import yt.wrapper as yt

from .coordinator import Coordinator
from .checkpoints import CheckpointManager

import torch
import torch.cuda

import typing as tp

import torch.distributed as dist


class JobClient:
    def __init__(self, coordinator: Coordinator, checkpoint_manager: CheckpointManager, yt_client: yt.YtClient):
        self.coordinator = coordinator
        self.checkpoint_manager = checkpoint_manager
        self.yt_client = yt_client

    def initialize(self):
        self.coordinator.prepare()
        mesh = self.coordinator.get_mesh()
        if mesh.gpu_per_process > 0:
            assert torch.cuda.is_available()

            if mesh.gpu_per_process > 1:
                raise RuntimeError('not supported')

            device_index = self.coordinator.get_process_index()     
            assert device_index < torch.cuda.device_count()       
            torch.cuda.set_device(torch.cuda.device(self.coordinator.get_process_index()))

        backend = 'gloo' if mesh.gpu_per_process == 0 else 'nccl'
        dist.init_process_group(
            backend=backend,
            init_method='tcp://' + self.coordinator.get_primary_endpoint(),
            rank=self.coordinator.get_self_index(),
            world_size=self.coordinator.get_total_peer_count(),
        )

        self.checkpoint_manager.initialize()

    def get_mesh(self):
        return self.coordinator.get_mesh()
