from typing import (
    Any,
    Dict,
)

import attrs
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

    def get_mesh(self) -> Mesh:
        return self.coordinator.get_mesh()
