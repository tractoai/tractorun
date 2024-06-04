import json
import os
from typing import (
    Any,
    Dict,
)

import attrs
import yt.wrapper as yt

from tractorun import constants as const
from tractorun.checkpoints import CheckpointManager
from tractorun.coordinator import Coordinator
from tractorun.mesh import Mesh


@attrs.define
class Toolbox:
    coordinator: Coordinator
    checkpoint_manager: CheckpointManager
    yt_client: yt.YtClient

    def get_mesh(self) -> Mesh:
        return self.coordinator.get_mesh()

    @staticmethod
    def get_user_config() -> Dict[Any, Any]:
        return json.loads(os.environ[const.YT_USER_CONFIG_ENV_VAR])
