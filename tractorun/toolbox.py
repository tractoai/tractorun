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
from tractorun.closet import TrainingMetadata
from tractorun.coordinator import Coordinator
from tractorun.mesh import Mesh


@attrs.define
class Toolbox:
    coordinator: Coordinator
    checkpoint_manager: CheckpointManager
    yt_client: yt.YtClient
    mesh: Mesh
    _yt_path: str
    _training_metadata: TrainingMetadata

    @staticmethod
    def get_user_config() -> Dict[Any, Any]:
        return json.loads(os.environ[const.YT_USER_CONFIG_ENV_VAR])

    def save_model(self, data: bytes, dataset_path: str, metadata: Dict[str, str]) -> str:
        incarnation_id = self.coordinator.get_incarnation_id()
        # TODO: refactor path creation
        models_path = self._yt_path + "/models"
        path = models_path + f"/{incarnation_id}"

        if not self.coordinator.is_primary():
            return path

        self.yt_client.create("map_node", models_path, ignore_existing=True)
        self.yt_client.write_file(path, data)
        attributes = {
            "dataset_path": dataset_path,
            "operation_id": self._training_metadata.operation_id,
            "job_id": self._training_metadata.job_id,
            "incarnation_id": incarnation_id,
            "metadata": metadata,
            "user_config": self.get_user_config(),
            "mesh": attrs.asdict(self.mesh),  # type: ignore
        }
        self.yt_client.set_attribute(
            path=path,
            attribute="tractorun",
            value=attributes,
        )
        return path
