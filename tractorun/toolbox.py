import json
import os
from typing import Any

import attrs
import yt.wrapper as yt

from tractorun.checkpoint import CheckpointManager
from tractorun.coordinator import Coordinator
from tractorun.description import DescriptionManager
from tractorun.mesh import Mesh
from tractorun.private import constants as _constants
from tractorun.private.closet import TrainingMetadata as _TrainingMetadata
from tractorun.private.training_dir import TrainingDir as _TrainingDir


__all__ = ["Toolbox"]


@attrs.define
class Toolbox:
    coordinator: Coordinator
    checkpoint_manager: CheckpointManager
    description_manager: DescriptionManager
    yt_client: yt.YtClient
    mesh: Mesh
    _training_dir: _TrainingDir
    _training_metadata: _TrainingMetadata

    @staticmethod
    def get_user_config() -> dict[Any, Any]:
        return json.loads(os.environ[_constants.YT_USER_CONFIG_ENV_VAR])

    def save_model(self, data: bytes, dataset_path: str, metadata: dict[str, str]) -> str:
        incarnation_id = self.coordinator.get_incarnation_id()
        path = self._training_dir.models_path + f"/{incarnation_id}"

        if not self.coordinator.is_primary():
            return path

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
