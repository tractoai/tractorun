import json as _json
import os as _os
from typing import Any as _Any

import attrs
import yt.wrapper as _yt

from tractorun.checkpoint import CheckpointManager as _CheckpointManager
from tractorun.mesh import Mesh as _Mesh
from tractorun.private import constants as _constants
from tractorun.private.closet import TrainingMetadata as _TrainingMetadata
from tractorun.private.coordinator import Coordinator as _Coordinator
from tractorun.private.training_dir import TrainingDir as _TrainingDir


@attrs.define
class Toolbox:
    coordinator: _Coordinator
    checkpoint_manager: _CheckpointManager
    yt_client: _yt.YtClient
    mesh: _Mesh
    _training_dir: _TrainingDir
    _training_metadata: _TrainingMetadata

    @staticmethod
    def get_user_config() -> dict[_Any, _Any]:
        return _json.loads(_os.environ[_constants.YT_USER_CONFIG_ENV_VAR])

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
