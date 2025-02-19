import uuid

import attrs
import yt.wrapper as yt


DEFAULT_TMP_WORKING_DIR = "//tmp/tractorun"


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class TrainingDir:
    base_path: str
    incarnations_path: str
    models_path: str
    checkpoints_path: str
    primary_lock_path: str
    logs_path: str
    worker_logs_path: str
    sidecar_logs_path: str

    @classmethod
    def create(cls, path: str) -> "TrainingDir":
        return TrainingDir(
            base_path=path,
            primary_lock_path=path + "/primary_lock",
            incarnations_path=path + "/incarnations",
            models_path=path + "/models",
            checkpoints_path=path + "/checkpoints",
            logs_path=path + "/logs",
            worker_logs_path=path + f"/_raw_logs/{uuid.uuid4()}",
            sidecar_logs_path=path + f"/_raw_logs/{uuid.uuid4()}",
        )

    def get_incarnation_path(self, incarnation_id: int) -> str:
        return self.incarnations_path + f"/{incarnation_id:05d}"


def prepare_training_dir(training_dir: TrainingDir, yt_client: yt.YtClient) -> None:
    yt_client.create(
        "map_node", training_dir.base_path, attributes={"incarnation_id": -1}, recursive=True, ignore_existing=True
    )
    yt_client.create("map_node", training_dir.primary_lock_path, ignore_existing=True)
    yt_client.create("map_node", training_dir.incarnations_path, ignore_existing=True)
    yt_client.create("map_node", training_dir.models_path, ignore_existing=True)
    yt_client.create("map_node", training_dir.checkpoints_path, ignore_existing=True)
    yt_client.create("map_node", training_dir.logs_path, ignore_existing=True)
    yt_client.create("map_node", training_dir.worker_logs_path, recursive=True, ignore_existing=True)
    yt_client.create("map_node", training_dir.sidecar_logs_path, recursive=True, ignore_existing=True)
