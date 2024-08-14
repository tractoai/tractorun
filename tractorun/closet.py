import base64
import json
import os
import pickle
import socket

import attrs
from yt.wrapper import YtClient

from tractorun.bootstrapper import ProcConfig
from tractorun.constants import TRACTO_CONFIG_ENV_VAR
from tractorun.coordinator import Coordinator
from tractorun.helpers import AttrSerializer
from tractorun.mesh import Mesh
from tractorun.training_dir import TrainingDir


@attrs.define
class TrainingMetadata:
    # TODO: choose a good place for this data
    operation_id: str
    job_id: str


@attrs.define
class Closet:
    mesh: Mesh
    coordinator: Coordinator
    yt_client: YtClient
    training_dir: TrainingDir
    training_metadata: TrainingMetadata


def get_closet() -> Closet:
    config_path = os.environ[TRACTO_CONFIG_ENV_VAR]
    with open(config_path, "r") as ff:
        deserializer = AttrSerializer(ProcConfig)
        config: ProcConfig = deserializer.deserialize(json.load(ff))

    self_endpoint = socket.gethostname() + ":" + str(config.port)
    yt_client = YtClient(
        config=pickle.loads(
            base64.b64decode(config.yt_client_config),
        ),
    )
    training_metadata = TrainingMetadata(
        operation_id=os.environ["YT_OPERATION_ID"],
        job_id=os.environ["YT_JOB_ID"],
    )
    coordinator = Coordinator.create(
        yt_client=yt_client,
        training_dir=config.training_dir,
        self_endpoint=self_endpoint,
        mesh=config.mesh,
        node_index=config.node_index,
        process_index=config.proc_index,
        operation_id=training_metadata.operation_id,
        job_id=training_metadata.job_id,
    )

    return Closet(
        mesh=config.mesh,
        coordinator=coordinator,
        yt_client=yt_client,
        training_dir=config.training_dir,
        training_metadata=training_metadata,
    )
