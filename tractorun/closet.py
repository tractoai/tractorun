import base64
import json
import os
import pickle
import socket

import attr
from yt.wrapper import YtClient

from tractorun.coordinator import Coordinator
from tractorun.mesh import (
    Mesh,
    MeshSerializer,
)


@attr.define
class TrainingMetadata:
    # TODO: choose a good place for this data
    operation_id: str
    job_id: str


@attr.define
class Closet:
    mesh: Mesh
    coordinator: Coordinator
    yt_client: YtClient
    yt_path: str
    training_metadata: TrainingMetadata


def get_closet() -> Closet:
    config_path = os.environ["TRACTO_CONFIG"]
    with open(config_path, "r") as ff:
        config = json.load(ff)

    port = int(config["port"])
    path = config["path"]
    self_endpoint = socket.gethostname() + ":" + str(port)
    mesh = MeshSerializer.deserialize(config["mesh"])
    yt_client = YtClient(config=pickle.loads(base64.b64decode(config["yt_client_config"])))
    training_metadata = TrainingMetadata(
        operation_id=os.environ["YT_OPERATION_ID"],
        job_id=os.environ["YT_JOB_ID"],
    )
    coordinator = Coordinator.create(
        yt_client=yt_client,
        yt_path=path,
        self_endpoint=self_endpoint,
        mesh=mesh,
        node_index=int(config["node_index"]),
        process_index=int(config["proc_index"]),
        operation_id=training_metadata.operation_id,
        job_id=training_metadata.job_id,
    )

    return Closet(
        mesh=mesh,
        coordinator=coordinator,
        yt_client=yt_client,
        yt_path=path,
        training_metadata=training_metadata,
    )
