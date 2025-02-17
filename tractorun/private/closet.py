import base64
import json
import os
import pickle
import socket

import attrs
from yt.wrapper import YtClient

from tractorun.checkpoint import CheckpointManager
from tractorun.coordinator import Coordinator
from tractorun.description import DescriptionManager
from tractorun.mesh import Mesh
from tractorun.private.constants import TRACTO_CONFIG_ENV_VAR
from tractorun.private.coordinator import CoordinatorFactory
from tractorun.private.helpers import AttrSerializer
from tractorun.private.training_dir import TrainingDir
from tractorun.private.worker import WorkerConfig
from tractorun.private.yt_cluster import TractorunClusterConfig


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
    checkpoint_manager: CheckpointManager
    training_dir: TrainingDir
    training_metadata: TrainingMetadata
    cluster_config: TractorunClusterConfig
    description_manager: DescriptionManager


def get_closet() -> Closet:
    config_path = os.environ[TRACTO_CONFIG_ENV_VAR]
    with open(config_path, "r") as ff:
        deserializer = AttrSerializer(WorkerConfig)
        config: WorkerConfig = deserializer.deserialize(json.load(ff))

    yt_client = YtClient(
        config=pickle.loads(
            base64.b64decode(config.yt_client_config),
        ),
    )
    training_metadata = TrainingMetadata(
        operation_id=config.operation_id,
        job_id=config.job_id,
    )
    description_manager = DescriptionManager(
        operation_id=training_metadata.operation_id,
        yt_client=yt_client,
        cypress_link_template=config.cluster_config.cypress_link_template,
    )
    coordinator = CoordinatorFactory(
        yt_client=yt_client,
        training_dir=config.training_dir,
        self_endpoint=f"{get_hostname()}:{config.port}",
        mesh=config.mesh,
        process_index=config.proc_index,
        node_index=config.node_index,
        self_index=config.self_index,
        operation_id=training_metadata.operation_id,
        job_id=training_metadata.job_id,
    ).create()

    checkpoint_manager = CheckpointManager(config.training_dir.checkpoints_path, yt_client)
    # TODO: make CheckpointFactory instead of the mystical initialize method
    checkpoint_manager.initialize()

    return Closet(
        mesh=config.mesh,
        coordinator=coordinator,
        yt_client=yt_client,
        training_dir=config.training_dir,
        training_metadata=training_metadata,
        checkpoint_manager=checkpoint_manager,
        cluster_config=config.cluster_config,
        description_manager=description_manager,
    )


def get_hostname() -> str:
    return socket.getaddrinfo(socket.gethostname(), 0, flags=socket.AI_CANONNAME)[0][3]
