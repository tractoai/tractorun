import base64
import json
import os
import pickle
import socket

import attrs
from yt.wrapper import YtClient

from tractorun.checkpoint import CheckpointManager
from tractorun.coordinator import Coordinator
from tractorun.mesh import Mesh
from tractorun.private.bootstrapper import ProcConfig
from tractorun.private.constants import TRACTO_CONFIG_ENV_VAR
from tractorun.private.coordinator import CoordinatorFactory
from tractorun.private.helpers import AttrSerializer
from tractorun.private.training_dir import TrainingDir
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
    coordinator = CoordinatorFactory(
        yt_client=yt_client,
        training_dir=config.training_dir,
        self_endpoint=self_endpoint,
        mesh=config.mesh,
        node_index=config.node_index,
        process_index=config.proc_index,
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
    )
