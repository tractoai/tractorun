import base64
import json
import os
import pickle
import socket
from typing import (
    Any,
    Dict,
)

from yt import wrapper as yt

from tractorun.checkpoints import CheckpointManager
from tractorun.coordinator import Coordinator
from tractorun.job_client import JobClient
from tractorun.mesh import Mesh


def prepare_environment(user_config: Dict[Any, Any]) -> JobClient:
    # Runs in a job

    config_path = os.environ["TRACTO_CONFIG"]
    with open(config_path, "r") as ff:
        config = json.load(ff)

    port = int(config["port"])
    path = config["path"]
    self_endpoint = socket.gethostname() + ":" + str(port)
    mesh = Mesh(int(config["nnodes"]), int(config["nproc"]), int(config["ngpu_per_proc"]))
    yt_cli = yt.YtClient(config=pickle.loads(base64.b64decode(config["yt_client_config"])))
    coordinator = Coordinator(
        yt_cli=yt_cli,
        path=path,
        self_endpoint=self_endpoint,
        mesh=mesh,
        node_index=int(config["node_index"]),
        process_index=int(config["proc_index"]),
    )
    # TODO: make CoordinatorFactory instead of the mystical prepare method
    coordinator.prepare()

    checkpoint_manager = CheckpointManager(path + "/checkpoints", yt_cli)
    # TODO: make CheckpointFactory instead of the mystical initialize method
    checkpoint_manager.initialize()
    # TODO: coordinator should be with prerequisites
    job_client = JobClient(coordinator, checkpoint_manager, yt_cli, user_config=user_config)

    ep = coordinator.get_primary_endpoint()
    os.environ["MASTER_ADDR"] = ep.split(":")[0]
    os.environ["MASTER_PORT"] = ep.split(":")[1]
    os.environ["WORLD_SIZE"] = str(coordinator.get_total_peer_count())
    os.environ["NODE_RANK"] = str(coordinator.get_self_index() // mesh.process_per_node)
    os.environ["LOCAL_RANK"] = str(coordinator.get_self_index() % mesh.process_per_node)

    return job_client
