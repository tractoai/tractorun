import base64
import json
import os
import pickle
import socket

import attr
from yt.wrapper import YtClient

from tractorun.coordinator import Coordinator
from tractorun.mesh import Mesh


@attr.define
class Closet:
    mesh: Mesh
    coordinator: Coordinator
    yt_cli: YtClient
    yt_path: str


def get_closet() -> Closet:
    config_path = os.environ["TRACTO_CONFIG"]
    with open(config_path, "r") as ff:
        config = json.load(ff)

    port = int(config["port"])
    path = config["path"]
    self_endpoint = socket.gethostname() + ":" + str(port)
    mesh = Mesh(int(config["nnodes"]), int(config["nproc"]), int(config["ngpu_per_proc"]))
    yt_cli = YtClient(config=pickle.loads(base64.b64decode(config["yt_client_config"])))
    coordinator = Coordinator.create(
        yt_client=yt_cli,
        yt_path=path,
        self_endpoint=self_endpoint,
        mesh=mesh,
        node_index=int(config["node_index"]),
        process_index=int(config["proc_index"]),
        operation_id=os.environ["YT_OPERATION_ID"],
        job_id=os.environ["YT_JOB_ID"],
    )

    return Closet(mesh=mesh, coordinator=coordinator, yt_cli=yt_cli, yt_path=path)
