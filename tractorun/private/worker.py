import base64
import io
import json
import os
import pickle
import subprocess
from typing import (
    TYPE_CHECKING,
    NewType,
)

import attrs

from tractorun.mesh import Mesh
from tractorun.private.constants import TRACTO_CONFIG_ENV_VAR
from tractorun.private.helpers import AttrSerializer
from tractorun.private.training_dir import TrainingDir
from tractorun.private.yt_cluster import TractorunClusterConfig


WorkerIndex = NewType("WorkerIndex", int)


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class WorkerConfig:
    mesh: Mesh
    port: int
    node_index: int
    proc_index: int
    self_index: int
    training_dir: TrainingDir
    yt_client_config: str
    operation_id: str
    job_id: str
    cluster_config: TractorunClusterConfig


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class WorkerRun:
    worker_config: WorkerConfig
    _process: subprocess.Popen

    @classmethod
    def run(
        cls,
        command: list[str],
        mesh: Mesh,
        self_index: int,
        node_index: int,
        proc_index: int,
        port: int,
        training_dir: TrainingDir,
        cluster_config: TractorunClusterConfig,
        yt_config: dict,
        env: dict,
    ) -> "WorkerRun":
        worker_config = WorkerConfig(
            mesh=mesh,
            node_index=node_index,
            proc_index=proc_index,
            self_index=self_index,
            port=port,
            training_dir=training_dir,
            yt_client_config=base64.b64encode(pickle.dumps(yt_config)).decode("utf-8"),
            cluster_config=cluster_config,
            operation_id=env["YT_OPERATION_ID"],
            job_id=env["YT_JOB_ID"],
        )
        config_name = f"config_{proc_index}.json"
        with open(config_name, "w") as f:
            serializer = AttrSerializer(WorkerConfig)
            json.dump(serializer.serialize(worker_config), f)

        # TODO: torch multiprocessing is better (or another backend-specific tool),
        # but pickling does not work in this case.
        # torch.multiprocessing.spawn(wrapped_run_mp, nprocs=mesh.process_per_node, args=(f, c, path,), join=True)
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,
            env={
                TRACTO_CONFIG_ENV_VAR: config_name,
                "YT_PROXY": yt_config["proxy"]["url"],
                "YT_TOKEN": yt_config["token"],
                **env,
            },
        )
        if TYPE_CHECKING:
            assert isinstance(process.stdout, io.TextIOWrapper)
            assert isinstance(process.stderr, io.TextIOWrapper)
        os.set_blocking(process.stdout.fileno(), False)
        os.set_blocking(process.stderr.fileno(), False)
        return WorkerRun(
            worker_config=worker_config,
            process=process,
        )

    def poll(self) -> int | None:
        return self._process.poll()

    def wait(self) -> None:
        self._process.wait()

    def terminate(self) -> None:
        self._process.terminate()

    def stdout(self) -> io.TextIOWrapper:
        if TYPE_CHECKING:
            assert isinstance(self._process.stdout, io.TextIOWrapper)
        return self._process.stdout

    def stderr(self) -> io.TextIOWrapper:
        if TYPE_CHECKING:
            assert isinstance(self._process.stderr, io.TextIOWrapper)
        return self._process.stderr
