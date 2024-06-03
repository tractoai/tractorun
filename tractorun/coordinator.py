import datetime
import os
import socket
import sys
import time
from typing import (
    Callable,
    Optional,
)

import attr
import yt.wrapper as yt

from tractorun.helpers import create_prerequisite_client
from tractorun.mesh import Mesh


@attr.define(kw_only=True)
class Coordinator:
    _yt_cli: yt.YtClient
    _path: str
    _mesh: Mesh
    _node_index: int
    _process_index: int
    _self_endpoint: str

    _epoch_id: Optional[int] = None

    _primary_endpoint: Optional[str] = None

    _epoch_transaction: Optional[yt.Transaction] = None
    _epoch_transaction_id: Optional[str] = None

    _epoch_client: Optional[yt.YtClient] = None

    def prepare(self, primary_cb: Optional[Callable] = None) -> None:
        self_index = self.get_self_index()
        if self_index == 0:
            self._prepare_primary(primary_cb)
        else:
            self._prepare_subordinate(self_index)
        print("Self address:", self._get_self_address(), file=sys.stderr)

    def get_self_index(self) -> int:
        return self._node_index * self._mesh.process_per_node + self._process_index

    def get_total_peer_count(self) -> int:
        return self._mesh.node_count * self._mesh.process_per_node

    def get_epoch_id(self) -> int:
        if self._epoch_id is None:
            raise RuntimeError("Torchesaurus coordinator is not prepared yet")
        return self._epoch_id

    def get_epoch_client(self) -> yt.YtClient:
        if self._epoch_client is None:
            raise RuntimeError("Torchesaurus coordinator is not prepared yet")
        return self._epoch_client

    def get_primary_endpoint(self) -> str:
        if self._primary_endpoint is None:
            raise RuntimeError("Torchesaurus coordinator is not prepared yet")
        return self._primary_endpoint

    def get_mesh(self) -> Mesh:
        return self._mesh

    def get_process_index(self) -> int:
        return self._process_index

    def _prepare_primary(self, primary_cb: Optional[Callable]) -> None:
        self._epoch_transaction_id = self._yt_cli.start_transaction()
        assert self._epoch_transaction_id is not None
        self._epoch_transaction = self._yt_cli.Transaction(
            transaction_id=self._epoch_transaction_id,
            acquire=False,
        )

        with self._make_transaction(self._epoch_transaction_id):
            self._yt_cli.lock(
                self._path + "/primary_lock",
                mode="exclusive",
                waitable=True,
                wait_for=int(datetime.timedelta(minutes=5).total_seconds() * 1000),
            )

        if self._yt_cli.exists(self._path + "/@epoch_id"):
            last_epoch_id = self._yt_cli.get(self._path + "/@epoch_id")
        else:
            last_epoch_id = -1
        self._epoch_id = last_epoch_id + 1

        self._epoch_client = create_prerequisite_client(self._yt_cli, [self._epoch_transaction_id])

        with self._epoch_client.Transaction():
            self._epoch_client.set(
                self._path + "/@epoch_id",
                self.get_epoch_id(),
            )

        if primary_cb:
            primary_cb()

        with self._epoch_client.Transaction():
            self._epoch_client.create("map_node", self._get_epoch_path())
            self._epoch_client.set(
                self._get_epoch_path() + "/@epoch_transaction_id",
                self._epoch_transaction_id,
            )
            self._epoch_client.set(
                self._get_epoch_path() + "/@epoch_operation_id",
                self._get_operation_id(),
            )
            self._epoch_client.set(
                self._get_epoch_path() + "/@primary_endpoint",
                self._self_endpoint,
            )

            topology = [{"endpoint": self._self_endpoint, "job_id": self._get_job_id()}] + [
                {"address": "", "job_id": ""}
            ] * (self.get_total_peer_count() - 1)
            self._epoch_client.set(
                self._get_epoch_path() + "/@topology",
                topology,
            )

        self._primary_endpoint = self._self_endpoint

    def _prepare_subordinate(self, self_index: int) -> None:
        while True:
            try:
                self._epoch_id = self._yt_cli.get(self._path + "/@epoch_id")
                if (
                    self._yt_cli.get(
                        self._get_epoch_path() + "/@epoch_operation_id",
                    )
                    != self._get_operation_id()
                ):
                    raise RuntimeError("Operation id mismatch")

                self._epoch_transaction_id: str = self._yt_cli.get(
                    self._get_epoch_path() + "/@epoch_transaction_id",
                )
                assert self._epoch_transaction_id is not None
                self._epoch_client = create_prerequisite_client(self._yt_cli, [self._epoch_transaction_id])

                self._epoch_client.set(
                    self._get_epoch_path() + f"/@topology/{self_index}",
                    {"address": self._self_endpoint, "job_id": self._get_job_id()},
                )

                self._primary_endpoint = self._epoch_client.get(
                    self._get_epoch_path() + "/@primary_endpoint",
                )
            except Exception:
                time.sleep(1.0)
                continue
            break

    def _make_transaction(self, transaction_id: str) -> yt.Transaction:
        return self._yt_cli.Transaction(
            transaction_id=transaction_id,
            acquire=False,
            ping=False,
        )

    def _get_epoch_path(self) -> str:
        return self._path + f"/epochs/{self._epoch_id:05d}"

    @staticmethod
    def _get_operation_id() -> str:
        return os.environ["YT_OPERATION_ID"]

    @staticmethod
    def _get_job_id() -> str:
        return os.environ["YT_JOB_ID"]

    @staticmethod
    def _get_self_address() -> str:
        return socket.gethostname()
