from .helpers import create_prerequisite_client

import yt.wrapper as yt

import datetime
import os
import socket
import time
import typing as tp
import sys
import torch

from .mesh import Mesh


class Coordinator:
    def __init__(self, client: yt.YtClient, path: str, self_endpoint: str, mesh: Mesh, node_index: int, process_index: int) -> None:
        self._client = client
        self._path = path
        self._mesh = mesh
        self._node_index = node_index
        self._process_index = process_index
        self._self_endpoint = self_endpoint

        self._epoch_id = None

        self._primary_endpoint = None

        self._epoch_transaction = None
        self._epoch_transaction_id = None

        self._epoch_client = None

    def prepare(self, primary_cb: tp.Optional[tp.Callable] = None) -> None:
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
            raise RuntimeError('Torchesaurus coordinator is not prepared yet')
        return self._epoch_id
    
    def get_epoch_client(self) -> yt.YtClient:
        if self._epoch_client is None:
            raise RuntimeError('Torchesaurus coordinator is not prepared yet')
        return self._epoch_client
    
    def get_primary_endpoint(self) -> yt.YtClient:
        if self._primary_endpoint is None:
            raise RuntimeError('Torchesaurus coordinator is not prepared yet')
        return self._primary_endpoint
    
    def get_mesh(self) -> Mesh:
        return self._mesh
    
    def get_process_index(self) -> int:
        return self._process_index

    def _prepare_primary(self, primary_cb: tp.Optional[tp.Callable]) -> None:
        self._epoch_transaction_id = yt.start_transaction(client=self._client)
        self._epoch_transaction = yt.Transaction(
            transaction_id=self._epoch_transaction_id,
            acquire=False,
            client=self._client,
        )

        with self._make_transaction(self._epoch_transaction_id):
            yt.lock(
                self._path + '/primary_lock',
                mode='exclusive',
                waitable=True,
                wait_for=datetime.timedelta(minutes=5).total_seconds() * 1000,
                client=self._client,
            )

        if yt.exists(self._path + '/@epoch_id', client=self._client):
            last_epoch_id = yt.get(self._path + '/@epoch_id', client=self._client)
        else:
            last_epoch_id = -1
        self._epoch_id = last_epoch_id + 1

        self._epoch_client = create_prerequisite_client(self._client, [self._epoch_transaction_id])

        with yt.Transaction(client=self._epoch_client):
            yt.set(self._path + '/@epoch_id', self.get_epoch_id(), client=self._epoch_client)

        if primary_cb:
            primary_cb()

        with yt.Transaction(client=self._epoch_client):
            yt.create('map_node', self._get_epoch_path(), client=self._epoch_client)
            yt.set(self._get_epoch_path() + '/@epoch_transaction_id', self._epoch_transaction_id, client=self._epoch_client)
            yt.set(self._get_epoch_path() + '/@epoch_operation_id', self._get_operation_id(), client=self._epoch_client)
            yt.set(self._get_epoch_path() + '/@primary_endpoint', self._self_endpoint, client=self._epoch_client)

            topology = [{'endpoint': self._self_endpoint, 'job_id': self._get_job_id()}] + [{'address': '', 'job_id': ''}] * (self.get_total_peer_count() - 1)
            yt.set(self._get_epoch_path() + '/@topology', topology, client=self._epoch_client)

        self._primary_endpoint = self._self_endpoint

    def _prepare_subordinate(self, self_index: int) -> None:
        while True:
            try:
                self._epoch_id = yt.get(self._path + '/@epoch_id', client=self._client)
                if yt.get(self._get_epoch_path() + '/@epoch_operation_id', client=self._client) != self._get_operation_id():
                    raise RuntimeError('Operation id mismatch')

                self._epoch_transaction_id = yt.get(self._get_epoch_path() + '/@epoch_transaction_id', client=self._client)
                self._epoch_client = create_prerequisite_client(self._client, [self._epoch_transaction_id])

                yt.set(
                    self._get_epoch_path() + f'/@topology/{self_index}',
                    {'address': self._self_endpoint, 'job_id': self._get_job_id()},
                    client=self._epoch_client,
                )

                self._primary_endpoint = yt.get(self._get_epoch_path() + '/@primary_endpoint', client=self._epoch_client)
            except:
                time.sleep(1.0)
                continue
            break

    def _make_transaction(self, transaction_id: str) -> yt.Transaction:
        return yt.Transaction(
            transaction_id=transaction_id,
            acquire=False,
            ping=False,
            client=self._client,
        )
    
    def _get_epoch_path(self) -> str:
        return self._path + f'/epochs/{self._epoch_id:05d}'

    
    @staticmethod
    def _get_operation_id() -> str:
        return os.environ['YT_OPERATION_ID']
    
    @staticmethod
    def _get_job_id() -> str:
        return os.environ['YT_JOB_ID']
    
    @staticmethod
    def _get_self_address() -> str:
        return socket.gethostname()
