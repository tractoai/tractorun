import datetime
import sys
import time

import attr
import yt.wrapper as yt

from tractorun.helpers import create_prerequisite_client
from tractorun.mesh import Mesh


@attr.define(kw_only=True)
class Coordinator:
    _mesh: Mesh
    _self_index: int
    _node_index: int
    _process_index: int
    _self_endpoint: str
    _incarnation_id: int
    _primary_endpoint: str

    @classmethod
    def create(
        cls,
        self_endpoint: str,
        node_index: int,
        mesh: Mesh,
        process_index: int,
        yt_client: yt.YtClient,
        yt_path: str,
        operation_id: str,
        job_id: str,
    ) -> "Coordinator":
        cls._create_directory(yt_client=yt_client, yt_path=yt_path)

        self_index = node_index * mesh.process_per_node + process_index
        if self_index == 0:
            return cls._make_primary(
                self_endpoint=self_endpoint,
                node_index=node_index,
                mesh=mesh,
                process_index=process_index,
                yt_client=yt_client,
                yt_path=yt_path,
                operation_id=operation_id,
                job_id=job_id,
                self_index=self_index,
            )
        else:
            return cls._make_subordinate(
                self_endpoint=self_endpoint,
                node_index=node_index,
                mesh=mesh,
                process_index=process_index,
                yt_client=yt_client,
                yt_path=yt_path,
                self_index=self_index,
                operation_id=operation_id,
                job_id=job_id,
            )

    def get_self_index(self) -> int:
        return self._self_index

    @classmethod
    def _get_total_peer_count(cls, mesh: Mesh) -> int:
        return mesh.node_count * mesh.process_per_node

    def get_total_peer_count(self) -> int:
        return self._get_total_peer_count(self._mesh)

    def get_incarnation_id(self) -> int:
        return self._incarnation_id

    def is_primary(self) -> bool:
        return self._self_index == 0

    def get_primary_endpoint(self) -> str:
        if self._primary_endpoint is None:
            raise RuntimeError("Torchesaurus coordinator is not prepared yet")
        return self._primary_endpoint

    def get_process_index(self) -> int:
        return self._process_index

    @classmethod
    def _create_directory(
        cls,
        yt_client: yt.YtClient,
        yt_path: str,
    ) -> None:
        yt_client.create("map_node", yt_path, attributes={"incarnation_id": -1}, ignore_existing=True)
        yt_client.create("map_node", yt_path + "/primary_lock", ignore_existing=True)
        yt_client.create("map_node", yt_path + "/incarnations", ignore_existing=True)

    @classmethod
    def _make_primary(
        cls,
        self_index: int,
        self_endpoint: str,
        node_index: int,
        mesh: Mesh,
        process_index: int,
        operation_id: str,
        job_id: str,
        yt_client: yt.YtClient,
        yt_path: str,
    ) -> "Coordinator":
        incarnation_transaction_id = yt_client.start_transaction()
        assert incarnation_transaction_id is not None

        with yt_client.Transaction(
            transaction_id=incarnation_transaction_id,
            acquire=False,
            ping=False,
        ):
            yt_client.lock(
                yt_path + "/primary_lock",
                mode="exclusive",
                waitable=True,
                wait_for=int(datetime.timedelta(minutes=5).total_seconds() * 1000),
            )

        if yt_client.exists(yt_path + "/@incarnation_id"):
            last_incarnation_id = yt_client.get(yt_path + "/@incarnation_id")
        else:
            last_incarnation_id = -1
        incarnation_id = last_incarnation_id + 1

        incarnation_yt_client = create_prerequisite_client(
            yt_client,
            [incarnation_transaction_id],
        )

        with incarnation_yt_client.Transaction():
            incarnation_yt_client.set(
                yt_path + "/@incarnation_id",
                incarnation_id,
            )

        with incarnation_yt_client.Transaction():
            incarnation_path = cls._get_incarnation_path(
                yt_path=yt_path,
                incarnation_id=incarnation_id,
            )
            incarnation_yt_client.create("map_node", incarnation_path)
            incarnation_yt_client.set(
                incarnation_path + "/@incarnation_transaction_id",
                incarnation_transaction_id,
            )
            incarnation_yt_client.set(
                incarnation_path + "/@incarnation_operation_id",
                operation_id,
            )
            incarnation_yt_client.set(
                incarnation_path + "/@primary_endpoint",
                self_endpoint,
            )

            topology = [
                {
                    "endpoint": self_endpoint,
                    "job_id": job_id,
                },
            ] + [
                {"address": "", "job_id": ""}
            ] * (cls._get_total_peer_count(mesh) - 1)
            incarnation_yt_client.set(
                incarnation_path + "/@topology",
                topology,
            )

        return Coordinator(
            self_index=self_index,
            incarnation_id=incarnation_id,
            mesh=mesh,
            node_index=node_index,
            process_index=process_index,
            self_endpoint=self_endpoint,
            primary_endpoint=self_endpoint,
        )

    @classmethod
    def _make_subordinate(
        cls,
        self_index: int,
        self_endpoint: str,
        node_index: int,
        mesh: Mesh,
        process_index: int,
        operation_id: str,
        job_id: str,
        yt_client: yt.YtClient,
        yt_path: str,
    ) -> "Coordinator":
        while True:
            try:
                incarnation_id = yt_client.get(yt_path + "/@incarnation_id")
                incarnation_path = cls._get_incarnation_path(yt_path=yt_path, incarnation_id=incarnation_id)
                if (
                    yt_client.get(
                        incarnation_path + "/@incarnation_operation_id",
                    )
                    != operation_id
                ):
                    raise RuntimeError("Operation id mismatch")

                incarnation_transaction_id: str = yt_client.get(
                    incarnation_path + "/@incarnation_transaction_id",
                )
                assert incarnation_transaction_id is not None
                incarnation_yt_client = create_prerequisite_client(
                    yt_client,
                    [incarnation_transaction_id],
                )

                incarnation_yt_client.set(
                    incarnation_path + f"/@topology/{self_index}",
                    {"address": self_endpoint, "job_id": job_id},
                )

                primary_endpoint = incarnation_yt_client.get(
                    incarnation_path + "/@primary_endpoint",
                )
            except Exception:
                time.sleep(1.0)
                continue
            return Coordinator(
                self_index=self_index,
                incarnation_id=incarnation_id,
                mesh=mesh,
                node_index=node_index,
                process_index=process_index,
                self_endpoint=self_endpoint,
                primary_endpoint=primary_endpoint,
            )

    @classmethod
    def _get_incarnation_path(cls, yt_path: str, incarnation_id: int) -> str:
        return yt_path + f"/incarnations/{incarnation_id:05d}"
