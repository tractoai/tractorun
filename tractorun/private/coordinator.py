import datetime
import logging
import time

import attrs
from yt import wrapper as yt
from yt.wrapper.errors import YtResolveError

from tractorun.coordinator import Coordinator
from tractorun.mesh import Mesh
from tractorun.private.helpers import create_prerequisite_client
from tractorun.private.training_dir import TrainingDir
from tractorun.private.training_dir import TrainingDir as _TrainingDir


_LOGGER = logging.getLogger(__name__)


def get_incarnation_id(yt_client: yt.YtClient, training_dir: TrainingDir, raise_if_not_exists: bool = False) -> int:
    try:
        return yt_client.get(training_dir.base_path + "/@incarnation_id")
    except YtResolveError:
        if raise_if_not_exists:
            raise
        return -1


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class CoordinatorFactory:
    _self_endpoint: str
    _self_index: int
    _process_index: int
    _node_index: int
    _mesh: Mesh
    _yt_client: yt.YtClient
    _training_dir: _TrainingDir
    _operation_id: str
    _job_id: str

    def create(self) -> Coordinator:
        if self._self_index == 0:
            return self._make_primary(self_index=self._self_index)
        else:
            return self._make_subordinate(self_index=self._self_index)

    def _wait_for_gang_barrier(self, incarnation_path: str) -> None:
        _LOGGER.debug("Waiting for all peers to start")
        while True:
            try:
                topology = self._yt_client.get(incarnation_path + "/@topology")
                if all(peer["address"] != "" for peer in topology):
                    _LOGGER.debug("All peers started")
                    break
            except Exception as e:
                _LOGGER.exception("_wait_for_gang_barrier raised exception", exc_info=e)
            time.sleep(1.0)

    def _make_primary(self, self_index: int) -> "Coordinator":
        incarnation_transaction_id = self._yt_client.start_transaction()
        assert incarnation_transaction_id is not None

        with self._yt_client.Transaction(
            transaction_id=incarnation_transaction_id,
            acquire=False,
            ping=False,
        ):
            self._yt_client.lock(
                self._training_dir.primary_lock_path,
                mode="exclusive",
                waitable=True,
                wait_for=int(datetime.timedelta(minutes=5).total_seconds() * 1000),
            )

        last_incarnation_id = get_incarnation_id(self._yt_client, self._training_dir)
        incarnation_id = last_incarnation_id + 1

        incarnation_yt_client = create_prerequisite_client(
            self._yt_client,
            [incarnation_transaction_id],
        )

        with incarnation_yt_client.Transaction():
            incarnation_yt_client.set(
                self._training_dir.base_path + "/@incarnation_id",
                incarnation_id,
            )

        with incarnation_yt_client.Transaction():
            incarnation_path = self._training_dir.get_incarnation_path(incarnation_id)
            incarnation_yt_client.create("map_node", incarnation_path)
            incarnation_yt_client.set(
                incarnation_path + "/@incarnation_transaction_id",
                incarnation_transaction_id,
            )
            incarnation_yt_client.set(
                incarnation_path + "/@incarnation_operation_id",
                self._operation_id,
            )
            incarnation_yt_client.set(
                incarnation_path + "/@primary_endpoint",
                self._self_endpoint,
            )

            topology = [
                {
                    "address": self._self_endpoint,
                    "job_id": self._job_id,
                },
            ] + [
                {"address": "", "job_id": ""}
            ] * (self._mesh.peer_count - 1)
            incarnation_yt_client.set(
                incarnation_path + "/@topology",
                topology,
            )

        self._wait_for_gang_barrier(incarnation_path)
        _LOGGER.debug("Primary coordinator started")

        return Coordinator(
            self_index=self_index,
            incarnation_id=incarnation_id,
            mesh=self._mesh,
            process_index=self._process_index,
            self_endpoint=self._self_endpoint,
            primary_endpoint=self._self_endpoint,
        )

    def _make_subordinate(self, self_index: int) -> Coordinator:
        while True:
            try:
                incarnation_id = get_incarnation_id(
                    self._yt_client,
                    self._training_dir,
                    raise_if_not_exists=True,
                )
                incarnation_path = self._training_dir.get_incarnation_path(incarnation_id)
                if (
                    self._yt_client.get(
                        incarnation_path + "/@incarnation_operation_id",
                    )
                    != self._operation_id
                ):
                    raise RuntimeError("Operation id mismatch")

                # incarnation_transaction_id: str = self._yt_client.get(
                #     incarnation_path + "/@incarnation_transaction_id",
                # )
                # assert incarnation_transaction_id is not None
                # incarnation_yt_client = create_prerequisite_client(
                #     self._yt_client,
                #     [incarnation_transaction_id],
                # )
                incarnation_yt_client = self._yt_client

                incarnation_yt_client.set(
                    incarnation_path + f"/@topology/{self_index}",
                    {"address": self._self_endpoint, "job_id": self._job_id},
                )

                primary_endpoint = incarnation_yt_client.get(
                    incarnation_path + "/@primary_endpoint",
                )

                self._wait_for_gang_barrier(incarnation_path)
            except Exception:
                time.sleep(1.0)
                continue
            _LOGGER.debug("Subordinate coordinator started")
            return Coordinator(
                self_index=self_index,
                incarnation_id=incarnation_id,
                mesh=self._mesh,
                process_index=self._process_index,
                self_endpoint=self._self_endpoint,
                primary_endpoint=primary_endpoint,
            )
