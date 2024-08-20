import asyncio
import copy
from functools import partial
import json
from typing import (
    Generic,
    TypeVar,
)

import attrs
import yt.wrapper as yt


_T = TypeVar("_T")


__all__ = ["CheckpointManager", "Checkpoint"]


@attrs.define
class Checkpoint:
    index: int
    value: bytes
    metadata: dict


def _save_checkpoint(yt_client: yt.YtClient, path: str, metadata: bytes, value: bytes) -> None:
    with yt_client.Transaction():
        yt_client.create("map_node", path)
        yt_client.write_file(path + "/value", value)
        yt_client.write_file(
            path + "/metadata",
            metadata,
        )


@attrs.define
class _Task(Generic[_T]):
    _task: asyncio.Future[_T]

    def wait(self, timeout: int) -> _T:
        return asyncio.get_event_loop().run_until_complete(asyncio.wait_for(self._task, timeout=timeout))


@attrs.define
class CheckpointManager:
    _path: str
    _yt_client: yt.YtClient
    _last_checkpoint_index: int = -1

    def initialize(self) -> None:
        last_checkpoint_index = -1
        for index in self._yt_client.list(self._path):
            try:
                index = int(index)
            except Exception:
                continue
            last_checkpoint_index = max(last_checkpoint_index, index)
        self._last_checkpoint_index = last_checkpoint_index

    def get_last_checkpoint(self) -> Checkpoint | None:
        if self._last_checkpoint_index == -1:
            return None

        checkpoint_path = self._path + "/" + str(self._last_checkpoint_index)
        value = self._yt_client.read_file(checkpoint_path + "/value").read()
        metadata = json.loads(self._yt_client.read_file(checkpoint_path + "/metadata").read())

        return Checkpoint(self._last_checkpoint_index, value, metadata)

    def save_checkpoint(self, value: bytes, metadata: dict | None = None) -> _Task:
        if metadata is None:
            metadata = {}
        # TODO: prerequisites

        checkpoint_index = self._last_checkpoint_index + 1
        checkpoint_path = self._path + "/" + str(checkpoint_index)
        serialized_metadata = json.dumps(metadata).encode("utf-8")
        self._last_checkpoint_index = checkpoint_index

        yt_client = yt.YtClient(config=copy.deepcopy(self._yt_client.config))
        save_checkpoint_task = partial(
            _save_checkpoint,
            yt_client=yt_client,
            path=checkpoint_path,
            metadata=serialized_metadata,
            value=value,
        )
        task = asyncio.get_event_loop().run_in_executor(None, save_checkpoint_task)
        return _Task[None](
            task=task,
        )
