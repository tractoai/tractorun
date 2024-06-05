import json
from typing import Optional

import attrs
import yt.wrapper as yt


@attrs.define
class Checkpoint:
    index: int
    value: bytes
    metadata: dict


@attrs.define
class CheckpointManager:
    _path: str
    _yt_cli: yt.YtClient
    _last_checkpoint_index: int = -1

    def initialize(self) -> None:
        self._yt_cli.create("map_node", self._path, ignore_existing=True)

        last_checkpoint_index = -1
        for index in self._yt_cli.list(self._path):
            try:
                index = int(index)
            except Exception:
                continue
            last_checkpoint_index = max(last_checkpoint_index, index)
        self._last_checkpoint_index = last_checkpoint_index

    def get_last_checkpoint(self) -> Optional[Checkpoint]:
        if self._last_checkpoint_index == -1:
            return None

        checkpoint_path = self._path + "/" + str(self._last_checkpoint_index)
        value = self._yt_cli.read_file(checkpoint_path + "/value").read()
        metadata = json.loads(self._yt_cli.read_file(checkpoint_path + "/metadata").read())

        return Checkpoint(self._last_checkpoint_index, value, metadata)

    def save_checkpoint(self, value: bytes, metadata: Optional[dict] = None) -> None:
        if metadata is None:
            metadata = {}
        # TODO: prerequisites
        with self._yt_cli.Transaction():
            checkpoint_index = self._last_checkpoint_index + 1
            checkpoint_path = self._path + "/" + str(checkpoint_index)
            self._yt_cli.create("map_node", checkpoint_path)
            self._yt_cli.write_file(checkpoint_path + "/value", value)
            serialized_metadata = json.dumps(metadata).encode("utf-8")
            self._yt_cli.write_file(
                checkpoint_path + "/metadata",
                serialized_metadata,
            )

        self._last_checkpoint_index = checkpoint_index
