from dataclasses import dataclass
import typing as tp

import yt.wrapper as yt


@dataclass
class Checkpoint:
    index: int
    value: bytes
    metadata: dict


class CheckpointManager:
    def __init__(self, path: str, client: yt.YtClient) -> None:
        self._path = path
        self._client = client
        self._last_checkpoint_index = None

    def initialize(self) -> None:
        yt.create("map_node", self._path, ignore_existing=True, client=self._client)

        last_checkpoint_index = -1
        for index in yt.list(self._path, client=self._client):
            try:
                index = int(index)
            except Exception:
                continue
            last_checkpoint_index = max(last_checkpoint_index, index)
        self._last_checkpoint_index = last_checkpoint_index

    def get_last_checkpoint(self) -> tp.Optional[Checkpoint]:
        if self._last_checkpoint_index == -1:
            return None

        checkpoint_path = self._path + "/" + str(self._last_checkpoint_index)
        value = yt.read_file(checkpoint_path + "/value", client=self._client).read()
        metadata = yt.read_file(checkpoint_path + "/metadata", client=self._client).read()
        # TODO: do it normally.
        metadata = eval(metadata.decode("utf-8"))

        return Checkpoint(self._last_checkpoint_index, value, metadata)

    def save_checkpoint(self, value: str, metadata: dict = {}) -> None:
        # TODO: prerequisites
        with yt.Transaction(client=self._client):
            checkpoint_index = self._last_checkpoint_index + 1
            checkpoint_path = self._path + "/" + str(checkpoint_index)
            yt.create("map_node", checkpoint_path, client=self._client)
            yt.write_file(checkpoint_path + "/value", value, client=self._client)
            yt.write_file(
                checkpoint_path + "/metadata",
                str(metadata).encode("utf-8"),
                client=self._client,
            )

        self._last_checkpoint_index = checkpoint_index
