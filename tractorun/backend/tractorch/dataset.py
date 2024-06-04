from typing import (
    Iterator,
    Optional,
    Sized,
)

import torch.utils.data
from yt import wrapper as yt

from tractorun.backend.tractorch.serializer import TensorSerializer
from tractorun.job_client import JobClient


class YtDataset(torch.utils.data.IterableDataset, Sized):
    def __init__(
        self,
        client: JobClient,
        path: str,
        device: torch.device,
        start: int = 0,
        end: Optional[int] = None,
        columns: Optional[list] = None,
    ) -> None:
        self._yt_cli = client.yt_client
        self._device = device

        row_count = self._yt_cli.get(path + "/@row_count")
        if end is None:
            end = row_count
        else:
            assert end <= row_count

        all_columns = set()
        for column in self._yt_cli.get(path + "/@schema"):
            all_columns.add(column["name"])
        if columns is None:
            columns = list(all_columns)
        else:
            for column in columns:
                assert column in all_columns

        # TODO: pass list of columns
        read_path = f"{path}[#{start}:#{end}]"
        self._len = end - start
        self._read_path = read_path
        self._columns = columns
        self._serializer = TensorSerializer()

    def __iter__(self) -> Iterator:
        def transform(row: dict) -> tuple:
            return tuple(
                [
                    self._serializer.load_tensor(yt.yson.get_bytes(row[name]), device=self._device)
                    for name in self._columns
                ]
            )

        return map(transform, self._yt_cli.read_table(self._read_path))

    def __len__(self) -> int:
        return self._len
