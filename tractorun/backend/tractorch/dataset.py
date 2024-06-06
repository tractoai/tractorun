from typing import (
    Iterator,
    Optional,
    Sized,
)

import torch.utils.data
from yt import wrapper as yt

from tractorun.backend.tractorch.serializer import TensorSerializer
from tractorun.toolbox import Toolbox


class YtDataset(torch.utils.data.IterableDataset, Sized):
    def __init__(
        self,
        toolbox: Toolbox,
        path: str,
        start: int = 0,
        end: Optional[int] = None,
        columns: Optional[list] = None,
    ) -> None:
        self._yt_cli = toolbox.yt_client

        row_count = self._yt_cli.get(path + "/@row_count")
        if end is None:
            end = row_count
        else:
            assert end <= row_count

        all_columns = []
        for column in self._yt_cli.get(path + "/@schema"):
            all_columns.append(column["name"])
        if columns is None:
            columns = all_columns
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
            return tuple([self._serializer.load_tensor(yt.yson.get_bytes(row[name])) for name in self._columns])

        return map(transform, self._yt_cli.read_table(self._read_path))

    def __len__(self) -> int:
        return self._len
