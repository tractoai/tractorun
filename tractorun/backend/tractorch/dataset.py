from typing import (
    TYPE_CHECKING,
    Callable,
    Iterator,
    Sized,
    TypeVar,
)
import warnings

import attrs
import torch.utils.data
from yt import wrapper as yt

from tractorun.backend.tractorch.serializer import TensorSerializer
from tractorun.toolbox import Toolbox


T_co = TypeVar("T_co")


__all__ = ["YTTensorTransform", "YtTensorDataset", "YtDataset"]


@attrs.define(frozen=True, slots=True)
class YTTensorTransform:
    _serializer: TensorSerializer = attrs.field(default=TensorSerializer())

    def __call__(self, columns: list[str], row: dict) -> tuple:
        return tuple((self._serializer.desirialize(yt.yson.get_bytes(row[name])) for name in columns))


class YtDataset(torch.utils.data.IterableDataset[T_co], Sized):
    def __init__(
        self,
        path: str,
        yt_client: yt.YtClient,
        transform: Callable[[list[str], dict], T_co],
        start: int = 0,
        end: int | None = None,
        columns: list | None = None,
    ) -> None:
        self._yt_client = yt_client

        row_count = self._yt_client.get(path + "/@row_count")
        if end is None:
            end = row_count
        else:
            assert end <= row_count

        all_columns = []
        for column in self._yt_client.get(path + "/@schema"):
            if TYPE_CHECKING:
                assert isinstance(column["name"], str)
            all_columns.append(column["name"])
        if columns is None:
            columns = all_columns
        else:
            for column in columns:
                assert column in all_columns

        read_path = f"{path}[#{start}:#{end}]"
        self._len = end - start
        self._read_path = read_path
        self._columns = columns
        self._serializer = TensorSerializer()
        self._transform = transform

    def __iter__(self) -> Iterator[T_co]:
        return (self._transform(self._columns, row) for row in self._yt_client.read_table(self._read_path))

    def __len__(self) -> int:
        return self._len


class YtTensorDataset(YtDataset[tuple]):
    def __init__(
        self,
        yt_client: yt.YtClient,
        path: str,
        start: int = 0,
        toolbox: Toolbox | None = None,
        end: int | None = None,
        columns: list | None = None,
    ) -> None:
        if yt_client is not None:
            self._yt_client = yt_client
        # for migration
        elif toolbox is not None:
            warnings.warn("Set yt_client instead of toolbox")
            self._yt_client = toolbox.yt_client
        else:
            raise ValueError("missing 1 required positional argument: yt_client")

        super().__init__(
            yt_client=yt_client,
            path=path,
            start=start,
            end=end,
            columns=columns,
            transform=YTTensorTransform(),
        )
