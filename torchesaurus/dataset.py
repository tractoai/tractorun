import torch.utils.data

import yt.wrapper as yt

from .utils import load_tensor
from .job_client import JobClient


class YtDataset(torch.utils.data.IterableDataset):
    def __init__(self, client: JobClient, path: str, device: torch.device, start=0, end=None, columns=None):
        self._client = client.yt_client
        self._device = device

        row_count = yt.get(path + "/@row_count", client=self._client)
        if end is None:
            end = row_count
        else:
            assert end <= row_count

        all_columns = []
        for column in yt.get(path + "/@schema", client=self._client):
            all_columns.append(column["name"])
        if columns is None:
            columns = all_columns
        else:
            all_columns = set(all_columns)
            for column in columns:
                assert column in all_columns
        
        # TODO: pass list of columns
        read_path = f"{path}[#{start}:#{end}]"
        self._len = end - start
        self._read_path = read_path
        self._columns = columns

    def __iter__(self):
        def transform(row):
            return tuple([load_tensor(yt.yson.get_bytes(row[name]), device=self._device) for name in self._columns])
        return map(transform, yt.read_table(self._read_path, client=self._client))
    
    def __len__(self) -> int:
        return self._len
