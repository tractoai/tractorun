import os
from typing import Optional

import pytest
from yt.wrapper import yt_dataclass

from .utils import (
    get_data_path,
    get_random_string,
)
from .yt_instances import (
    YtInstanceExternal,
    YtInstanceTestContainers,
)


@pytest.fixture(scope="session")
def yt_instance():
    yt_mode = os.environ.get("YT_MODE", "testcontainers")
    if yt_mode == "testcontainers":
        with YtInstanceTestContainers() as yt_instance:
            yield yt_instance
    elif yt_mode == "external":
        proxy_url = os.environ["YT_PROXY"]
        yt_token = os.environ.get("YT_TOKEN")
        yield YtInstanceExternal(proxy_url=proxy_url, token=yt_token)
    else:
        raise ValueError(f"Unknown yt_mode: {yt_mode}")


@pytest.fixture(scope="session")
def mnist_ds_path(yt_instance):
    @yt_dataclass
    class Row:
        data: Optional[bytes]
        labels: Optional[bytes]

    table_path = f"//tmp/{get_random_string(13)}"

    yt_cli = yt_instance.get_client()

    # TODO: generalize and move to utils
    parsed_data = []
    with open(get_data_path("mnist_small"), "rb") as mnist_file:
        for line in mnist_file:
            pairs = [p.split(b"=") for p in line.split(b"\t")]
            parsed_data.append(Row(data=pairs[0][1], labels=pairs[1][1]))

    yt_cli.write_table_structured(
        table=table_path,
        input_stream=parsed_data,
        row_type=Row,
    )

    yield table_path

    yt_cli.remove(table_path)
