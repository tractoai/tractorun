from abc import (
    ABC,
    abstractmethod,
)
from datetime import timedelta
from typing import Any

import attrs
import testcontainers_yt_local.container
from yt.wrapper.client import YtClient
from yt.wrapper.default_config import retries_config


class YtInstance(ABC):
    @abstractmethod
    def get_client(self) -> YtClient:
        pass


@attrs.define
class YtInstanceExternal(YtInstance):
    proxy_url: str
    token: str = attrs.field(repr=False)

    def get_client(self) -> YtClient:
        return YtClient(
            proxy=self.proxy_url,
            token=self.token,
        )


class YtInstanceTestContainers(YtInstance):
    def __init__(self) -> None:
        self.yt_container = testcontainers_yt_local.container.YtLocalContainer(
            use_ng_image=True,
            enable_cri_jobs=True,
            privileged=True,
            set_yt_config_env_var=True,
        )

    def __enter__(self) -> "YtInstanceTestContainers":
        self.yt_container.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.yt_container.stop()

    def get_client(self) -> YtClient:
        return self.yt_container.get_client(
            config={
                "proxy": {
                    "retries": retries_config(
                        enable=True,
                        total_timeout=timedelta(minutes=5),
                        backoff={
                            "policy": "constant_time",
                            "constant_time": 5,
                        },
                    ),
                },
            },
        )
