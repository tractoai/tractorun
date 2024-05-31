from abc import (
    ABC,
    abstractmethod,
)
from typing import Any

import attr
import testcontainers_yt_local.container
from yt.wrapper.client import YtClient


class YtInstance(ABC):
    @abstractmethod
    def get_client(self) -> YtClient:
        pass


@attr.define
class YtInstanceExternal(YtInstance):
    proxy_url: str
    token: str = attr.ib(repr=False)

    def get_client(self) -> YtClient:
        return YtClient(
            proxy=self.proxy_url,
            token=self.token,
            config={
                "pickling": {"ignore_system_modules": True},  # otherwise fat torch will be sent
            },
        )


class YtInstanceTestContainers(YtInstance):
    def __init__(self) -> None:
        self.yt_container = testcontainers_yt_local.container.YtLocalContainer()

    def __enter__(self) -> "YtInstanceTestContainers":
        self.yt_container.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.yt_container.stop()

    def get_client(self) -> YtClient:
        return self.yt_container.get_client()
