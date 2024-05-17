from abc import ABC, abstractmethod

import attr

import testcontainers_yt_local.container
from yt.wrapper.client import YtClient


class YtInstance(ABC):
    @abstractmethod
    def get_client(self) -> YtClient:
        pass


@attr.define
class YtInstanceExternal(YtInstance):
    proxy_url: str = attr.ib()
    token: str = attr.ib(repr=False)

    def get_client(self) -> YtClient:
        return YtClient(
            proxy=self.proxy_url,
            token=self.token,
            config={"proxy": {"enable_proxy_discovery": False}}
        )


class YtInstanceTestContainers(YtInstance):
    def __init__(self):
        self.yt_container = testcontainers_yt_local.container.YtLocalContainer()

    def __enter__(self):
        self.yt_container.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.yt_container.stop()

    def get_client(self) -> YtClient:
        return self.yt_container.get_client()
