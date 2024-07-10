import abc

from tractorun.closet import Closet


class EnvironmentBase(abc.ABC):
    @abc.abstractmethod
    def prepare(self, closet: Closet) -> None:
        pass


class BackendBase(abc.ABC):
    @property
    @abc.abstractmethod
    def environment(self) -> EnvironmentBase:
        pass
