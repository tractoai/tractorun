import abc as _abc

from tractorun.private.closet import Closet as _Closet


class EnvironmentBase(_abc.ABC):
    @_abc.abstractmethod
    def prepare(self, closet: _Closet) -> None:
        pass


class BackendBase(_abc.ABC):
    @property
    @_abc.abstractmethod
    def environment(self) -> EnvironmentBase:
        pass
