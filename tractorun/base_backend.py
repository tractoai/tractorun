import abc

from tractorun.private.closet import Closet as _Closet


__all__ = ["EnvironmentBase", "BackendBase"]
print('BOOO')

class EnvironmentBase(abc.ABC):
    @abc.abstractmethod
    def prepare(self, closet: _Closet) -> None:
        pass


class BackendBase(abc.ABC):
    @property
    @abc.abstractmethod
    def environment(self) -> EnvironmentBase:
        pass
