from tractorun.base_backend import EnvironmentBase
from tractorun.private.closet import Closet as _Closet


__all__ = ["Environment"]


class Environment(EnvironmentBase):
    def prepare(self, closet: _Closet) -> None:
        return None
