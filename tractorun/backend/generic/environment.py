from tractorun.base_backend import EnvironmentBase
from tractorun.private.closet import Closet as _Closet
from tractorun.private.environment import prepare_environment as _common_prepare_environment


__all__ = ["Environment"]


class Environment(EnvironmentBase):
    def prepare(self, closet: _Closet) -> None:
        _common_prepare_environment(closet)
