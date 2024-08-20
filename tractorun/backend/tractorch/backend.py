import attrs

from tractorun.backend.tractorch.environment import Environment
from tractorun.base_backend import BackendBase


__all__ = ["Tractorch"]


@attrs.define
class Tractorch(BackendBase):
    @property
    def environment(self) -> Environment:
        return Environment()
