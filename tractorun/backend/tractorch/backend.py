import attrs

from tractorun.backend.tractorch.environment import Environment
from tractorun.private.base_backend import BackendBase


@attrs.define
class Tractorch(BackendBase):
    @property
    def environment(self) -> Environment:
        return Environment()
