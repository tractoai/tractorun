import attr

from tractorun.backend.tractorch.environment import Environment
from tractorun.base_backend import BackendBase


@attr.define
class Tractorch(BackendBase):
    @property
    def environment(self) -> Environment:
        return Environment()