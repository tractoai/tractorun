import attr

from tractorun.backend.tractorax.environment import Environment
from tractorun.base_backend import BackendBase


@attr.define
class Tractorax(BackendBase):
    @property
    def environment(self) -> Environment:
        return Environment()
