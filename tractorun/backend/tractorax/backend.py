import attrs

from tractorun.backend.tractorax.environment import Environment
from tractorun.private.base_backend import BackendBase


@attrs.define
class Tractorax(BackendBase):
    @property
    def environment(self) -> Environment:
        return Environment()
