import attrs

from tractorun.backend.generic.environment import Environment
from tractorun.base_backend import BackendBase


@attrs.define
class GenericBackend(BackendBase):
    @property
    def environment(self) -> Environment:
        return Environment()
