import attrs

from tractorun.backend.ray.environment import Environment
from tractorun.base_backend import BackendBase


__all__ = ["Ray"]


@attrs.define
class Ray(BackendBase):
    @property
    def environment(self) -> Environment:
        return Environment()
