from tractorun.base_backend import EnvironmentBase
from tractorun.closet import Closet
from tractorun.environment import prepare_environment as common_prepare_environment


class Environment(EnvironmentBase):
    def prepare(self, closet: Closet) -> None:
        common_prepare_environment(closet)
