import enum

import attrs


__all__ = ["Sidecar", "RestartPolicy"]

from tractorun.private.helpers import AttrSerializer as _AttrSerializer


class RestartPolicy(str, enum.Enum):
    ON_FAILURE = "on_failure"
    ALWAYS = "always"
    NEVER = "never"
    FAIL = "fail"


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class Sidecar:
    command: list[str]
    restart_policy: RestartPolicy

    @staticmethod
    def from_args(value: str) -> "Sidecar":
        return _AttrSerializer(Sidecar).deserialize(value)
