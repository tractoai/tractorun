import enum

import attrs


__all__ = ["Sidecar", "RestartPolicy"]


class RestartPolicy(str, enum.Enum):
    ON_FAILURE = "on_failure"
    ALWAYS = "always"
    NEVER = "never"
    FAIL = "fail"


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class Sidecar:
    command: list[str]
    restart_policy: RestartPolicy
