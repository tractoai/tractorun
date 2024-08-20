import enum as _enum

import attrs as _attrs


class RestartPolicy(str, _enum.Enum):
    ON_FAILURE = "on_failure"
    ALWAYS = "always"
    NEVER = "never"
    FAIL = "fail"


@_attrs.define(kw_only=True, slots=True, auto_attribs=True)
class Sidecar:
    command: list[str]
    restart_policy: RestartPolicy
