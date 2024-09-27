__all__ = ["RunInfo", "YtRunInfo", "LocalRunInfo"]

from typing import Any

import attrs


@attrs.define
class RunInfo:
    pass


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class YtRunInfo(RunInfo):
    operation_spec: dict
    operation_id: str | None
    operation_attributes: dict[Any, Any] | None


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class LocalRunInfo(RunInfo):
    pass
