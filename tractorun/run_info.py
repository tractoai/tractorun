__all__ = ["RunInfo"]

from typing import Any

import attrs


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class RunInfo:
    operation_spec: dict
    operation_id: str | None
    operation_attributes: dict[Any, Any] | None
