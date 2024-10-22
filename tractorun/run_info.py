__all__ = ["RunInfo"]

import attrs


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class RunInfo:
    operation_spec: dict
    operation_id: str | None
