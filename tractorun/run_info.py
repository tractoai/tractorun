__all__ = ["RunInfo", "YtRunInfo", "LocalRunInfo"]


import attrs


@attrs.define
class RunInfo:
    pass


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class YtRunInfo(RunInfo):
    operation_spec: dict


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class LocalRunInfo(RunInfo):
    pass
