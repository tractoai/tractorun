import os

import attrs


__all__ = ["BindLocal", "BindCypress"]


def _to_abs_path(path: str) -> str:
    # mypy workaround
    return os.path.abspath(path)


@attrs.define(kw_only=True, slots=True)
class BindLocal:
    # TODO: just use pathlib
    source: str = attrs.field(converter=_to_abs_path)
    destination: str = attrs.field(converter=_to_abs_path)


@attrs.define(kw_only=True, slots=True)
class BindCypress:
    source: str = attrs.field()
    destination: str = attrs.field()
