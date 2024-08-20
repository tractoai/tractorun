import os as _os

import attrs as _attrs


def _to_abs_path(path: str) -> str:
    # mypy workaround
    return _os.path.abspath(path)


@_attrs.define(kw_only=True, slots=True)
class BindLocal:
    # TODO: just use pathlib
    source: str = _attrs.field(converter=_to_abs_path)
    destination: str = _attrs.field(converter=_to_abs_path)
