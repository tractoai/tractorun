import os

import attrs


__all__ = ["BindLocal", "BindCypress", "BindAttributes"]


def _to_abs_path(path: str) -> str:
    # mypy workaround
    return os.path.abspath(path)


@attrs.define(kw_only=True, slots=True)
class BindAttributes:
    executable: bool = attrs.field(default=True)
    format: str | None = attrs.field(default=None)
    bypass_artifact_cache: bool = attrs.field(default=False)


@attrs.define(kw_only=True, slots=True)
class BindLocal:
    # TODO: just use pathlib
    source: str = attrs.field(converter=_to_abs_path)
    destination: str = attrs.field(converter=_to_abs_path)


@attrs.define(kw_only=True, slots=True)
class BindCypress:
    source: str = attrs.field()
    destination: str = attrs.field()
    attributes: BindAttributes = attrs.field(default=BindAttributes())
