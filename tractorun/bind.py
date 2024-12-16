import pathlib

import attrs


def _to_abs_local(path: str) -> str:
    return str(pathlib.Path(path).absolute())


def _to_abs_posix(path: str) -> str:
    return str(pathlib.PosixPath(path).absolute())


__all__ = ["BindLocal", "BindCypress", "BindAttributes"]


@attrs.define(kw_only=True, slots=True)
class BindAttributes:
    executable: bool = attrs.field(default=True)
    format: str | None = attrs.field(default=None)
    bypass_artifact_cache: bool = attrs.field(default=False)


@attrs.define(kw_only=True, slots=True)
class BindLocal:
    # TODO: just use pathlib
    source: str = attrs.field(converter=_to_abs_local)
    destination: str = attrs.field(converter=_to_abs_posix)


@attrs.define(kw_only=True, slots=True)
class BindCypress:
    source: str = attrs.field()
    destination: str = attrs.field()
    attributes: BindAttributes = attrs.field(default=BindAttributes())
