import attrs as attrs


__all__ = ["EnvVariable"]


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class EnvVariable:
    name: str
    value: str | None = None
    cypress_path: str | None = None
