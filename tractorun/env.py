import attrs

from tractorun.private.helpers import AttrSerializer as _AttrSerializer


__all__ = ["EnvVariable"]


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class EnvVariable:
    name: str
    value: str | None = None
    cypress_path: str | None = None

    @staticmethod
    def from_args(value: str) -> "EnvVariable":
        return _AttrSerializer(EnvVariable).deserialize(value)
