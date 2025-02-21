import attrs


__all__ = ["Resources"]


@attrs.define
class Resources:
    cpu_limit: float = attrs.field(default=1)
    memory_limit: int | None = None
