import attrs


__all__ = ["Resources"]


@attrs.define
class Resources:
    cpu_limit: float | None = None
    memory_limit: int | None = None
