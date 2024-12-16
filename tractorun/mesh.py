import attrs


__all__ = ["Mesh"]


@attrs.define
class Mesh:
    node_count: int = attrs.field(default=1)
    process_per_node: int = attrs.field(default=1)
    gpu_per_process: int = attrs.field(default=0)
    pool: str | None = attrs.field(default=None)
    pool_trees: list[str] | None = attrs.field(default=None)

    @property
    def peer_count(self) -> int:
        return self.node_count * self.process_per_node
