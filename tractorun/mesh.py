import attrs


@attrs.define
class Mesh:
    node_count: int
    process_per_node: int
    gpu_per_process: int
