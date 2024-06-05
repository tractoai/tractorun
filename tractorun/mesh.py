from typing import Optional

import attrs


@attrs.define
class Mesh:
    node_count: int = attrs.field()
    process_per_node: int = attrs.field()
    gpu_per_process: int = attrs.field()
    gpu_type: Optional[str] = attrs.field(default=None)
