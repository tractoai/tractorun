from typing import Optional as _Optional

import attrs as _attrs


@_attrs.define
class Mesh:
    node_count: int = _attrs.field()
    process_per_node: int = _attrs.field()
    gpu_per_process: int = _attrs.field()
    pool_trees: _Optional[list[str]] = _attrs.field(default=None)
