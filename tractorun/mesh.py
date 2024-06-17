from typing import (
    List,
    Optional,
)
import json

import attrs
import cattrs


@attrs.define
class Mesh:
    node_count: int = attrs.field()
    process_per_node: int = attrs.field()
    gpu_per_process: int = attrs.field()
    pool_trees: Optional[List[str]] = attrs.field(default=None)


class MeshSerializer:
    @staticmethod
    def serialize(mesh: Mesh) -> str:
        return json.dumps(cattrs.unstructure(mesh))

    @staticmethod
    def deserialize(data: str) -> Mesh:
        return cattrs.structure(json.loads(data), Mesh)
