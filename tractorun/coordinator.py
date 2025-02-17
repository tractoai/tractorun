import attrs

from tractorun.mesh import Mesh


__all__ = ["Coordinator"]


@attrs.define(kw_only=True)
class Coordinator:
    _mesh: Mesh
    _self_index: int
    _process_index: int
    _self_endpoint: str
    _incarnation_id: int
    _primary_endpoint: str

    def get_self_index(self) -> int:
        return self._self_index

    def get_total_peer_count(self) -> int:
        return self._mesh.peer_count

    def get_incarnation_id(self) -> int:
        return self._incarnation_id

    def is_primary(self) -> bool:
        return self._self_index == 0

    def get_primary_endpoint(self) -> str:
        return self._primary_endpoint

    def get_process_index(self) -> int:
        return self._process_index

    def get_self_endpoint(self) -> str:
        return self._self_endpoint
