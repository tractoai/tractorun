import attrs

from tractorun.description import Link
from tractorun.mesh import Mesh


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class TractorunDescription:
    training_dir: Link
    primary_stderr: Link
    logs: Link
    primary_address: str
    incarnation: int
    mesh: Mesh

    def to_dict(self) -> dict:
        return {
            "training_dir": self.training_dir,
            "primary": {
                "job_stderr": self.primary_stderr,
                "address": self.primary_address,
            },
            "logs": self.logs,
            "incarnation": self.incarnation,
            "mesh": attrs.asdict(self.mesh),  # type: ignore
        }
