import os

import attrs
import yt.wrapper as yt

from tractorun.private.closet import Closet
from tractorun.private.yt_cluster import make_cypress_link
from tractorun.toolbox import Toolbox


def get_toolbox(closet: Closet) -> Toolbox:
    toolbox = Toolbox(
        coordinator=closet.coordinator,
        checkpoint_manager=closet.checkpoint_manager,
        yt_client=closet.yt_client,
        mesh=closet.mesh,
        training_dir=closet.training_dir,
        training_metadata=closet.training_metadata,
    )

    return toolbox


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class Link:
    _value: str | None

    def to_yson(self) -> yt.yson.yson_types.YsonUnicode | None:
        if self._value is None:
            return None
        value = yt.yson.yson_types.YsonUnicode(f"{self._value}")
        value.attributes = {"_type_tag": "url"}
        return value


def prepare_environment(closet: Closet) -> None:
    # Runs in a job
    ep = closet.coordinator.get_primary_endpoint()
    os.environ["MASTER_ADDR"] = ep.split(":")[0]
    os.environ["MASTER_PORT"] = ep.split(":")[1]
    os.environ["WORLD_SIZE"] = str(closet.coordinator.get_total_peer_count())
    os.environ["NODE_RANK"] = str(closet.coordinator.get_self_index() // closet.mesh.process_per_node)
    os.environ["LOCAL_RANK"] = str(closet.coordinator.get_self_index() % closet.mesh.process_per_node)

    if closet.coordinator.is_primary():
        training_dir = make_cypress_link(
            path=closet.training_dir.base_path,
            cypress_link_template=closet.cluster_config.cypress_link_template,
        )
        description = (
            {
                "tractorun": {
                    "training_dir": Link(value=training_dir).to_yson(),
                    "primary_address": ep,
                    "incarnation": closet.coordinator.get_incarnation_id(),
                },
            },
        )
        closet.yt_client.update_operation_parameters(
            closet.training_metadata.operation_id,
            parameters={
                "annotations": {"description": description},
            },
        )
