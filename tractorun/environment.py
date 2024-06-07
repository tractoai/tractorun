import os

from tractorun.checkpoints import CheckpointManager
from tractorun.closet import Closet
from tractorun.toolbox import Toolbox


def get_toolbox(closet: Closet) -> Toolbox:
    checkpoint_manager = CheckpointManager(closet.yt_path + "/checkpoints", closet.yt_client)
    # TODO: make CheckpointFactory instead of the mystical initialize method
    checkpoint_manager.initialize()
    # TODO: coordinator should be with prerequisites
    toolbox = Toolbox(
        coordinator=closet.coordinator,
        checkpoint_manager=checkpoint_manager,
        yt_client=closet.yt_client,
        mesh=closet.mesh,
    )

    return toolbox


def prepare_environment(closet: Closet) -> None:
    # Runs in a job

    ep = closet.coordinator.get_primary_endpoint()
    os.environ["MASTER_ADDR"] = ep.split(":")[0]
    os.environ["MASTER_PORT"] = ep.split(":")[1]
    os.environ["WORLD_SIZE"] = str(closet.coordinator.get_total_peer_count())
    os.environ["NODE_RANK"] = str(closet.coordinator.get_self_index() // closet.mesh.process_per_node)
    os.environ["LOCAL_RANK"] = str(closet.coordinator.get_self_index() % closet.mesh.process_per_node)
