import os
from typing import Optional

from tractorun.checkpoints import CheckpointManager
from tractorun.closet import Closet
from tractorun.exceptions import TractorunWandbError
from tractorun.toolbox import Toolbox


def get_toolbox(closet: Closet) -> Toolbox:
    checkpoint_manager = CheckpointManager(closet.training_dir.checkpoints_path, closet.yt_client)
    # TODO: make CheckpointFactory instead of the mystical initialize method
    checkpoint_manager.initialize()
    # TODO: coordinator should be with prerequisites
    toolbox = Toolbox(
        coordinator=closet.coordinator,
        checkpoint_manager=checkpoint_manager,
        yt_client=closet.yt_client,
        mesh=closet.mesh,
        training_dir=closet.training_dir,
        training_metadata=closet.training_metadata,
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

    if os.environ["WANDB_ENABLED"] == "1":
        _prepare_wandb(key=os.environ.get("YT_SECURE_VAULT_WANDB_API_KEY"))


def _prepare_wandb(key: Optional[str]) -> None:
    if key is None:
        raise TractorunWandbError(
            "WandB token is not set. Token can be set by env `YT_SECURE_VAULT_WANDB_API_KEY` or by `wandb_api_key` option",
        )
    try:
        import wandb
    except ImportError as e:
        raise TractorunWandbError("WandB is not found in python env") from e
    wandb.login(key=key)
