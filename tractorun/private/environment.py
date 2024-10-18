import os

from tractorun.description import Link
from tractorun.private.closet import Closet
from tractorun.private.constants import TRACTORUN_DESCRIPTION_MANAGER_NAME
from tractorun.private.description import TractorunDescription
from tractorun.private.yt_cluster import (
    make_cypress_link,
    make_job_stderr_link,
)
from tractorun.toolbox import Toolbox


def get_toolbox(closet: Closet) -> Toolbox:
    user_description_manager = closet.description_manager.get_child("extra")
    toolbox = Toolbox(
        coordinator=closet.coordinator,
        checkpoint_manager=closet.checkpoint_manager,
        yt_client=closet.yt_client,
        mesh=closet.mesh,
        training_dir=closet.training_dir,
        training_metadata=closet.training_metadata,
        description_manager=user_description_manager,
    )

    return toolbox


def prepare_log_dir(closet: Closet) -> None:
    logs_dir = f"{closet.training_dir.logs_path}/{closet.coordinator.get_incarnation_id()}"
    worker_logs_dir = f"{logs_dir}/workers"
    sidecar_logs_dir = f"{logs_dir}/sidecars"
    closet.yt_client.link(
        closet.training_dir.worker_logs_path,
        worker_logs_dir,
        recursive=True,
    )
    closet.yt_client.link(
        closet.training_dir.sidecar_logs_path,
        sidecar_logs_dir,
        recursive=True,
    )


def make_description(closet: Closet) -> TractorunDescription:
    logs_dir = f"{closet.training_dir.logs_path}/{closet.coordinator.get_incarnation_id()}"
    training_dir = make_cypress_link(
        path=closet.training_dir.base_path,
        cypress_link_template=closet.cluster_config.cypress_link_template,
    )
    job_stderr_link = make_job_stderr_link(
        operation_id=closet.training_metadata.operation_id,
        job_id=closet.training_metadata.job_id,
        job_stderr_link_template=closet.cluster_config.job_stderr_link_template,
    )
    logs_path = make_cypress_link(
        path=f"{logs_dir}",
        cypress_link_template=closet.cluster_config.cypress_link_template,
    )
    return TractorunDescription(
        training_dir=Link(value=training_dir),
        primary_address=closet.coordinator.get_primary_endpoint(),
        incarnation=closet.coordinator.get_incarnation_id(),
        mesh=closet.mesh,
        primary_stderr=Link(value=job_stderr_link),
        logs=Link(value=logs_path),
    )


def prepare_environment(closet: Closet) -> None:
    # Runs in a job
    ep = closet.coordinator.get_primary_endpoint()
    os.environ["MASTER_ADDR"] = ep.split(":")[0]
    os.environ["MASTER_PORT"] = ep.split(":")[1]
    os.environ["WORLD_SIZE"] = str(closet.coordinator.get_total_peer_count())
    os.environ["NODE_RANK"] = str(closet.coordinator.get_self_index() // closet.mesh.process_per_node)
    os.environ["LOCAL_RANK"] = str(closet.coordinator.get_self_index() % closet.mesh.process_per_node)

    if closet.coordinator.is_primary():
        prepare_log_dir(closet=closet)
        description_manager = closet.description_manager.get_child(TRACTORUN_DESCRIPTION_MANAGER_NAME)
        description = make_description(closet=closet)
        description_manager.set(description.to_dict())
