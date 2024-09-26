import argparse
import os
import sys
from typing import Any
import uuid

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import wandb
import yt.wrapper as yt

from tractorun.backend.tractorch import Tractorch
from tractorun.backend.tractorch.dataset import YtTensorDataset
from tractorun.backend.tractorch.serializer import TensorSerializer
from tractorun.env import EnvVariable
from tractorun.mesh import Mesh
from tractorun.resources import Resources
from tractorun.run import run
from tractorun.toolbox import Toolbox


DEFAULT_DATASET_PATH = "//home/yt-team/chiffa/tractorun/mnist/datasets/train"


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x: Any) -> torch.Tensor:
        return torch.relu(self.l1(x.view(x.size(0), -1)))


def train(toolbox: Toolbox) -> None:
    user_config = toolbox.get_user_config()
    dataset_path = user_config["dataset_path"]
    wandb_enabled = user_config["wandb_enabled"]
    if wandb_enabled:
        wandb.login(key=os.environ["WANDB_TOKEN"])
        wandb.init(
            project="tractorun",
            name="torch_mnist_checkpoints",
            id=user_config["wandb_run_id"],
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    serializer = TensorSerializer()
    print("Running on device:", device, file=sys.stderr)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)

    checkpoint = toolbox.checkpoint_manager.get_last_checkpoint()
    first_batch_index = 0
    if checkpoint is not None:
        first_batch_index = checkpoint.metadata["first_batch_index"]
        print(
            "Found checkpoint with index",
            checkpoint.index,
            "and first batch index",
            first_batch_index,
            file=sys.stderr,
        )

        checkpoint_dict = serializer.desirialize(checkpoint.value)
        model.load_state_dict(checkpoint_dict["model"])
        optimizer.load_state_dict(checkpoint_dict["optimizer"])

    train_dataset = YtTensorDataset(
        toolbox,
        path=dataset_path,
        start=0,
        end=4000,
    )
    dataset_len = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)

    model.train()
    if wandb_enabled:
        wandb.watch(model, log_freq=100)

    final_loss = None

    for batch_idx, (data, target) in enumerate(train_loader):
        # TODO: Do it normally. YT dataloader? YT loader!
        if batch_idx < first_batch_index:
            continue

        if batch_idx == 10 and first_batch_index == 0:
            assert False, "Force restart to test checkpoints"

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 5 == 0:
            print(
                "Train[{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    batch_idx * len(data),
                    dataset_len,
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                ),
                file=sys.stderr,
            )
            if wandb_enabled:
                wandb.log({"loss": loss.item(), "batch_idx": batch_idx})

        if batch_idx % 3 == 0:
            state_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            metadata_dict = {
                "first_batch_index": batch_idx + 1,
                "loss": loss.item(),
            }
            toolbox.checkpoint_manager.save_checkpoint(serializer.serialize(state_dict), metadata_dict)
            # save checkpoint synchronously
            # task = toolbox.checkpoint_manager.save_checkpoint(serializer.serialize(state_dict), metadata_dict)
            # task.wait(timeout=10)
            print("Saved checkpoint after batch with index", batch_idx, file=sys.stderr)
        final_loss = loss.item()

    # Save the model
    model_path = toolbox.save_model(
        data=serializer.serialize(model.state_dict()),
        dataset_path=dataset_path,
        metadata={
            "loss": str(final_loss),
        },
    )
    print("Model saved to", model_path, file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yt-home-dir", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)  # type: ignore  # FIXME: min python -> 3.10
    args = parser.parse_args()

    # Remove old checkpoints.
    workdir = args.yt_home_dir
    if yt.exists(f"{workdir}/checkpoints"):
        yt.remove(f"{workdir}/checkpoints", recursive=True)

    mesh = Mesh(node_count=1, process_per_node=1, gpu_per_process=0)
    run(
        train,
        backend=Tractorch(),
        yt_path=workdir,
        mesh=mesh,
        user_config={
            "workdir": workdir,
            "dataset_path": args.dataset_path,
            "wandb_run_id": str(uuid.uuid4()),
            "wandb_enabled": args.wandb,
        },
        env=[
            EnvVariable(
                name="WANDB_API_KEY",
                cypress_path=os.environ.get("WANDB_SECRET"),
            ),
        ],
        resources=Resources(memory_limit=4 * (1024**3)),
    )
