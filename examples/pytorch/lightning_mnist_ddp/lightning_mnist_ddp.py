import argparse
import os
import sys
from typing import (
    Any,
    Tuple,
)
import uuid

from pytorch_lightning import (
    LightningModule,
    Trainer,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
import wandb

from tractorun.backend.tractorch import Tractorch
from tractorun.backend.tractorch.dataset import YtTensorDataset
from tractorun.env import EnvVariable
from tractorun.mesh import Mesh
from tractorun.resources import Resources
from tractorun.run import run
from tractorun.toolbox import Toolbox


DEFAULT_DATASET_PATH = "//home/yt-team/chiffa/tractorun/mnist/datasets/train"


class MNISTModel(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 50)
        self.l2 = torch.nn.Linear(50, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.l1(x.view(x.size(0), -1)))
        x = torch.relu(self.l2(x))
        return x

    def training_step(self, batch: Tuple[Tensor, ...], batch_nb: Any) -> STEP_OUTPUT:
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("train_loss", loss)
        self.log("batch_nb", batch_nb)
        return loss

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), lr=0.02)


def train(toolbox: Toolbox) -> None:
    user_config = toolbox.get_user_config()
    dataset_path = user_config["dataset_path"]
    wandb_enabled = user_config["wandb_enabled"]
    if wandb_enabled:
        wandb.login(key=os.environ["WANDB_TOKEN"])
        wandb_logger = WandbLogger(
            project="tractorun",
            name="lightning_mnist_ddp",
            id=user_config["wandb_run_id"],
            log_model="all",
        )
    else:
        wandb_logger = None

    print("Mesh", toolbox.mesh, file=sys.stderr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device:", device, file=sys.stderr)

    mnist_model = MNISTModel()
    train_dataset = YtTensorDataset(toolbox, dataset_path)
    train_loader = DataLoader(train_dataset, batch_size=64)

    trainer = Trainer(
        max_epochs=3,
        devices=toolbox.mesh.process_per_node,
        num_nodes=toolbox.mesh.node_count,
        strategy="ddp",
        logger=wandb_logger,
    )

    trainer.fit(mnist_model, train_loader)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yt-home-dir", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)  # type: ignore  # FIXME: min python -> 3.10
    args = parser.parse_args()

    workdir = args.yt_home_dir
    mesh = Mesh(node_count=1, process_per_node=3, gpu_per_process=1)
    run(
        train,
        backend=Tractorch(),
        yt_path=workdir,
        mesh=mesh,
        resources=Resources(
            memory_limit=8076021002,
        ),
        env=[
            EnvVariable(
                name="WANDB_API_KEY",
                cypress_path=os.environ.get("WANDB_SECRET"),
            ),
        ],
        user_config={
            "dataset_path": args.dataset_path,
            "wandb_run_id": str(uuid.uuid4()),
            "wandb_enabled": args.wandb,
        },
    )


if __name__ == "__main__":
    main()
