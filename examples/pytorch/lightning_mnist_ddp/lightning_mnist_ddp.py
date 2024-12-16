import argparse
import os
from pathlib import Path
import random
import string
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


DEFAULT_DATASET_PATH = "//home/samples/mnist-torch-train"


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
    if os.environ.get("WANDB_TOKEN"):
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
    train_dataset = YtTensorDataset(
        yt_client=toolbox.yt_client,
        path=dataset_path,
        columns=["data", "labels"],
        start=0,
        end=10,
    )
    train_loader = DataLoader(train_dataset, batch_size=64)

    trainer = Trainer(
        max_epochs=1,
        devices=toolbox.mesh.process_per_node,
        num_nodes=toolbox.mesh.node_count,
        strategy="ddp",
        logger=wandb_logger,
    )

    trainer.fit(mnist_model, train_loader)


def _default_home_dir() -> str:
    rnm = "".join(random.choices(string.ascii_letters + string.digits, k=8))
    return f"//tmp/tractorun_examples/{rnm}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yt-home-dir", type=str, default=_default_home_dir())
    parser.add_argument("--pool-tree", type=str, default="default")
    parser.add_argument("--dataset-path", type=str, default="//home/samples/mnist-torch-train")
    parser.add_argument("--docker-image", type=str, default=os.environ.get("DOCKER_IMAGE"))
    parser.add_argument("--gpu-per-process", type=int, default=0)
    args = parser.parse_args()

    mesh = Mesh(node_count=1, process_per_node=8, gpu_per_process=args.gpu_per_process, pool_trees=[args.pool_tree])

    tractorun_path = (Path(__file__).parent.parent.parent.parent / "tractorun").resolve()
    env = []
    if os.environ.get("WANDB_SECRET"):
        env = [
            EnvVariable(
                name="WANDB_API_KEY",
                cypress_path=os.environ.get("WANDB_SECRET"),
            )
        ]
    run(
        train,
        backend=Tractorch(),
        yt_path=args.yt_home_dir,
        mesh=mesh,
        resources=Resources(
            memory_limit=8076021002,
        ),
        docker_image=args.docker_image,
        env=env,
        binds_local_lib=[str(tractorun_path)],
        user_config={
            "dataset_path": args.dataset_path,
            "wandb_run_id": str(uuid.uuid4()),
        },
    )


if __name__ == "__main__":
    main()
