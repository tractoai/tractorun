import sys
from typing import (
    Any,
    Tuple,
)

from pytorch_lightning import (
    LightningModule,
    Trainer,
)
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from tractorun.backend.tractorch.dataset import YtTensorDataset
from tractorun.mesh import Mesh
from tractorun.resources import Resources
from tractorun.run import run
from tractorun.toolbox import Toolbox


DATASET_PATH = "//home/yt-team/chiffa/tractorun/mnist/datasets/train"


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
        return loss

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), lr=0.02)


def train(toolbox: Toolbox) -> None:
    print("Mesh", toolbox.mesh, file=sys.stderr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device:", device, file=sys.stderr)

    mnist_model = MNISTModel()
    train_dataset = YtTensorDataset(toolbox, DATASET_PATH)
    train_loader = DataLoader(train_dataset, batch_size=64)

    trainer = Trainer(
        max_epochs=3,
        devices=toolbox.mesh.process_per_node,
        num_nodes=toolbox.mesh.node_count,
        strategy="ddp",
    )

    trainer.fit(mnist_model, train_loader)


def main() -> None:
    workdir = "//home/yt-team/chiffa/tractorun/mnist"
    mesh = Mesh(node_count=1, process_per_node=3, gpu_per_process=1)
    run(
        train,
        yt_path=f"{workdir}/trainings/dense_two_layers",
        mesh=mesh,
        resources=Resources(
            memory_limit=8076021002,
        ),
    )


if __name__ == "__main__":
    main()
