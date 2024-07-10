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

from tractorun.backend.tractorch import Tractorch
from tractorun.backend.tractorch.dataset import YtTensorDataset
from tractorun.mesh import Mesh
from tractorun.resources import Resources
from tractorun.run import run
from tractorun.toolbox import Toolbox


class MNISTModel(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x: Any) -> torch.Tensor:
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch: Tuple[Tensor, ...], batch_nb: Any) -> STEP_OUTPUT:
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), lr=0.02)


def train(toolbox: Toolbox) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device:", device, file=sys.stderr)

    mnist_model = MNISTModel()
    train_dataset = YtTensorDataset(
        toolbox,
        "//home/gritukan/mnist/datasets/train",
        start=0,
        end=1000,
    )
    train_loader = DataLoader(train_dataset, batch_size=64)

    trainer = Trainer(max_epochs=3)

    trainer.fit(mnist_model, train_loader)


if __name__ == "__main__":
    mesh = Mesh(node_count=1, process_per_node=1, gpu_per_process=0)
    run(
        train,
        backend=Tractorch(),
        yt_path="//home/gritukan/mnist/trainings/dense",
        mesh=mesh,
        resources=Resources(
            memory_limit=1076021002,
        ),
    )
