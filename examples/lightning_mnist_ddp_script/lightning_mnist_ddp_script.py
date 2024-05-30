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

from tractorun.dataset import YtDataset
import tractorun.run


module_locations = ["./tmpfs/modules", "./modules"]
sys.path = sys.path + module_locations


class MNISTModel(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 50)
        self.l2 = torch.nn.Linear(50, 10)

    def forward(self, x: Any) -> torch.Tensor:
        x = torch.relu(self.l1(x.view(x.size(0), -1)))
        x = torch.relu(self.l2(x))
        return x

    def training_step(self, batch: Tuple[Tensor, ...], batch_nb: Any) -> STEP_OUTPUT:
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), lr=0.02)


job_client = tractorun.run.initialize(user_config={})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device:", device, file=sys.stderr)

mnist_model = MNISTModel()
train_dataset = YtDataset(job_client, "//home/gritukan/mnist/datasets/train", device=device)
train_loader = DataLoader(train_dataset, batch_size=64)

trainer = Trainer(
    max_epochs=3,
    devices=job_client.get_mesh().process_per_node,
    num_nodes=job_client.get_mesh().node_count,
    strategy="ddp",
)

trainer.fit(mnist_model, train_loader)
