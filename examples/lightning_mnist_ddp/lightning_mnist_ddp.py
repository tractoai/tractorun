import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import sys

import torch

from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F

from torchesaurus.dataset import YtDataset
from torchesaurus.job_client import JobClient
from torchesaurus.mesh import Mesh
from torchesaurus.run import run


class MNISTModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


def train(job_client: JobClient) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Running on device:', device, file=sys.stderr)

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


if __name__ == '__main__':
    mesh = Mesh(node_count=4, process_per_node=8, gpu_per_process=0)
    run(train, "//home/gritukan/mnist/train", mesh)
