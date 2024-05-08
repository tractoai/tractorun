from __future__ import print_function

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import StepLR

from torchesaurus.job_client import JobClient
from torchesaurus.dataset import YtDataset
from torchesaurus.mesh import Mesh
from torchesaurus.run import run
from torchesaurus.utils import load_tensor, save_tensor

import yt.wrapper as yt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)


    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))


def train(job_client: JobClient) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Running on device:', device, file=sys.stderr)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)

    checkpoint = job_client.checkpoint_manager.get_last_checkpoint()
    first_batch_index = 0
    if checkpoint is not None:
        first_batch_index = checkpoint.metadata['first_batch_index']
        print('Found checkpoint with index', checkpoint.index, 'and first batch index', first_batch_index, file=sys.stderr)

        checkpoint_dict = load_tensor(checkpoint.value)
        model.load_state_dict(checkpoint_dict['model'])
        optimizer.load_state_dict(checkpoint_dict['optimizer'])

    train_dataset = YtDataset(job_client, "//home/gritukan/mnist/datasets/train", device=device, start=0, end=4000)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # TODO: Do it normally. YT dataloader?
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
            print('Train[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()), file=sys.stderr)
        if batch_idx % 3 == 0:
            state_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            metadata_dict = {
                "first_batch_index": batch_idx + 1,
                "loss": loss.item(),
            }
            job_client.checkpoint_manager.save_checkpoint(save_tensor(state_dict), metadata_dict)
            print('Saved checkpoint after batch with index', batch_idx, file=sys.stderr)

    # Save the model
    yt.create("map_node", "//home/gritukan/mnist/models", recursive=True, ignore_existing=True)
    epoch_id = job_client.coordinator.get_epoch_id()
    model_path = f"//home/gritukan/mnist/models/model_{epoch_id}.pt"
    yt.write_file(model_path, save_tensor(model.state_dict()))
    print("Model saved to", model_path, file=sys.stderr)


if __name__ == '__main__':
    # Remove old checkpoints.
    if yt.exists("//home/gritukan/mnist/trainings/dense/checkpoints"):
        yt.remove("//home/gritukan/mnist/trainings/dense/checkpoints", recursive=True)

    mesh = Mesh(node_count=1, process_per_node=1, gpu_per_process=0)
    run(train, "//home/gritukan/mnist/trainings/dense", mesh)
