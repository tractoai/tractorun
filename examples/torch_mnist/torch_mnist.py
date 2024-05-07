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

from torchesaurus.utils import save_tensor

import yt.wrapper as yt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)
        """ self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10) """

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))
        """x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output """


def train(job_client: JobClient) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Running on device:', device, file=sys.stderr)

    train_dataset = YtDataset(job_client, "//home/gritukan/mnist/datasets/train", device=device, start=0, end=2000)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()), file=sys.stderr)

    # Save the model
    yt.create("map_node", "//home/gritukan/mnist/models", recursive=True, ignore_existing=True)
    epoch_id = job_client.coordinator.get_epoch_id()
    model_path = f"//home/gritukan/mnist/models/model_{epoch_id}.pt"
    yt.write_file(model_path, save_tensor(model.state_dict()))
    print("Model saved to", model_path, file=sys.stderr)


if __name__ == '__main__':
    mesh = Mesh(node_count=1, process_per_node=1, gpu_per_process=0)
    run(train, "//home/gritukan/mnist/train", mesh)
