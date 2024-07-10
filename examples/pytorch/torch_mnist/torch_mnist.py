import argparse
import sys
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from tractorun.backend.tractorch import Tractorch
from tractorun.backend.tractorch.dataset import YtTensorDataset
from tractorun.backend.tractorch.serializer import TensorSerializer
from tractorun.mesh import Mesh
from tractorun.run import run
from tractorun.toolbox import Toolbox


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)
        # self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        # self.fc1 = nn.Linear(9216, 128)
        # self.fc2 = nn.Linear(128, 10)

    def forward(self, x: Any) -> torch.Tensor:
        return torch.relu(self.l1(x.view(x.size(0), -1)))
        # x = self.conv1(x)
        # x = F.relu(x)
        # x = self.conv2(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        # x = torch.flatten(x, 1)
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.dropout2(x)
        # x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        # return output


def train(toolbox: Toolbox) -> None:
    dataset_path = "//home/gritukan/mnist/datasets/train"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    serializer = TensorSerializer()
    print("Running on device:", device, file=sys.stderr)

    train_dataset = YtTensorDataset(
        toolbox,
        dataset_path,
        start=0,
        end=2000,
    )
    dataset_len = len(train_dataset)

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
            print(
                "Train[{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    batch_idx * len(data),
                    dataset_len,
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                ),
                file=sys.stderr,
            )

    model_path = toolbox.save_model(
        data=serializer.serialize(model.state_dict()),
        dataset_path=dataset_path,
        metadata={},
    )
    print("Model saved to", model_path, file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yt-home-dir", required=True)
    args = parser.parse_args()

    mesh = Mesh(node_count=1, process_per_node=1, gpu_per_process=0)
    run(
        train,
        backend=Tractorch(),
        yt_path=f"{args.yt_home_dir}/mnist/trainings/dense",
        mesh=mesh,
        user_config={"yt_home_dir": args.yt_home_dir},
    )
