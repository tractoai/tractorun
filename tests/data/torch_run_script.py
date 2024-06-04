from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from tractorun.backend.tractorch.dataset import YtDataset
from tractorun.backend.tractorch.environment import prepare_environment
from tractorun.utils import get_user_config


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.l1 = nn.Linear(28 * 28, 10)

    def forward(self, x: Any) -> torch.Tensor:
        return torch.relu(self.l1(x.view(x.size(0), -1)))


if __name__ == "__main__":
    user_config = get_user_config()
    mnist_ds_path = user_config["MNIST_DS_PATH"]
    job_client = prepare_environment(user_config={})
    device = torch.device("cpu")
    train_dataset = YtDataset(job_client, mnist_ds_path, device=device)
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
