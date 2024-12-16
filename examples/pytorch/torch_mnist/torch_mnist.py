import argparse
import os
from pathlib import Path
import random
import string
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
from tractorun.resources import Resources
from tractorun.run import run
from tractorun.toolbox import Toolbox


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x: Any) -> torch.Tensor:
        return torch.relu(self.l1(x.view(x.size(0), -1)))


def train(toolbox: Toolbox) -> None:
    dataset_path = toolbox.get_user_config()["dataset_path"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    serializer = TensorSerializer()
    print("Running on device:", device, file=sys.stderr)

    train_dataset = YtTensorDataset(
        toolbox.yt_client,
        dataset_path,
        columns=["data", "labels"],
        start=0,
        end=10,
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


def _default_home_dir() -> str:
    rnm = "".join(random.choices(string.ascii_letters + string.digits, k=8))
    return f"//tmp/tractorun_examples/{rnm}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yt-home-dir", type=str, default=_default_home_dir())
    parser.add_argument("--pool-tree", type=str, default="default")
    parser.add_argument("--dataset-path", type=str, default="//home/samples/mnist-torch-train")
    parser.add_argument("--docker-image", type=str, default=os.environ.get("DOCKER_IMAGE"))
    parser.add_argument("--gpu-per-process", type=int, default=0)
    args = parser.parse_args()

    mesh = Mesh(node_count=1, process_per_node=8, gpu_per_process=args.gpu_per_process, pool_trees=[args.pool_tree])

    tractorun_path = (Path(__file__).parent.parent.parent.parent / "tractorun").resolve()
    run(
        train,
        backend=Tractorch(),
        yt_path=args.yt_home_dir,
        resources=Resources(
            memory_limit=8076021002,
        ),
        mesh=mesh,
        docker_image=args.docker_image,
        binds_local_lib=[str(tractorun_path)],
        user_config={"dataset_path": args.dataset_path},
    )
