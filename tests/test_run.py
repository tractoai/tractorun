import json
import subprocess
import sys
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from tests.utils import (
    DOCKER_IMAGE,
    DOCKER_IMAGE_TRTRCH,
    get_data_path,
    get_random_string,
)
from tests.yt_instances import YtInstance
from tractorun.backend.tractorch.dataset import YtTensorDataset
from tractorun.backend.tractorch.serializer import TensorSerializer
from tractorun.mesh import Mesh
from tractorun.run import run
from tractorun.toolbox import Toolbox


def test_prepare_dataset(yt_instance: YtInstance, mnist_ds_path: str) -> None:
    yt_cli = yt_instance.get_client()
    row_count = yt_cli.get_attribute(mnist_ds_path, "row_count")
    assert row_count == 100
    schema = yt_cli.get_attribute(mnist_ds_path, "schema")
    assert len(schema) == 2


def test_run_torch_simple(yt_instance: YtInstance, mnist_ds_path: str) -> None:
    model_path = f"//tmp/{get_random_string(13)}"
    yt_training_dir = f"//tmp/{get_random_string(13)}"
    yt_cli = yt_instance.get_client()

    def train(toolbox: Toolbox) -> None:
        class Net(nn.Module):
            def __init__(self) -> None:
                super(Net, self).__init__()
                self.l1 = torch.nn.Linear(28 * 28, 10)

            def forward(self, x: Any) -> torch.Tensor:
                return torch.relu(self.l1(x.view(x.size(0), -1)))

        device = torch.device("cpu")
        serializer = TensorSerializer()
        train_dataset = YtTensorDataset(toolbox, mnist_ds_path)
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
        toolbox.yt_client.write_file(model_path, serializer.serialize(model.state_dict()))

    mesh = Mesh(node_count=1, process_per_node=1, gpu_per_process=0)
    run(train, yt_path=yt_training_dir, mesh=mesh, yt_cli=yt_cli, docker_image=DOCKER_IMAGE)

    # The operation did not fail => success!


def test_run_torch_with_checkpoints(yt_instance: YtInstance, mnist_ds_path: str) -> None:
    yt_cli = yt_instance.get_client()
    yt_training_dir = f"//tmp/{get_random_string(13)}"

    def train(toolbox: Toolbox) -> None:
        class Net(nn.Module):
            def __init__(self) -> None:
                super(Net, self).__init__()
                self.l1 = torch.nn.Linear(28 * 28, 10)

            def forward(self, x: Any) -> torch.Tensor:
                return torch.relu(self.l1(x.view(x.size(0), -1)))

        device = torch.device("cpu")
        serializer = TensorSerializer()
        model = Net().to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=1.0)

        checkpoint = toolbox.checkpoint_manager.get_last_checkpoint()
        first_batch_index = 0
        if checkpoint is not None:
            first_batch_index = checkpoint.metadata["first_batch_index"]
            print(
                "Found checkpoint with index",
                checkpoint.index,
                "and first batch index",
                first_batch_index,
                file=sys.stderr,
            )

            checkpoint_dict = serializer.desirialize(checkpoint.value)
            model.load_state_dict(checkpoint_dict["model"])
            optimizer.load_state_dict(checkpoint_dict["optimizer"])

        train_dataset = YtTensorDataset(toolbox, mnist_ds_path)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10)

        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # TODO: Do it normally. YT dataloader?
            if batch_idx < first_batch_index:
                continue

            if batch_idx == 3 and first_batch_index == 0:
                raise Exception("Force restart to test checkpoints")

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 3 == 0:
                state_dict = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                metadata_dict = {
                    "first_batch_index": batch_idx + 1,
                    "loss": loss.item(),
                }
                toolbox.checkpoint_manager.save_checkpoint(serializer.serialize(state_dict), metadata_dict)
                print("Saved checkpoint after batch with index", batch_idx, file=sys.stderr)

    mesh = Mesh(node_count=1, process_per_node=1, gpu_per_process=0)
    run(train, yt_path=yt_training_dir, mesh=mesh, yt_cli=yt_cli, docker_image=DOCKER_IMAGE)


def test_run_script(yt_instance: YtInstance, mnist_ds_path: str) -> None:
    process = subprocess.Popen(
        [
            get_data_path("../../tractorun/cli/tractorun_runner.py"),
            "--nnodes",
            "1",
            "--nproc_per_node",
            "1",
            "--ngpu_per_proc",
            "0",
            "--yt-path",
            "//tmp",
            "--docker-image",
            DOCKER_IMAGE_TRTRCH,  # TODO: run on usual DOCKER_IMAGE
            "--user-config",
            json.dumps({"MNIST_DS_PATH": mnist_ds_path}),
            get_data_path("../data/torch_run_script.py"),
        ]
    )
    process.wait()
    assert process.returncode == 0
