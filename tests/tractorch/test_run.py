import json
import sys
from typing import (
    Any,
    Callable,
)

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from tests.utils import (
    TRACTORCH_DOCKER_IMAGE,
    TractoCli,
    get_data_path,
)
from tests.yt_instances import YtInstance
from tractorun.backend.tractorch import Tractorch
from tractorun.backend.tractorch.dataset import YtTensorDataset
from tractorun.backend.tractorch.serializer import TensorSerializer
from tractorun.mesh import Mesh
from tractorun.run import run
from tractorun.toolbox import Toolbox


def _get_simple_train(mnist_ds_path: str) -> Callable:
    def _simple_train(toolbox: Toolbox) -> None:
        class Net(nn.Module):
            def __init__(self) -> None:
                super(Net, self).__init__()
                self.l1 = torch.nn.Linear(28 * 28, 10)

            def forward(self, x: Any) -> torch.Tensor:
                return torch.relu(self.l1(x.view(x.size(0), -1)))

        device = torch.device("cpu")
        serializer = TensorSerializer()
        train_dataset = YtTensorDataset(toolbox.yt_client, mnist_ds_path)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)
        model = Net().to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=1.0)
        model.train()
        final_loss = None
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()
        toolbox.save_model(
            data=serializer.serialize(model.state_dict()),
            dataset_path=mnist_ds_path,
            metadata={
                "loss": str(final_loss),
            },
        )

    return _simple_train


def test_run_pickle(yt_instance: YtInstance, yt_path: str, mnist_ds_path: str) -> None:
    yt_client = yt_instance.get_client()

    train_func = _get_simple_train(mnist_ds_path)

    mesh = Mesh(node_count=1, process_per_node=1, gpu_per_process=0)
    # The operation did not fail => success!
    run(
        train_func,
        backend=Tractorch(),
        yt_path=yt_path,
        mesh=mesh,
        yt_client=yt_client,
        docker_image=TRACTORCH_DOCKER_IMAGE,
    )


def test_run_torch_distributed(yt_instance: YtInstance, yt_path: str, mnist_ds_path: str) -> None:
    yt_client = yt_instance.get_client()

    train_func = _get_simple_train(mnist_ds_path)

    mesh = Mesh(node_count=2, process_per_node=1, gpu_per_process=0)
    # The operation did not fail => success!
    run(
        train_func,
        backend=Tractorch(),
        yt_path=yt_path,
        mesh=mesh,
        yt_client=yt_client,
        docker_image=TRACTORCH_DOCKER_IMAGE,
    )


def test_run_torch_with_checkpoints(yt_instance: YtInstance, yt_path: str, mnist_ds_path: str) -> None:
    yt_client = yt_instance.get_client()

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

        train_dataset = YtTensorDataset(toolbox.yt_client, mnist_ds_path)
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

    def _run() -> None:
        run(
            train,
            backend=Tractorch(),
            yt_path=yt_path,
            mesh=mesh,
            yt_client=yt_client,
            docker_image=TRACTORCH_DOCKER_IMAGE,
        )

    # First launch is failed because of the exception in the train function.
    with pytest.raises(Exception):
        _run()
    # And second one should be successful.
    _run()


def test_run_script(yt_instance: YtInstance, yt_path: str, mnist_ds_path: str) -> None:
    yt_client = yt_instance.get_client()

    tracto_cli = TractoCli(
        command=["python3", "/tractorun_tests/torch_run_script.py"],
        docker_image=TRACTORCH_DOCKER_IMAGE,
        args=[
            "--mesh.node-count",
            "1",
            "--mesh.process-per-node",
            "1",
            "--mesh.gpu-per-process",
            "0",
            "--yt-path",
            yt_path,
            "--user-config",
            json.dumps({"MNIST_DS_PATH": mnist_ds_path}),
            "--bind-local",
            f"{get_data_path('../data/torch_run_script.py')}:/tractorun_tests/torch_run_script.py",
        ],
    )
    op_run = tracto_cli.run()
    assert op_run.is_exitcode_valid()
    assert op_run.is_operation_state_valid(yt_client=yt_client, job_count=1)
