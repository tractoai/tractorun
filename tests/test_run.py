import json
import subprocess
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
from tractorun.dataset import YtDataset
from tractorun.job_client import JobClient
from tractorun.mesh import Mesh
from tractorun.run import run
from tractorun.utils import save_tensor


def test_prepare_dataset(yt_instance: YtInstance, mnist_ds_path: str) -> None:
    yt_cli = yt_instance.get_client()
    row_count = yt_cli.get_attribute(mnist_ds_path, "row_count")
    assert row_count == 100
    schema = yt_cli.get_attribute(mnist_ds_path, "schema")
    assert len(schema) == 2


def test_run_torch_simple(yt_instance: YtInstance, mnist_ds_path: str) -> None:
    model_path = f"//tmp/{get_random_string(13)}"
    yt_cli = yt_instance.get_client()

    def train(job_client: JobClient) -> None:
        class Net(nn.Module):
            def __init__(self) -> None:
                super(Net, self).__init__()
                self.l1 = torch.nn.Linear(28 * 28, 10)

            def forward(self, x: Any) -> torch.Tensor:
                return torch.relu(self.l1(x.view(x.size(0), -1)))

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
        job_client.yt_client.write_file(model_path, save_tensor(model.state_dict()))

    mesh = Mesh(node_count=1, process_per_node=1, gpu_per_process=0)
    run(train, "//tmp", mesh, yt_cli=yt_cli, docker_image=DOCKER_IMAGE)

    # The operation did not fail => success!

    # TODO: figure out why
    # Jobs may fail with `RuntimeError: mat1 and mat2 must have the same dtype, but got Long and Float`
    # but eventually finish successfully. Problems with `mnist_ds_path`?


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
