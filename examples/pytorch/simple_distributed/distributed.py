import torch
import torch.distributed as dist

from tractorun.backend.tractorch import Tractorch
from tractorun.mesh import Mesh
from tractorun.run import run
from tractorun.toolbox import Toolbox


def train(toolbox: Toolbox) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    print("Running on device:", device)

    if toolbox.coordinator.get_self_index() == 0:
        x = torch.as_tensor([2.0], device=device)
        y = torch.as_tensor([3.0], device=device)
        print(f"Sending summands to peer 1 (X = {x}, Y = {y})")
        dist.send(x, dst=1)
        dist.send(y, dst=1)
        print("Waiting for result from peer 1")
        sum = torch.zeros(1, device=device)
        dist.recv(tensor=sum, src=1)
        print(f"Received sum from peer 1: {sum}")
    else:
        x = torch.zeros(1, device=device)
        print("Waiting for X from peer 0")
        dist.recv(tensor=x, src=0)
        print(f"Received X from peer 0: {x}")
        print("Waiting for Y from peer 0")
        y = torch.zeros(1, device=device)
        dist.recv(tensor=y, src=0)
        print(f"Received Y from peer 1: {y}")
        sum = x + y
        print(f"Sending sum to peer 1: {sum}")
        dist.send(tensor=sum, dst=0)


if __name__ == "__main__":
    mesh = Mesh(1, 2, 0)
    run(
        train,
        backend=Tractorch(),
        yt_path="//home/gritukan/train",
        mesh=mesh,
    )
