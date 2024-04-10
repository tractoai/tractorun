import run
import job_client

import torch
import torch.distributed as dist
import sys

def train(job_client: job_client.JobClient) -> None:
    if job_client.coordinator.get_self_index() == 0:
        x = torch.as_tensor([42.])
        dist.send(tensor=x, dst=1)
    else:
        x = torch.zeros(1)
        dist.recv(tensor=x, src=0)
        print('x = ', x, file=sys.stderr)


if __name__ == "__main__":
    run.run(train, "//home/gritukan/train", 2)
