import argparse

from tractorun.run import run_script
from tractorun.mesh import Mesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tractorun")
    parser.add_argument("--nnodes", type=int)
    parser.add_argument("--nproc_per_node", type=int)
    parser.add_argument("--ngpu_per_proc", type=int)
    parser.add_argument("training_script")

    args = parser.parse_args()

    mesh = Mesh(node_count=args.nnodes, process_per_node=args.nproc_per_node, gpu_per_process=args.ngpu_per_proc)

    script_name = args.training_script.split("/")[-1]

    run_script(args, script_name)
