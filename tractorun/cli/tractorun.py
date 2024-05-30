import argparse

from tractorun.run import run_script


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tractorun")
    parser.add_argument("--nnodes", type=int)
    parser.add_argument("--nproc_per_node", type=int)
    parser.add_argument("--ngpu_per_proc", type=int)

    parser.add_argument("--path", type=str)

    parser.add_argument("training_script")

    args = parser.parse_args()

    nnodes = args.nnodes
    nproc_per_node = args.nproc_per_node
    ngpu_per_proc = args.ngpu_per_proc
    path = args.path

    script_name = args.training_script.split("/")[-1]

    run_script(args, script_name)
