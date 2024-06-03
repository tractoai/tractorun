import json
import os
import subprocess
import sys


with open("config.json", "r") as f:
    config = json.load(f)

training_script = config["training_script"]
nnodes = config["nnodes"]
nproc = config["nproc"]
ngpu_per_proc = config["ngpu_per_proc"]

processes = []

for i in range(nproc):
    proc_config = {
        "nnodes": nnodes,
        "nproc": nproc,
        "ngpu_per_proc": ngpu_per_proc,
        "node_index": os.environ["YT_JOB_COOKIE"],
        "proc_index": i,
    }
    with open(f"config_{i}.json", "w") as f:
        json.dump(proc_config, f)

    command = ["python3", training_script]
    process = subprocess.Popen(
        command,
        stdout=sys.stdout,
        stderr=sys.stderr,
        bufsize=1,
        universal_newlines=True,
        env={**os.environ, "TRACTO_CONFIG": f"config_{i}.json"},
    )
    processes.append(process)

for process in processes:
    exit_code = process.wait()
    if exit_code != 0:
        sys.exit(exit_code)
