import argparse

import yt.wrapper as yt


parser = argparse.ArgumentParser(description='Tractorunner')
parser.add_argument('nnodes', type=int)
parser.add_argument('nproc_per_node', type=int)
parser.add_argument('ngpu_per_proc', type=int)

parser.add_argument('training_script')

args = parser.parse_args()

op = yt.run_operation(
    yt.VanillaSpecBuilder()
        .begin_task("task")
            .command(f"python3 {args.training_script}")
            .job_count(args.nnodes)
            .gpu_limit(args.nproc_per_node * args.ngpu_per_proc)
            .port_count(args.nproc_per_node)
            .docker_image("cr.nemax.nebius.cloud/crnf2coti090683j5ssi/gritukan_ml:5")
            .environment({"YT_ALLOW_HTTP_REQUESTS_TO_YT_FROM_JOB": "1"})
            .file_paths()
        .end_task()
        .max_failed_job_count(1)
)

