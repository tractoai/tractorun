import argparse
import time
from typing import Callable

import yt.wrapper as yt
from yt.wrapper.errors import YtResolveError

from tractorun.private.stderr_reader import (
    YtStderrReader,
    get_job_stderr_with_retry,
)
from tractorun.private.training_dir import TrainingDir


def main() -> None:
    parser = argparse.ArgumentParser(description="Get the stderr of a tractorun peer")
    parser.add_argument(
        "training_dir", metavar="training-dir", type=str, help="The path to the training root directory"
    )
    parser.add_argument(
        "--incarnation",
        type=int,
        help="Number of the incarnation to read stderr. If not set, last incarnation will be used.",
    )
    parser.add_argument(
        "--peer-index",
        type=int,
        default=0,
        help="Index of the peer to read stderr. If not set, primary peer with index 0 will be used.",
    )
    parser.add_argument("-f", "--follow", action="store_true", help="Follow the stderr output")
    args = parser.parse_args()

    training_dir = TrainingDir.create(args.training_dir)
    yt_client = yt.YtClient(config=yt.default_config.get_config_from_env())

    incarnation = args.incarnation
    if incarnation is None:
        try:
            incarnation = yt_client.get(training_dir.base_path + "/@incarnation_id")
        except YtResolveError as e:
            raise Exception("Training dir does not have @incarnation_id attr") from e
        assert incarnation != -1, "No incarnations found"

    incarnation_path = training_dir.get_incarnation_path(incarnation_id=incarnation)
    try:
        yt_client.get(incarnation_path)
    except YtResolveError as e:
        raise Exception(f"Incarnation path is present on cluster {incarnation_path}") from e

    operation_id = yt_client.get(incarnation_path + "/@incarnation_operation_id")
    job_id = yt_client.get(incarnation_path + f"/@topology/{args.peer_index}/job_id")

    stderr_getter = get_job_stderr_with_retry(yt_client=yt_client, operation_id=operation_id, job_id=job_id)

    if args.follow:
        reader = YtStderrReader(stderr_getter=stderr_getter)
        try:
            for line in reader:
                print(line.decode("unicode_escape"), end="")
                time.sleep(1.0)
        except Exception as e:
            print()
            print(e)
    else:
        print_output(stderr_getter)


def print_output(generator: Callable[[], bytes]) -> None:
    stderr = generator().decode("unicode_escape")
    print(stderr)
