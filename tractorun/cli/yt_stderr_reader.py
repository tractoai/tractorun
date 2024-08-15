import argparse
import time
from typing import Callable

import yt.wrapper as yt
from yt.wrapper.errors import YtResolveError

from tractorun.stderr_reader import YtStderrReader
from tractorun.training_dir import TrainingDir


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

    incarnation = args.incarnation
    if incarnation is None:
        incarnation = -1
        for inc_path in yt.list(training_dir.incarnations_path):
            try:
                incarnation = max(incarnation, int(inc_path))
            except ValueError:
                pass
        assert incarnation != -1, "No incarnations found"

    incarnation_path = training_dir.get_incarnation_path(incarnation_id=incarnation)
    try:
        yt.get(incarnation_path)
    except YtResolveError as e:
        raise Exception(f"Incarnation path is present on cluster {incarnation_path}") from e

    operation_id = yt.get(incarnation_path + "/@incarnation_operation_id")
    job_id = yt.get(incarnation_path + f"/@topology/{args.peer_index}/job_id")

    stderr_getter = get_job_stderr(operation_id=operation_id, job_id=job_id)

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


def get_job_stderr(operation_id: str, job_id: str) -> Callable[[], bytes]:
    def _wrapped() -> bytes:
        data = yt.get_job_stderr(operation_id=operation_id, job_id=job_id).read()
        assert isinstance(data, bytes)
        return data

    return _wrapped


def print_output(generator: Callable[[], bytes]) -> None:
    stderr = generator().decode("unicode_escape")
    print(stderr)
