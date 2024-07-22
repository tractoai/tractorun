import argparse
import time
from typing import Callable

import yt.wrapper as yt


def tail_output(generator: Callable[[], bytes], polling_interval: float) -> None:
    def prefix_function(s: bytes) -> list[int]:
        pi = [0] * len(s)
        for i in range(1, len(s)):
            j = pi[i - 1]
            while j > 0 and s[i] != s[j]:
                j = pi[j - 1]
            if s[i] == s[j]:
                j += 1
            pi[i] = j
        return pi

    def get_new_data(s: bytes, t: bytes) -> bytes:
        uuid = b"8a5c221d-c111d561-13440384-186"
        k = t + uuid + s
        pi = prefix_function(k)
        max_pi = 0
        for i in range(len(t) + len(uuid), len(k)):
            max_pi = max(max_pi, pi[i])
        return t[max_pi:]

    last = b""
    try:
        while True:
            current = generator()
            new_data = get_new_data(last, current)
            print(new_data.decode("unicode_escape"), end="")
            last = current

            time.sleep(polling_interval)
    except Exception as e:
        # Print a newline to deal with special symbols in output.
        print()
        print(e)


def print_output(generator: Callable[[], bytes]) -> None:
    stderr = generator().decode("unicode_escape")
    print(stderr)


parser = argparse.ArgumentParser(description="Get the stderr of a tractorun peer")
parser.add_argument("training_root", metavar="training-root", type=str, help="The path to the training root directory")
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
if args.incarnation is None:
    incarnation = -1
    for dir in yt.list(args.training_root + "/incarnations"):
        try:
            incarnation = max(incarnation, int(dir))
        except ValueError:
            pass
    assert incarnation != -1, "No incarnations found"
else:
    incarnation = args.incarnation

incarnation_dir = None
for dir in yt.list(args.training_root + "/incarnations"):
    try:
        if int(dir) == incarnation:
            incarnation_dir = dir
            break
    except ValueError:
        pass
assert incarnation_dir is not None, "Incarnation not found"

operation_id = yt.get(args.training_root + "/incarnations/" + incarnation_dir + "/@incarnation_operation_id")
job_id = yt.get(args.training_root + "/incarnations/" + incarnation_dir + f"/@topology/{args.peer_index}/job_id")


def _gen() -> bytes:
    return yt.get_job_stderr(operation_id=operation_id, job_id=job_id).read()


if args.follow:
    tail_output(_gen, 1.0)
else:
    print_output(_gen)
