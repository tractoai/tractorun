import sys
import time

import yt.wrapper as yt


def main() -> None:
    path = sys.argv[1]
    value = None
    while value is None:
        value = yt.get(path)
        print(f"sidecar reads {value}", file=sys.stderr)
        yt.set(path, value + 1)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
